import logging
from dataclasses import dataclass

import asyncpg

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_pg_vector(text: str) -> list[float]:
    """Parse PostgreSQL vector text representation '[0.1,0.2,...]' to list of floats."""
    return [float(v) for v in text.strip("[]").split(",")]


async def _fetch_embeddings(
    conn: asyncpg.Connection,
    trace_ids: list[str],
) -> dict[str, list[float]]:
    """Fetch embeddings for a set of trace IDs, returned as a dict."""
    if not trace_ids:
        return {}
    rows = await conn.fetch(
        "SELECT id, embedding::text FROM poc_iter_traces WHERE id = ANY($1::uuid[])",
        trace_ids,
    )
    return {str(row["id"]): _parse_pg_vector(row["embedding"]) for row in rows}


DEFAULT_SPREAD_WEIGHT = 0.1


@dataclass
class RetrievalResult:
    trace_id: str
    content: str
    score: float
    source: str  # "vector", "spread", "iterative"
    hop_depth: int = 0  # 0 for vector, N for spread-discovered


async def _spread_from_seed(
    conn: asyncpg.Connection,
    seed_id: str,
    decay: float,
    threshold: float,
    max_depth: int,
) -> list[tuple[str, str, float, int]]:
    """Run recursive CTE from a single seed.

    Returns list of (id, content, activation_level, depth).
    """
    rows = await conn.fetch(
        """
        WITH RECURSIVE activation AS (
            SELECT id, content,
                   1.0::float AS activation_level,
                   0 AS depth
            FROM poc_iter_traces WHERE id = $1

            UNION ALL

            SELECT mt.id, mt.content,
                   (a.activation_level * $2 *
                    CASE WHEN assoc.source_trace_id = a.id
                         THEN assoc.forward_strength
                         ELSE assoc.backward_strength
                    END)::float,
                   a.depth + 1
            FROM activation a
            JOIN poc_iter_associations assoc
                ON assoc.source_trace_id = a.id
                OR assoc.target_trace_id = a.id
            JOIN poc_iter_traces mt
                ON mt.id = CASE
                    WHEN assoc.source_trace_id = a.id
                    THEN assoc.target_trace_id
                    ELSE assoc.source_trace_id
                END
            WHERE a.depth < $3
              AND a.activation_level * $2 *
                  CASE WHEN assoc.source_trace_id = a.id
                       THEN assoc.forward_strength
                       ELSE assoc.backward_strength
                  END > $4
        )
        SELECT DISTINCT ON (id) id, content, activation_level, depth
        FROM activation
        WHERE id != $1
        ORDER BY id, activation_level DESC
        """,
        seed_id,
        decay,
        max_depth,
        threshold,
    )
    return [
        (
            str(row["id"]),
            row["content"],
            float(row["activation_level"]),
            int(row["depth"]),
        )
        for row in rows
    ]


async def baseline_recall(
    pool: asyncpg.Pool,
    embedding: list[float],
    top_k: int = 10,
) -> list[RetrievalResult]:
    """Pure vector search, no spreading activation."""
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
            FROM poc_iter_traces
            ORDER BY similarity DESC
            LIMIT $2
            """,
            embedding_str,
            top_k,
        )

    results = [
        RetrievalResult(
            trace_id=str(row["id"]),
            content=row["content"],
            score=float(row["similarity"]),
            source="vector",
        )
        for row in rows
    ]
    logger.info("Baseline recall returned %d results", len(results))
    return results


async def _spread_and_collect(
    conn: asyncpg.Connection,
    seed_ids: list[str],
    decay: float,
    threshold: float,
    max_depth: int,
) -> dict[str, tuple[str, float, int]]:
    """Spread from multiple seeds.

    Returns {trace_id: (content, best_activation, depth)}.
    """
    spread_pool: dict[str, tuple[str, float, int]] = {}
    for seed_id in seed_ids:
        activated = await _spread_from_seed(conn, seed_id, decay, threshold, max_depth)
        for tid, content, activation, depth in activated:
            if tid not in spread_pool or activation > spread_pool[tid][1]:
                spread_pool[tid] = (content, activation, depth)
    return spread_pool


def _merge_results(
    vector_results: dict[str, RetrievalResult],
    spread_pool: dict[str, tuple[str, float, int]],
    query_embedding: list[float],
    trace_embeddings: dict[str, list[float]],
    source_label: str,
    spread_weight: float = DEFAULT_SPREAD_WEIGHT,
) -> dict[str, RetrievalResult]:
    """Merge vector and spread results with cosine sim for spread-only traces.

    Spread-only traces: score = cosine_sim(query, trace) + activation * spread_weight.
    Overlap traces: score = vector_score + activation * spread_weight.
    All scores clamped to [0, 1].
    """
    merged = dict(vector_results)

    for tid, (content, activation, depth) in spread_pool.items():
        if tid in merged:
            # Overlap: add activation bonus to existing vector score
            existing = merged[tid]
            existing.score = min(
                existing.score + activation * spread_weight,
                1.0,
            )
            existing.source = source_label
            existing.hop_depth = depth
        else:
            # Spread-only: compute actual cosine similarity as base
            if tid in trace_embeddings:
                cosine_sim = _cosine_similarity(query_embedding, trace_embeddings[tid])
            else:
                cosine_sim = 0.0
            spread_score = cosine_sim + activation * spread_weight
            merged[tid] = RetrievalResult(
                trace_id=tid,
                content=content,
                score=min(spread_score, 1.0),
                source=source_label,
                hop_depth=depth,
            )

    return merged


def _top_k_ids(merged: dict[str, RetrievalResult], k: int) -> set[str]:
    """Return the top-k trace IDs by score."""
    sorted_items = sorted(merged.values(), key=lambda r: r.score, reverse=True)
    return {r.trace_id for r in sorted_items[:k]}


async def fixed_spread_recall(
    pool: asyncpg.Pool,
    embedding: list[float],
    top_k: int = 10,
    spread_decay: float = 0.7,
    spread_threshold: float = 0.1,
    spread_max_depth: int = 2,
    num_seeds: int = 3,
    spread_weight: float = DEFAULT_SPREAD_WEIGHT,
) -> list[RetrievalResult]:
    """Vector search + fixed-depth spreading activation."""
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    async with pool.acquire() as conn:
        # Step 1: Vector search
        vector_rows = await conn.fetch(
            """
            SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
            FROM poc_iter_traces
            ORDER BY similarity DESC
            LIMIT $2
            """,
            embedding_str,
            top_k,
        )

        vector_results: dict[str, RetrievalResult] = {}
        for row in vector_rows:
            tid = str(row["id"])
            vector_results[tid] = RetrievalResult(
                trace_id=tid,
                content=row["content"],
                score=float(row["similarity"]),
                source="vector",
            )

        if not vector_results:
            return []

        # Step 2: Spread from top seeds
        seed_ids = [
            r.trace_id
            for r in sorted(
                vector_results.values(),
                key=lambda r: r.score,
                reverse=True,
            )
        ][:num_seeds]

        spread_pool = await _spread_and_collect(
            conn,
            seed_ids,
            spread_decay,
            spread_threshold,
            spread_max_depth,
        )

        # Step 3: Fetch embeddings for spread-only traces
        spread_only_ids = [tid for tid in spread_pool if tid not in vector_results]
        trace_embeddings = await _fetch_embeddings(conn, spread_only_ids)

        # Step 4: Merge and score
        merged = _merge_results(
            vector_results,
            spread_pool,
            embedding,
            trace_embeddings,
            "spread",
            spread_weight=spread_weight,
        )

    # Step 5: Sort and truncate
    results = sorted(merged.values(), key=lambda r: r.score, reverse=True)[:top_k]
    logger.info(
        "Fixed spread recall returned %d results (%d vector, %d spread-discovered)",
        len(results),
        len(vector_results),
        len(spread_pool),
    )
    return results


async def iterative_recall(  # noqa: C901
    pool: asyncpg.Pool,
    embedding: list[float],
    top_k: int = 10,
    spread_decay: float = 0.7,
    spread_threshold: float = 0.1,
    spread_max_depth: int = 2,
    num_seeds: int = 3,
    max_rounds: int = 3,
    stability_threshold: float = 0.95,
    spread_weight: float = DEFAULT_SPREAD_WEIGHT,
) -> tuple[list[RetrievalResult], int]:
    """Vector search + iterative spreading activation with re-seeding.

    Returns (results, num_rounds_executed).
    """
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    async with pool.acquire() as conn:
        # Initial vector search
        vector_rows = await conn.fetch(
            """
            SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
            FROM poc_iter_traces
            ORDER BY similarity DESC
            LIMIT $2
            """,
            embedding_str,
            top_k,
        )

        vector_results: dict[str, RetrievalResult] = {}
        for row in vector_rows:
            tid = str(row["id"])
            vector_results[tid] = RetrievalResult(
                trace_id=tid,
                content=row["content"],
                score=float(row["similarity"]),
                source="vector",
            )

        if not vector_results:
            return [], 0

        # Initial spread from top seeds
        seed_ids = [
            r.trace_id
            for r in sorted(
                vector_results.values(),
                key=lambda r: r.score,
                reverse=True,
            )
        ][:num_seeds]

        seen_seeds: set[str] = set(seed_ids)
        all_spread: dict[str, tuple[str, float, int]] = {}

        spread_pool = await _spread_and_collect(
            conn,
            seed_ids,
            spread_decay,
            spread_threshold,
            spread_max_depth,
        )
        all_spread.update(spread_pool)

        # Fetch embeddings for spread-only traces
        spread_only_ids = [tid for tid in all_spread if tid not in vector_results]
        all_embeddings = await _fetch_embeddings(conn, spread_only_ids)

        # Merge initial results
        merged = _merge_results(
            vector_results,
            all_spread,
            embedding,
            all_embeddings,
            "iterative",
            spread_weight=spread_weight,
        )
        previous_top_k = _top_k_ids(merged, top_k)
        rounds_executed = 1

        # Re-seed loop
        for round_num in range(2, max_rounds + 1):
            # Pick top-scoring traces not yet used as seeds
            current_ranked = sorted(
                merged.values(),
                key=lambda r: r.score,
                reverse=True,
            )
            new_seed_ids = []
            for r in current_ranked:
                if r.trace_id not in seen_seeds:
                    new_seed_ids.append(r.trace_id)
                    if len(new_seed_ids) >= num_seeds:
                        break

            if not new_seed_ids:
                logger.info("Round %d: no new seeds available, stopping", round_num)
                break

            seen_seeds.update(new_seed_ids)

            # Spread from new seeds
            new_spread = await _spread_and_collect(
                conn,
                new_seed_ids,
                spread_decay,
                spread_threshold,
                spread_max_depth,
            )

            # Update spread pool with better activations
            for tid, val in new_spread.items():
                if tid not in all_spread or val[1] > all_spread[tid][1]:
                    all_spread[tid] = val

            # Fetch embeddings for newly discovered traces
            new_trace_ids = [
                tid
                for tid in new_spread
                if tid not in all_embeddings and tid not in vector_results
            ]
            new_embeddings = await _fetch_embeddings(conn, new_trace_ids)
            all_embeddings.update(new_embeddings)

            # Re-merge everything
            merged = _merge_results(
                vector_results,
                all_spread,
                embedding,
                all_embeddings,
                "iterative",
                spread_weight=spread_weight,
            )
            rounds_executed = round_num

            # Stability check
            current_top_k = _top_k_ids(merged, top_k)
            overlap = len(current_top_k & previous_top_k)
            overlap_ratio = overlap / top_k if top_k > 0 else 1.0

            logger.info(
                "Round %d: %d new seeds, %d new traces, overlap=%.2f",
                round_num,
                len(new_seed_ids),
                len(new_spread),
                overlap_ratio,
            )

            if overlap_ratio >= stability_threshold:
                logger.info(
                    "Round %d: ranking stabilized (%.2f >= %.2f), stopping",
                    round_num,
                    overlap_ratio,
                    stability_threshold,
                )
                break

            previous_top_k = current_top_k

    results = sorted(merged.values(), key=lambda r: r.score, reverse=True)[:top_k]
    logger.info(
        "Iterative recall returned %d results after %d rounds"
        " (%d vector, %d spread-discovered, %d seeds used)",
        len(results),
        rounds_executed,
        len(vector_results),
        len(all_spread),
        len(seen_seeds),
    )
    return results, rounds_executed
