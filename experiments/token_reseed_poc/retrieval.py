import logging
from dataclasses import dataclass

import asyncpg
import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _parse_pg_vector(text: str) -> list[float]:
    """Parse PostgreSQL vector text representation '[0.1,0.2,...]' to list of floats."""
    return [float(v) for v in text.strip("[]").split(",")]


def _compute_stability(prev_top_k: list[str], curr_top_k: list[str]) -> float:
    """Fraction of IDs in prev_top_k that also appear in curr_top_k."""
    if not prev_top_k:
        return 0.0
    overlap = len(set(prev_top_k) & set(curr_top_k))
    return overlap / len(prev_top_k)


@dataclass
class RetrievalResult:
    trace_id: str
    content: str
    score: float
    source: str  # "vector", "token", "token+vector", "token_hop_N"
    token_label: str = ""
    hop_depth: int = 0  # 0 = vector, 1 = one-hop token, 2+ = iterative


async def baseline_recall(
    pool: asyncpg.Pool,
    embedding: list[float],
    top_k: int = 20,
    significance_weight: float = 0.15,
    valence_weight: float = 0.05,
) -> list[RetrievalResult]:
    """Pure vector search only."""
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, significance, emotional_valence,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM poc_traces
            ORDER BY similarity DESC
            LIMIT $2
            """,
            embedding_str,
            top_k,
        )

    results = []
    for row in rows:
        cosine_sim = float(row["similarity"])
        sig_boost = float(row["significance"]) * significance_weight
        val_boost = abs(float(row["emotional_valence"])) * valence_weight
        score = min(cosine_sim + sig_boost + val_boost, 1.0)
        results.append(
            RetrievalResult(
                trace_id=str(row["id"]),
                content=row["content"],
                score=score,
                source="vector",
                hop_depth=0,
            )
        )
    logger.info("Baseline recall returned %d results", len(results))
    return results


async def token_recall(  # noqa: C901
    pool: asyncpg.Pool,
    embedding: list[float],
    top_k: int = 20,
    strength_threshold: float = 0.1,
    reinforce_boost: float = 0.1,
    hop_decay: float = 0.85,
    significance_weight: float = 0.15,
    valence_weight: float = 0.05,
    propagation_blend: float = 0.5,
) -> list[RetrievalResult]:
    """Vector search + token activation + reinforcement."""
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    # Step 1: Vector search
    async with pool.acquire() as conn:
        vector_rows = await conn.fetch(
            """
            SELECT id, content, significance, emotional_valence,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM poc_traces
            ORDER BY similarity DESC
            LIMIT $2
            """,
            embedding_str,
            top_k,
        )

    vector_results: dict[str, RetrievalResult] = {}
    vector_trace_ids: list[str] = []
    # Store per-trace metadata: (base_sim, significance, valence)
    trace_meta: dict[str, tuple[float, float, float]] = {}

    for row in vector_rows:
        tid = str(row["id"])
        vector_trace_ids.append(tid)
        cosine_sim = float(row["similarity"])
        sig = float(row["significance"])
        val = float(row["emotional_valence"])
        trace_meta[tid] = (cosine_sim, sig, val)
        sig_boost = sig * significance_weight
        val_boost = abs(val) * valence_weight
        score = min(cosine_sim + sig_boost + val_boost, 1.0)
        vector_results[tid] = RetrievalResult(
            trace_id=tid,
            content=row["content"],
            score=score,
            source="vector",
            hop_depth=0,
        )

    if not vector_trace_ids:
        return []

    # Step 2: Token activation -- join with seed_stamps to get anchor_id
    async with pool.acquire() as conn:
        token_rows = await conn.fetch(
            """
            SELECT t.trace_id, rt.id AS token_id,
                   rt.label, rt.strength, rt.significance,
                   seed_stamps.trace_id AS anchor_id
            FROM poc_token_stamps t
            JOIN poc_recall_tokens rt ON rt.id = t.token_id
            JOIN poc_token_stamps seed_stamps
                ON seed_stamps.token_id = t.token_id
                AND seed_stamps.trace_id = ANY($1::uuid[])
            WHERE t.trace_id != ALL($1::uuid[])
            AND rt.strength > $2
            ORDER BY rt.strength DESC
            """,
            vector_trace_ids,
            strength_threshold,
        )

    activated_token_ids: set[str] = set()
    # Best propagated info per trace:
    #   {tid: (propagated_sim, token_strength, label, significance)}
    token_activated_info: dict[str, tuple[float, float, str, float]] = {}

    for row in token_rows:
        tid = str(row["trace_id"])
        activated_token_ids.add(str(row["token_id"]))
        anchor_id = str(row["anchor_id"])
        anchor_cosine = trace_meta[anchor_id][0]
        token_strength = float(row["strength"])
        token_significance = float(row["significance"])
        propagated = (
            anchor_cosine * hop_decay * token_strength * token_significance
        )
        existing = token_activated_info.get(tid)
        if existing is None or propagated > existing[0]:
            token_activated_info[tid] = (
                propagated, token_strength, row["label"],
                token_significance,
            )

    # Build initial RetrievalResult stubs for token-activated traces
    token_activated: dict[str, RetrievalResult] = {}
    for tid, (prop_sim, _tok_str, label, _tok_sig) in token_activated_info.items():
        token_activated[tid] = RetrievalResult(
            trace_id=tid,
            content="",
            score=prop_sim,
            source="token",
            token_label=label,
            hop_depth=1,
        )

    # Fetch content + embedding + metadata for token-only traces
    if token_activated:
        missing_ids = [tid for tid in token_activated if tid not in vector_results]
        if missing_ids:
            async with pool.acquire() as conn:
                content_rows = await conn.fetch(
                    "SELECT id, content, embedding::text,"
                    " significance, emotional_valence"
                    " FROM poc_traces"
                    " WHERE id = ANY($1::uuid[])",
                    missing_ids,
                )
            for row in content_rows:
                tid = str(row["id"])
                if tid in token_activated:
                    token_activated[tid].content = row["content"]
                    trace_emb = _parse_pg_vector(row["embedding"])
                    cosine_sim = _cosine_similarity(embedding, trace_emb)
                    sig = float(row["significance"])
                    val = float(row["emotional_valence"])
                    trace_meta[tid] = (cosine_sim, sig, val)
                    propagated_sim = token_activated_info[tid][0]
                    effective_sim = cosine_sim + propagated_sim * propagation_blend
                    sig_boost = sig * significance_weight
                    val_boost = abs(val) * valence_weight
                    score = min(effective_sim + sig_boost + val_boost, 1.0)
                    token_activated[tid].score = score
                    logger.debug(
                        "Token-activated %s:"
                        " cosine=%.3f,"
                        " propagated=%.3f,"
                        " effective=%.3f,"
                        " sig=%.3f, val=%.3f,"
                        " score=%.3f",
                        tid,
                        cosine_sim,
                        propagated_sim,
                        effective_sim,
                        sig,
                        val,
                        score,
                    )

    # Step 3: Reinforce activated tokens
    if activated_token_ids:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE poc_recall_tokens
                SET strength = LEAST(1.0, strength + $2),
                    last_activated_at = now()
                WHERE id = ANY($1::uuid[])
                """,
                list(activated_token_ids),
                reinforce_boost,
            )
        logger.info(
            "Reinforced %d tokens by %.2f",
            len(activated_token_ids),
            reinforce_boost,
        )

    # Step 4: Merge -- boost overlaps, append token-only
    for tid, token_result in token_activated.items():
        if tid in vector_results:
            existing = vector_results[tid]
            base_sim, sig, val = trace_meta[tid]
            propagated_sim = token_activated_info[tid][0]
            effective_sim = base_sim + propagated_sim * propagation_blend
            sig_boost = sig * significance_weight
            val_boost = abs(val) * valence_weight
            boosted = min(effective_sim + sig_boost + val_boost, 1.0)
            existing.score = boosted
            existing.source = "token+vector"
            existing.token_label = token_result.token_label
            logger.debug(
                "Overlap %s: base_sim=%.3f, propagated=%.3f, boosted=%.3f",
                tid,
                base_sim,
                propagated_sim,
                boosted,
            )

    merged: list[RetrievalResult] = list(vector_results.values())
    for tid, result in token_activated.items():
        if tid not in vector_results:
            merged.append(result)

    merged.sort(key=lambda r: r.score, reverse=True)
    merged = merged[:top_k]

    logger.info(
        "Token recall returned %d results"
        " (%d from pool of %d vector + %d token-activated)",
        len(merged),
        top_k,
        len(vector_results),
        len(token_activated),
    )
    return merged


async def iterative_token_recall(  # noqa: C901
    pool: asyncpg.Pool,
    embedding: list[float],
    top_k: int = 20,
    strength_threshold: float = 0.1,
    reinforce_boost: float = 0.1,
    hop_decay: float = 0.85,
    max_rounds: int = 3,
    stability_threshold: float = 0.95,
    top_seeds: int = 3,
    significance_weight: float = 0.15,
    valence_weight: float = 0.05,
    propagation_blend: float = 0.5,
) -> tuple[list[RetrievalResult], int]:
    """Vector search + iterative token re-seeding.

    Each round discovers new traces via tokens stamped on the previous
    round's top results.  Stops when the top-K ranking stabilizes or
    max_rounds is reached.

    Returns (results, rounds_executed).
    """
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    # Per-trace metadata: (base_sim, significance, valence)
    trace_meta: dict[str, tuple[float, float, float]] = {}

    # -- Round 1: vector search ------------------------------------------

    async with pool.acquire() as conn:
        vector_rows = await conn.fetch(
            """
            SELECT id, content, significance, emotional_valence,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM poc_traces
            ORDER BY similarity DESC
            LIMIT $2
            """,
            embedding_str,
            top_k,
        )

    all_known: dict[str, RetrievalResult] = {}
    vector_trace_ids: list[str] = []
    # Track best propagated similarity for each trace across all hops
    propagated_sims: dict[str, float] = {}

    for row in vector_rows:
        tid = str(row["id"])
        vector_trace_ids.append(tid)
        cosine_sim = float(row["similarity"])
        sig = float(row["significance"])
        val = float(row["emotional_valence"])
        trace_meta[tid] = (cosine_sim, sig, val)
        propagated_sims[tid] = cosine_sim  # vector results anchor themselves
        sig_boost = sig * significance_weight
        val_boost = abs(val) * valence_weight
        score = min(cosine_sim + sig_boost + val_boost, 1.0)
        all_known[tid] = RetrievalResult(
            trace_id=tid,
            content=row["content"],
            score=score,
            source="vector",
            hop_depth=0,
        )

    if not vector_trace_ids:
        return [], 1

    # -- Round 1: one-hop token activation --------------------------------

    seed_ids = list(vector_trace_ids)
    exclude_ids = list(vector_trace_ids)

    async with pool.acquire() as conn:
        token_rows = await conn.fetch(
            """
            SELECT t.trace_id, rt.id AS token_id,
                   rt.label, rt.strength, rt.significance,
                   seed_stamps.trace_id AS anchor_id
            FROM poc_token_stamps t
            JOIN poc_recall_tokens rt ON rt.id = t.token_id
            JOIN poc_token_stamps seed_stamps
                ON seed_stamps.token_id = t.token_id
                AND seed_stamps.trace_id = ANY($1::uuid[])
            WHERE t.trace_id != ALL($2::uuid[])
            AND rt.strength > $3
            ORDER BY rt.strength DESC
            """,
            seed_ids,
            exclude_ids,
            strength_threshold,
        )

    # Best propagated info per trace:
    #   {tid: (propagated_sim, token_strength, label, significance)}
    round1_info: dict[str, tuple[float, float, str, float]] = {}
    for row in token_rows:
        tid = str(row["trace_id"])
        anchor_id = str(row["anchor_id"])
        anchor_cosine = propagated_sims[anchor_id]
        token_strength = float(row["strength"])
        token_significance = float(row["significance"])
        propagated = (
            anchor_cosine * hop_decay * token_strength * token_significance
        )
        existing = round1_info.get(tid)
        if existing is None or propagated > existing[0]:
            round1_info[tid] = (
                propagated, token_strength, row["label"],
                token_significance,
            )

    if round1_info:
        new_ids = list(round1_info.keys())
        async with pool.acquire() as conn:
            content_rows = await conn.fetch(
                "SELECT id, content, embedding::text,"
                " significance, emotional_valence"
                " FROM poc_traces"
                " WHERE id = ANY($1::uuid[])",
                new_ids,
            )
        for row in content_rows:
            tid = str(row["id"])
            prop_sim, _tok_str, label, _tok_sig = round1_info[tid]
            trace_emb = _parse_pg_vector(row["embedding"])
            cosine_sim = _cosine_similarity(embedding, trace_emb)
            sig = float(row["significance"])
            val = float(row["emotional_valence"])
            trace_meta[tid] = (cosine_sim, sig, val)
            propagated_sims[tid] = prop_sim
            effective_sim = cosine_sim + prop_sim * propagation_blend
            sig_boost = sig * significance_weight
            val_boost = abs(val) * valence_weight
            score = min(effective_sim + sig_boost + val_boost, 1.0)
            all_known[tid] = RetrievalResult(
                trace_id=tid,
                content=row["content"],
                score=score,
                source="token_hop_1",
                token_label=label,
                hop_depth=1,
            )

    used_seeds: set[str] = set(vector_trace_ids)

    sorted_pool = sorted(all_known.values(), key=lambda r: r.score, reverse=True)
    prev_top_k_ids = [r.trace_id for r in sorted_pool[:top_k]]

    rounds_executed = 1

    # -- Rounds 2..max_rounds: iterative re-seeding ----------------------

    for round_num in range(2, max_rounds + 1):
        token_discovered = [
            r for r in sorted_pool if r.hop_depth >= 1 and r.trace_id not in used_seeds
        ]
        new_seed_ids = [r.trace_id for r in token_discovered[:top_seeds]]

        if not new_seed_ids:
            logger.info(
                "Round %d: no new seeds available, stopping",
                round_num,
            )
            break

        used_seeds.update(new_seed_ids)
        all_known_ids = list(all_known.keys())

        async with pool.acquire() as conn:
            hop_rows = await conn.fetch(
                """
                SELECT t.trace_id, rt.id AS token_id,
                       rt.label, rt.strength, rt.significance,
                       seed_stamps.trace_id AS anchor_id
                FROM poc_token_stamps t
                JOIN poc_recall_tokens rt ON rt.id = t.token_id
                JOIN poc_token_stamps seed_stamps
                    ON seed_stamps.token_id = t.token_id
                    AND seed_stamps.trace_id = ANY($1::uuid[])
                WHERE t.trace_id != ALL($2::uuid[])
                AND rt.strength > $3
                ORDER BY rt.strength DESC
                """,
                new_seed_ids,
                all_known_ids,
                strength_threshold,
            )

        # Best propagated info:
        #   {tid: (propagated_sim, token_strength, label, significance)}
        round_info: dict[str, tuple[float, float, str, float]] = {}
        for row in hop_rows:
            tid = str(row["trace_id"])
            anchor_id = str(row["anchor_id"])
            seed_propagated = propagated_sims.get(anchor_id, 0.0)
            token_strength = float(row["strength"])
            token_significance = float(row["significance"])
            propagated = (
                seed_propagated * hop_decay
                * token_strength * token_significance
            )
            existing = round_info.get(tid)
            if existing is None or propagated > existing[0]:
                round_info[tid] = (
                    propagated, token_strength, row["label"],
                    token_significance,
                )

        if round_info:
            new_ids = list(round_info.keys())
            async with pool.acquire() as conn:
                content_rows = await conn.fetch(
                    "SELECT id, content, embedding::text,"
                    " significance, emotional_valence"
                    " FROM poc_traces"
                    " WHERE id = ANY($1::uuid[])",
                    new_ids,
                )
            for row in content_rows:
                tid = str(row["id"])
                prop_sim, _tok_str, label, _tok_sig = round_info[tid]
                trace_emb = _parse_pg_vector(row["embedding"])
                cosine_sim = _cosine_similarity(embedding, trace_emb)
                sig = float(row["significance"])
                val = float(row["emotional_valence"])
                trace_meta[tid] = (cosine_sim, sig, val)
                propagated_sims[tid] = prop_sim
                effective_sim = cosine_sim + prop_sim * propagation_blend
                sig_boost = sig * significance_weight
                val_boost = abs(val) * valence_weight
                score = min(effective_sim + sig_boost + val_boost, 1.0)
                all_known[tid] = RetrievalResult(
                    trace_id=tid,
                    content=row["content"],
                    score=score,
                    source=f"token_hop_{round_num}",
                    token_label=label,
                    hop_depth=round_num,
                )

        rounds_executed = round_num

        sorted_pool = sorted(
            all_known.values(),
            key=lambda r: r.score,
            reverse=True,
        )
        curr_top_k_ids = [r.trace_id for r in sorted_pool[:top_k]]
        stability = _compute_stability(prev_top_k_ids, curr_top_k_ids)

        logger.info(
            "Round %d: %d new traces, stability=%.3f (%d/%d overlap)",
            round_num,
            len(round_info),
            stability,
            len(set(prev_top_k_ids) & set(curr_top_k_ids)),
            len(prev_top_k_ids),
        )

        if stability >= stability_threshold:
            logger.info(
                "Round %d: stability %.3f >= %.3f, stopping",
                round_num,
                stability,
                stability_threshold,
            )
            break

        prev_top_k_ids = curr_top_k_ids

    final = sorted(all_known.values(), key=lambda r: r.score, reverse=True)
    final = final[:top_k]

    logger.info(
        "Iterative token recall returned %d results"
        " (%d total discovered across %d rounds)",
        len(final),
        len(all_known),
        rounds_executed,
    )
    return final, rounds_executed
