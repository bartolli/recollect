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


@dataclass
class RetrievalResult:
    trace_id: str
    content: str
    score: float
    source: str  # "vector" or "token"
    token_label: str = ""


async def baseline_recall(
    pool: asyncpg.Pool,
    embedding: list[float],
    top_k: int = 20,
) -> list[RetrievalResult]:
    """Pure vector search only."""
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
            FROM poc_traces
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


async def token_recall(  # noqa: C901
    pool: asyncpg.Pool,
    embedding: list[float],
    top_k: int = 20,
    strength_threshold: float = 0.1,
    reinforce_boost: float = 0.1,
    token_bonus: float = 0.1,
) -> list[RetrievalResult]:
    """Vector search + token activation + reinforcement."""
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    # Step 1: Vector search
    async with pool.acquire() as conn:
        vector_rows = await conn.fetch(
            """
            SELECT id, content, 1 - (embedding <=> $1::vector) AS similarity
            FROM poc_traces
            ORDER BY similarity DESC
            LIMIT $2
            """,
            embedding_str,
            top_k,
        )

    vector_results: dict[str, RetrievalResult] = {}
    vector_trace_ids = []
    for row in vector_rows:
        tid = str(row["id"])
        vector_trace_ids.append(tid)
        vector_results[tid] = RetrievalResult(
            trace_id=tid,
            content=row["content"],
            score=float(row["similarity"]),
            source="vector",
        )

    if not vector_trace_ids:
        return []

    # Step 2: Token activation -- find traces linked via tokens to vector results
    async with pool.acquire() as conn:
        token_rows = await conn.fetch(
            """
            SELECT DISTINCT t.trace_id, rt.id AS token_id, rt.label, rt.strength
            FROM poc_token_stamps t
            JOIN poc_recall_tokens rt ON rt.id = t.token_id
            WHERE t.token_id IN (
                SELECT token_id FROM poc_token_stamps WHERE trace_id = ANY($1::uuid[])
            )
            AND t.trace_id != ALL($1::uuid[])
            AND rt.strength > $2
            ORDER BY rt.strength DESC
            """,
            vector_trace_ids,
            strength_threshold,
        )

    # Collect token IDs for reinforcement
    activated_token_ids: set[str] = set()
    token_activated: dict[str, RetrievalResult] = {}

    for row in token_rows:
        tid = str(row["trace_id"])
        activated_token_ids.add(str(row["token_id"]))
        if tid not in token_activated:
            # Need to fetch content for token-activated traces
            token_activated[tid] = RetrievalResult(
                trace_id=tid,
                content="",  # will fill below
                score=float(row["strength"]),
                source="token",
                token_label=row["label"],
            )

    # Fetch content + embedding for token-activated traces not
    # already in vector results, then compute gated score:
    # vector_sim + token_strength * token_bonus * vector_sim
    if token_activated:
        missing_ids = [tid for tid in token_activated if tid not in vector_results]
        if missing_ids:
            async with pool.acquire() as conn:
                content_rows = await conn.fetch(
                    "SELECT id, content, embedding::text"
                    " FROM poc_traces"
                    " WHERE id = ANY($1::uuid[])",
                    missing_ids,
                )
            for row in content_rows:
                tid = str(row["id"])
                if tid in token_activated:
                    token_activated[tid].content = row["content"]
                    trace_embedding = _parse_pg_vector(row["embedding"])
                    vector_sim = _cosine_similarity(embedding, trace_embedding)
                    token_strength = token_activated[tid].score
                    bonus = token_strength * token_bonus
                    score = min(vector_sim + bonus * vector_sim, 1.0)
                    token_activated[tid].score = score
                    logger.debug(
                        "Token-activated %s: vector_sim=%.3f,"
                        " token_strength=%.3f,"
                        " bonus=%.3f, score=%.3f",
                        tid,
                        vector_sim,
                        token_strength,
                        token_strength * token_bonus,
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

    # Step 4: Merge results -- boost overlapping traces, append token-only traces
    # Build a map of raw token strengths before blending
    # overwrites scores for missing traces
    raw_token_strengths: dict[str, float] = {}
    for row in token_rows:
        tid = str(row["trace_id"])
        if tid not in raw_token_strengths:
            raw_token_strengths[tid] = float(row["strength"])

    for tid, token_result in token_activated.items():
        if tid in vector_results:
            # Trace found by both vector and token -- boost the vector result
            existing = vector_results[tid]
            vector_sim = existing.score
            token_strength = raw_token_strengths.get(tid, 0.0)
            boosted = min(vector_sim + token_strength * token_bonus * vector_sim, 1.0)
            existing.score = boosted
            existing.source = "token+vector"
            existing.token_label = token_result.token_label
            logger.debug(
                "Overlap %s: vector_sim=%.3f,"
                " token_strength=%.3f,"
                " bonus=%.3f, boosted=%.3f",
                tid,
                vector_sim,
                token_strength,
                token_strength * token_bonus,
                boosted,
            )

    merged: list[RetrievalResult] = list(vector_results.values())
    for tid, result in token_activated.items():
        if tid not in vector_results:
            merged.append(result)

    # Sort by score descending and truncate to top_k for fair comparison
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
