"""PostgreSQL vector index implementation."""

from __future__ import annotations

import logging

import asyncpg

from recollect.config import config
from recollect.exceptions import StorageError
from recollect.models import MemoryTrace
from recollect.pool import PoolManager
from recollect.storage_utils import embedding_to_pgvector, row_to_trace

logger = logging.getLogger(__name__)


class PgVectorIndex:
    """PostgreSQL + pgvector implementation of VectorIndex protocol."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def search_semantic(
        self,
        query_embedding: list[float],
        limit: int = 10,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[tuple[MemoryTrace, float]]:
        """Find traces by vector similarity, filtered by strength.

        Returns (trace, cosine_similarity) tuples sorted by similarity
        descending.
        """
        try:
            pool = await self._pool_mgr.get_pool()
            embedding_str = embedding_to_pgvector(query_embedding)
            threshold = float(config.get("retrieval.selection_threshold", 0.1))
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT *, embedding <=> $1::vector AS distance
                    FROM memory_traces
                    WHERE strength >= $2
                      AND ($4::text IS NULL OR session_id = $4)
                      AND ($5::text IS NULL OR user_id = $5)
                    ORDER BY distance ASC
                    LIMIT $3
                    """,
                    embedding_str,
                    threshold,
                    limit,
                    session_id,
                    user_id,
                )
            return [(row_to_trace(dict(r)), 1.0 - float(r["distance"])) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to search")
            raise StorageError(f"Failed to search: {exc}") from exc

    async def spread_activation(
        self, seed_id: str, max_depth: int = 2
    ) -> list[tuple[MemoryTrace, float]]:
        """Spread activation from seed trace through associations.

        Returns connected traces with their activation levels,
        decaying by weight at each hop.
        """
        try:
            decay = float(config.get("activation.activation_decay", 0.7))
            threshold = float(config.get("activation.activation_threshold", 0.1))
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    WITH RECURSIVE activation AS (
                        SELECT id, content, pattern, context, embedding,
                               strength, activation_count, retrieval_count,
                               last_activation, last_retrieval, consolidated,
                               created_at, decay_rate, emotional_valence,
                               significance, session_id, user_id,
                               1.0::float AS activation_level,
                               0 AS depth
                        FROM memory_traces WHERE id = $1

                        UNION ALL

                        SELECT mt.id, mt.content, mt.pattern, mt.context,
                               mt.embedding, mt.strength,
                               mt.activation_count,
                               mt.retrieval_count, mt.last_activation,
                               mt.last_retrieval, mt.consolidated,
                               mt.created_at, mt.decay_rate,
                               mt.emotional_valence, mt.significance,
                               mt.session_id, mt.user_id,
                               (a.activation_level * $2 *
                                CASE WHEN assoc.source_trace_id = a.id
                                     THEN assoc.forward_strength
                                     ELSE assoc.backward_strength
                                END)::float,
                               a.depth + 1
                        FROM activation a
                        JOIN associations assoc
                            ON assoc.source_trace_id = a.id
                            OR (assoc.target_trace_id = a.id
                                AND assoc.association_type
                                    NOT IN ('temporal'))
                        JOIN memory_traces mt
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
                    SELECT DISTINCT ON (id) *
                    FROM activation
                    WHERE id != $1
                    ORDER BY id, activation_level DESC
                    """,
                    seed_id,
                    decay,
                    max_depth,
                    threshold,
                )
            return [(row_to_trace(dict(r)), float(r["activation_level"])) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to spread activation")
            raise StorageError(
                f"Failed to spread activation from {seed_id}: {exc}"
            ) from exc
