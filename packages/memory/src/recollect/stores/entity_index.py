"""PostgreSQL entity index implementation."""

from __future__ import annotations

import logging

import asyncpg

from recollect.config import config
from recollect.exceptions import StorageError
from recollect.models import TraceConcept, TraceEntity
from recollect.pool import PoolManager

logger = logging.getLogger(__name__)


class PgEntityIndex:
    """PostgreSQL implementation of EntityIndex protocol."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def store_trace_entities(
        self, trace_id: str, entities: list[TraceEntity]
    ) -> None:
        """Batch insert entity-trace links."""
        if not entities:
            return
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO trace_entities (entity_name, entity_type, trace_id)
                    VALUES ($1, $2, $3)
                    ON CONFLICT DO NOTHING
                    """,
                    [(e.entity_name, e.entity_type, trace_id) for e in entities],
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to store trace entities")
            raise StorageError(f"Failed to store trace entities: {exc}") from exc

    async def store_trace_concepts(
        self, trace_id: str, concepts: list[TraceConcept]
    ) -> None:
        """Batch insert concept-trace links."""
        if not concepts:
            return
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO trace_concepts (concept, trace_id)
                    VALUES ($1, $2)
                    ON CONFLICT DO NOTHING
                    """,
                    [(c.concept, trace_id) for c in concepts],
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to store trace concepts")
            raise StorageError(f"Failed to store trace concepts: {exc}") from exc

    async def get_traces_by_entity(
        self, entity_name: str, *, limit: int = 20
    ) -> list[str]:
        """Return trace IDs linked to an entity name."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT trace_id FROM trace_entities
                    WHERE entity_name = $1
                    LIMIT $2
                    """,
                    entity_name,
                    limit,
                )
            return [row["trace_id"] for row in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get traces by entity")
            raise StorageError(f"Failed to get traces by entity: {exc}") from exc

    async def get_traces_by_concept(
        self, concept: str, *, limit: int = 20
    ) -> list[str]:
        """Return trace IDs linked to a concept."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT trace_id FROM trace_concepts
                    WHERE concept = $1
                    LIMIT $2
                    """,
                    concept,
                    limit,
                )
            return [row["trace_id"] for row in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get traces by concept")
            raise StorageError(f"Failed to get traces by concept: {exc}") from exc

    async def match_entities(
        self, names: list[str], *, limit: int = 20
    ) -> list[tuple[str, float]]:
        """Match entity names using trigram similarity.

        Returns (trace_id, similarity_score) pairs ordered by score.
        """
        if not names:
            return []
        try:
            threshold = float(config.get("retrieval.entity_similarity_threshold", 0.3))
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT te.trace_id,
                           MAX(similarity(LOWER(te.entity_name), LOWER(n)))
                               AS sim
                    FROM trace_entities te,
                         UNNEST($1::text[]) AS n
                    WHERE similarity(LOWER(te.entity_name), LOWER(n)) > $2
                    GROUP BY te.trace_id
                    ORDER BY sim DESC
                    LIMIT $3
                    """,
                    names,
                    threshold,
                    limit,
                )
            return [(row["trace_id"], float(row["sim"])) for row in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to match entities")
            raise StorageError(f"Failed to match entities: {exc}") from exc
