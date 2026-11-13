"""PostgreSQL entity index implementation."""

from __future__ import annotations

import logging

import asyncpg

from recollect.config import config
from recollect.datetime_utils import now_utc
from recollect.exceptions import StorageError
from recollect.models import KnownEntity, TraceConcept, TraceEntity
from recollect.pool import ENTITY_NAME_KEY_SQL, PoolManager

logger = logging.getLogger(__name__)

# One statement: GROUP BY name_key keeps ON CONFLICT DO UPDATE from touching
# a row twice; ORDER BY name_key fixes lock order across concurrent batches.
# DO UPDATE carries no WHERE so RETURNING yields the id on every conflict
# path, including rows a concurrent writer committed after this snapshot.
# user_id comes from the owning trace; a missing trace fails the FK insert.
_STORE_TRACE_ENTITIES_SQL = f"""
WITH mention AS (
    SELECT m.name, m.entity_type, m.ord,
           {ENTITY_NAME_KEY_SQL.format("m.name")} AS name_key
    FROM unnest($2::text[], $3::text[])
         WITH ORDINALITY AS m(name, entity_type, ord)
),
resolved AS (
    INSERT INTO entities (id, user_id, name, entity_type, created_at)
    SELECT gen_random_uuid()::text,
           (SELECT user_id FROM memory_traces WHERE id = $1),
           (array_agg(m.name ORDER BY m.ord))[1],
           COALESCE(
               (array_agg(m.entity_type ORDER BY m.ord)
                   FILTER (WHERE m.entity_type <> 'unknown'))[1],
               'unknown'
           ),
           $4
    FROM mention m
    GROUP BY m.name_key
    ORDER BY m.name_key
    ON CONFLICT (user_id, name_key) DO UPDATE
    SET entity_type = CASE
        WHEN entities.entity_type = 'unknown' THEN EXCLUDED.entity_type
        ELSE entities.entity_type
    END
    RETURNING id, name_key
)
INSERT INTO trace_entities (entity_name, entity_type, trace_id, entity_id)
SELECT m.name, m.entity_type, $1, r.id
FROM mention m
JOIN resolved r ON r.name_key = m.name_key
ON CONFLICT DO NOTHING
"""  # noqa: S608 -- the interpolated key expression is a module constant

# DISTINCT: two surface forms in one trace resolve to one entity and would
# return the trace twice, spending max_links slots.
_TRACES_BY_ENTITY_SQL = f"""
SELECT DISTINCT te.trace_id FROM trace_entities te
JOIN entities e ON e.id = te.entity_id
JOIN memory_traces mt ON mt.id = te.trace_id
WHERE e.name_key = {ENTITY_NAME_KEY_SQL.format("$1::text")}
  AND ($3::text IS NULL OR mt.user_id = $3)
LIMIT $2
"""  # noqa: S608 -- the interpolated key expression is a module constant

# LEFT JOIN: an entity whose mentions are gone still lists, at trace_count 0.
_GET_ENTITIES_SQL = """
SELECT e.id, e.user_id, e.name, e.entity_type, e.created_at,
       COUNT(DISTINCT te.trace_id) AS trace_count
FROM entities e
LEFT JOIN trace_entities te ON te.entity_id = e.id
WHERE ($1::text IS NULL OR e.user_id = $1)
GROUP BY e.id
ORDER BY trace_count DESC, e.name
LIMIT $2
"""


class PgEntityIndex:
    """PostgreSQL implementation of EntityIndex protocol."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def store_trace_entities(
        self, trace_id: str, entities: list[TraceEntity]
    ) -> None:
        """Batch insert entity-trace links, resolving each to its entity row."""
        if not entities:
            return
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    _STORE_TRACE_ENTITIES_SQL,
                    trace_id,
                    [e.entity_name for e in entities],
                    [e.entity_type for e in entities],
                    now_utc(),
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
        self, entity_name: str, *, limit: int = 20, user_id: str | None = None
    ) -> list[str]:
        """Return trace IDs whose mentions resolve to the entity name's key.

        user_id scopes through memory_traces (the single ownership source;
        trace_entities has no user column). The LIMIT window is a shared cap:
        unscoped, earlier users' rows absorb it and late writers link
        cross-user instead of to their own siblings (adr-retrieval-user-isolation).
        Mentions with a NULL entity_id carry no identity and never match.
        """
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    _TRACES_BY_ENTITY_SQL, entity_name, limit, user_id
                )
            return [row["trace_id"] for row in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get traces by entity")
            raise StorageError(f"Failed to get traces by entity: {exc}") from exc

    async def get_traces_by_concept(
        self, concept: str, *, limit: int = 20, user_id: str | None = None
    ) -> list[str]:
        """Return trace IDs linked to a concept.

        Same shared-cap scoping as get_traces_by_entity.
        """
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT tc.trace_id FROM trace_concepts tc
                    JOIN memory_traces mt ON mt.id = tc.trace_id
                    WHERE tc.concept = $1
                      AND ($3::text IS NULL OR mt.user_id = $3)
                    LIMIT $2
                    """,
                    concept,
                    limit,
                    user_id,
                )
            return [row["trace_id"] for row in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get traces by concept")
            raise StorageError(f"Failed to get traces by concept: {exc}") from exc

    async def match_entities(
        self,
        names: list[str],
        *,
        limit: int = 20,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Match entity names using trigram similarity.

        Returns (trace_id, similarity_score) pairs ordered by score.
        user_id restricts matches to traces owned by that user
        (trace_entities carries no user column; the owning trace does).
        None means no filter.
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
                    FROM trace_entities te
                    JOIN memory_traces mt ON mt.id = te.trace_id,
                         UNNEST($1::text[]) AS n
                    WHERE similarity(LOWER(te.entity_name), LOWER(n)) > $2
                      AND ($4::text IS NULL OR mt.user_id = $4)
                    GROUP BY te.trace_id
                    ORDER BY sim DESC
                    LIMIT $3
                    """,
                    names,
                    threshold,
                    limit,
                    user_id,
                )
            return [(row["trace_id"], float(row["sim"])) for row in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to match entities")
            raise StorageError(f"Failed to match entities: {exc}") from exc

    async def get_entities(
        self, *, user_id: str | None = None, limit: int = 50
    ) -> list[KnownEntity]:
        """Registry entities by distinct mentioning traces, then name."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(_GET_ENTITIES_SQL, user_id, limit)
            return [KnownEntity.model_validate(dict(row)) for row in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get entities")
            raise StorageError(f"Failed to get entities: {exc}") from exc

    async def delete_unmentioned_entities(self) -> int:
        """Delete every entity no mention references; returns the count.

        A mention committed between the scan and the delete trips the FK and
        the sweep deletes nothing: that entity is live again, and the next
        sweep takes whatever stayed unmentioned.
        """
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM entities e
                    WHERE NOT EXISTS (
                        SELECT 1 FROM trace_entities te
                        WHERE te.entity_id = e.id
                    )
                    """
                )
            return int(result.split()[-1])
        except asyncpg.ForeignKeyViolationError:
            logger.exception("Entity sweep lost a race with a new mention")
            return 0
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to delete unmentioned entities")
            raise StorageError(f"Failed to delete unmentioned entities: {exc}") from exc
