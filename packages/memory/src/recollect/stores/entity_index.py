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

# One statement, no read-then-write. GROUP BY keeps each ON CONFLICT DO
# UPDATE from touching a row twice (two names of a merged entity share its
# entities row); ORDER BY fixes lock order: name rows, then entities. The
# name conflict path is a real no-op UPDATE, never DO NOTHING: RETURNING
# then yields an entity_id committed after this snapshot, and the rewrite
# fails a concurrent repeatable-read sweep of that row. The entities upsert
# covers new and existing entities, so the type upgrade also reaches one
# committed after this snapshot. The owner is a scalar subquery: a missing
# trace yields NULL and the mention insert fails its FK.
_STORE_TRACE_ENTITIES_SQL = f"""
WITH mention AS (
    SELECT m.name, m.entity_type, m.ord,
           {ENTITY_NAME_KEY_SQL.format("m.name")} AS name_key
    FROM unnest($2::text[], $3::text[])
         WITH ORDINALITY AS m(name, entity_type, ord)
),
keyed AS MATERIALIZED (
    SELECT m.name_key,
           (array_agg(m.name ORDER BY m.ord))[1] AS name,
           gen_random_uuid()::text AS new_id
    FROM mention m
    GROUP BY m.name_key
),
named AS (
    INSERT INTO entity_names (entity_id, user_id, name, created_at)
    SELECT k.new_id, (SELECT user_id FROM memory_traces WHERE id = $1), k.name, $4
    FROM keyed k
    ORDER BY k.name_key
    ON CONFLICT (name_key, user_id) DO UPDATE SET name = entity_names.name
    RETURNING entity_id, name_key
),
registered AS (
    INSERT INTO entities (id, user_id, entity_type, created_at)
    SELECT n.entity_id,
           (SELECT user_id FROM memory_traces WHERE id = $1),
           COALESCE(
               (array_agg(m.entity_type ORDER BY m.ord)
                   FILTER (WHERE m.entity_type <> 'unknown'))[1],
               'unknown'
           ),
           $4
    FROM named n
    JOIN mention m ON m.name_key = n.name_key
    GROUP BY n.entity_id
    ORDER BY n.entity_id
    ON CONFLICT (id) DO UPDATE SET entity_type = EXCLUDED.entity_type
    WHERE entities.entity_type = 'unknown' AND EXCLUDED.entity_type <> 'unknown'
)
INSERT INTO trace_entities (entity_name, entity_type, trace_id, entity_id)
SELECT m.name, m.entity_type, $1, n.entity_id
FROM mention m
JOIN named n ON n.name_key = m.name_key
ON CONFLICT DO NOTHING
"""  # noqa: S608 -- the interpolated key expression is a module constant

# DISTINCT: two surface forms in one trace resolve to one entity and would
# return the trace twice, spending max_links slots.
_TRACES_BY_ENTITY_SQL = f"""
SELECT DISTINCT te.trace_id FROM trace_entities te
JOIN entity_names en ON en.entity_id = te.entity_id
JOIN memory_traces mt ON mt.id = te.trace_id
WHERE en.name_key = {ENTITY_NAME_KEY_SQL.format("$1::text")}
  AND ($3::text IS NULL OR mt.user_id = $3)
LIMIT $2
"""  # noqa: S608 -- the interpolated key expression is a module constant

# LATERAL aggregates: joining names and mentions directly would repeat
# each name once per mention.
_KNOWN_ENTITY_SQL = """
SELECT e.id, e.user_id, e.entity_type, e.created_at, n.names, c.trace_count
FROM entities e
CROSS JOIN LATERAL (
    SELECT array_agg(en.name ORDER BY en.created_at, en.name) AS names
    FROM entity_names en WHERE en.entity_id = e.id
) n
CROSS JOIN LATERAL (
    SELECT COUNT(DISTINCT te.trace_id) AS trace_count
    FROM trace_entities te WHERE te.entity_id = e.id
) c
"""

_GET_ENTITIES_SQL = (
    _KNOWN_ENTITY_SQL
    + """
WHERE ($1::text IS NULL OR e.user_id = $1)
ORDER BY c.trace_count DESC, n.names[1]
LIMIT $2
"""
)

_GET_ENTITY_SQL = _KNOWN_ENTITY_SQL + "WHERE e.id = $1"

_LOCK_MERGE_PAIR_SQL = """
SELECT s.id FROM entities s, entities i
WHERE s.id = $1 AND i.id = $2 AND s.id <> i.id
  AND s.user_id IS NOT DISTINCT FROM i.user_id
FOR UPDATE
"""

_REPOINT_RELATIONS_SQL = """
UPDATE entity_relations SET
    source_entity_id = CASE WHEN source_entity_id = $1 THEN $2
                            ELSE source_entity_id END,
    target_entity_id = CASE WHEN target_entity_id = $1 THEN $2
                            ELSE target_entity_id END
WHERE source_entity_id = $1 OR target_entity_id = $1
"""

_INHERIT_TYPE_SQL = """
UPDATE entities i SET entity_type = s.entity_type
FROM entities s
WHERE i.id = $2 AND s.id = $1 AND i.entity_type = 'unknown'
"""

_DELETE_UNUSED_NAMES_SQL = f"""
DELETE FROM entity_names en
WHERE NOT EXISTS (
    SELECT 1 FROM trace_entities te
    WHERE te.entity_id = en.entity_id
      AND {ENTITY_NAME_KEY_SQL.format("te.entity_name")} = en.name_key
)
"""  # noqa: S608 -- the interpolated key expression is a module constant

_DELETE_UNMENTIONED_ENTITIES_SQL = """
DELETE FROM entities e
WHERE NOT EXISTS (
    SELECT 1 FROM trace_entities te WHERE te.entity_id = e.id
)
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

    async def get_entity(self, entity_id: str) -> KnownEntity | None:
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(_GET_ENTITY_SQL, entity_id)
            return None if row is None else KnownEntity.model_validate(dict(row))
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get entity")
            raise StorageError(f"Failed to get entity: {exc}") from exc

    async def merge_entities(self, source_id: str, into_id: str) -> bool:
        """Move source's names, mentions and relation ends onto into; delete source.

        False with nothing changed unless both exist, differ, and share an
        owner. Merge rewrites identity only; associations are untouched.
        """
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn, conn.transaction():
                # store_trace_entities' lock order, name rows then entities;
                # the reverse deadlocks against a concurrent source mention.
                await conn.execute(
                    "SELECT 1 FROM entity_names WHERE entity_id = $1 FOR UPDATE",
                    source_id,
                )
                if (
                    await conn.fetchval(_LOCK_MERGE_PAIR_SQL, source_id, into_id)
                    is None
                ):
                    return False
                await conn.execute(
                    "UPDATE entity_names SET entity_id = $2 WHERE entity_id = $1",
                    source_id,
                    into_id,
                )
                await conn.execute(
                    "UPDATE trace_entities SET entity_id = $2 WHERE entity_id = $1",
                    source_id,
                    into_id,
                )
                await conn.execute(_REPOINT_RELATIONS_SQL, source_id, into_id)
                await conn.execute(_INHERIT_TYPE_SQL, source_id, into_id)
                await conn.execute("DELETE FROM entities WHERE id = $1", source_id)
            return True
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to merge entities")
            raise StorageError(f"Failed to merge entities: {exc}") from exc

    async def delete_unmentioned_entities(self) -> int:
        """Delete unused name rows, then unmentioned entities; returns the count.

        A name row is unused when no mention of its entity carries its key.
        Repeatable read turns both races with a new mention into a no-op
        sweep: a name row the mention just resolved through fails with a
        serialization error (resolution rewrites it), and a mention of an
        entity being deleted trips the FK. The next sweep takes whatever
        stayed unmentioned.
        """
        try:
            pool = await self._pool_mgr.get_pool()
            async with (
                pool.acquire() as conn,
                conn.transaction(isolation="repeatable_read"),
            ):
                await conn.execute(_DELETE_UNUSED_NAMES_SQL)
                result = await conn.execute(_DELETE_UNMENTIONED_ENTITIES_SQL)
            return int(result.split()[-1])
        except (asyncpg.ForeignKeyViolationError, asyncpg.SerializationError):
            logger.exception("Entity sweep lost a race with a new mention")
            return 0
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to delete unmentioned entities")
            raise StorageError(f"Failed to delete unmentioned entities: {exc}") from exc
