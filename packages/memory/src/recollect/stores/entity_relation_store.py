"""PostgreSQL entity relation store implementation."""

from __future__ import annotations

import logging

import asyncpg

from recollect.exceptions import StorageError
from recollect.models import EntityEdge, EntityRelation
from recollect.pool import ENTITY_NAME_KEY_SQL, PoolManager

logger = logging.getLogger(__name__)


def _endpoint_id_sql(param: str) -> str:
    # An endpoint resolves only through the asserting trace's own mentions
    # and never creates an entity or a name. MIN: two mentions of one key in
    # one trace carry the same entity_id; legacy NULL ids drop out.
    return f"""(
    SELECT MIN(te.entity_id) FROM trace_entities te
    WHERE te.trace_id = $7
      AND {ENTITY_NAME_KEY_SQL.format("te.entity_name")}
          = {ENTITY_NAME_KEY_SQL.format(f"{param}::text")}
)"""  # noqa: S608 -- the interpolated key expression is a module constant


_STORE_RELATION_SQL = f"""
INSERT INTO entity_relations (
    id, source_entity, relation, target_entity, context, weight,
    source_trace_id, created_at, source_entity_id, target_entity_id
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, {_endpoint_id_sql("$2")}, {_endpoint_id_sql("$4")}
)
ON CONFLICT (source_trace_id, source_entity, relation, target_entity)
DO UPDATE SET weight = GREATEST(entity_relations.weight, EXCLUDED.weight)
"""  # noqa: S608 -- the interpolated subqueries are module constants

# An unresolved end groups by its name key, kept NULL beside a resolved id
# so a key can never collide with an entity id. Visibility is inherited
# from the asserting trace: forgotten hides, the user predicate scopes.
_GET_EDGES_SQL = f"""
WITH assertion AS (
    SELECT er.source_entity, er.source_entity_id, er.relation,
           er.target_entity, er.target_entity_id, er.weight,
           er.source_trace_id, er.created_at,
           CASE WHEN er.source_entity_id IS NULL
                THEN {ENTITY_NAME_KEY_SQL.format("er.source_entity")} END
               AS source_key,
           CASE WHEN er.target_entity_id IS NULL
                THEN {ENTITY_NAME_KEY_SQL.format("er.target_entity")} END
               AS target_key
    FROM entity_relations er
    JOIN memory_traces mt ON mt.id = er.source_trace_id
    WHERE (er.source_entity_id = $1 OR er.target_entity_id = $1)
      AND mt.status <> 'forgotten'
      AND ($2::text IS NULL OR mt.user_id = $2)
)
SELECT (array_agg(source_entity ORDER BY created_at, source_entity))[1] AS source,
       source_entity_id,
       relation,
       (array_agg(target_entity ORDER BY created_at, target_entity))[1] AS target,
       target_entity_id,
       MAX(weight) AS weight,
       COUNT(DISTINCT source_trace_id) AS trace_count
FROM assertion
GROUP BY source_entity_id, source_key, relation, target_entity_id, target_key
ORDER BY trace_count DESC, weight DESC, relation, target, source
LIMIT $3
"""  # noqa: S608 -- the interpolated key expression is a module constant


class PgEntityRelationStore:
    """PostgreSQL implementation of EntityRelationStore protocol."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def store_relation(self, rel: EntityRelation) -> str:
        """Store one trace's assertion; endpoint ids resolve in the same statement."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    _STORE_RELATION_SQL,
                    rel.id,
                    rel.source_entity,
                    rel.relation,
                    rel.target_entity,
                    rel.context,
                    rel.weight,
                    rel.source_trace_id,
                    rel.created_at,
                )
            return rel.id
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to store entity relation")
            raise StorageError(f"Failed to store entity relation: {exc}") from exc

    async def get_edges(
        self, entity_id: str, *, user_id: str | None = None, limit: int = 50
    ) -> list[EntityEdge]:
        """Edges with the entity at either end, most-asserted first."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(_GET_EDGES_SQL, entity_id, user_id, limit)
            return [EntityEdge.model_validate(dict(row)) for row in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get entity edges")
            raise StorageError(f"Failed to get entity edges: {exc}") from exc

    async def delete_by_trace(self, trace_id: str) -> int:
        """Delete entity relations linked to a trace."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM entity_relations WHERE source_trace_id = $1",
                    trace_id,
                )
            # result is like "DELETE N"
            return int(result.split()[-1]) if result else 0
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to delete entity relations")
            raise StorageError(f"Failed to delete entity relations: {exc}") from exc
