"""PostgreSQL entity relation store implementation."""

from __future__ import annotations

import logging

import asyncpg

from recollect.exceptions import StorageError
from recollect.models import EntityRelation
from recollect.pool import PoolManager

logger = logging.getLogger(__name__)


class PgEntityRelationStore:
    """PostgreSQL implementation of EntityRelationStore protocol."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def store_relation(self, rel: EntityRelation) -> str:
        """Store an entity-to-entity relationship edge."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO entity_relations (
                        id, source_entity, relation, target_entity,
                        context, weight, source_trace_id, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (source_entity, relation, target_entity)
                    DO UPDATE SET
                        weight = GREATEST(entity_relations.weight, EXCLUDED.weight),
                        context = EXCLUDED.context,
                        source_trace_id = EXCLUDED.source_trace_id
                    """,
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

    async def get_relations(
        self, entity_name: str, *, limit: int = 20
    ) -> list[EntityRelation]:
        """Get direct relations for an entity (1-hop)."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM entity_relations
                    WHERE LOWER(source_entity) = LOWER($1)
                       OR LOWER(target_entity) = LOWER($1)
                    ORDER BY weight DESC
                    LIMIT $2
                    """,
                    entity_name,
                    limit,
                )
            return [_row_to_entity_relation(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get entity relations")
            raise StorageError(f"Failed to get entity relations: {exc}") from exc

    async def get_related_entities(
        self, entity_name: str, *, max_depth: int = 3, limit: int = 50
    ) -> list[str]:
        """Traverse entity relation graph up to max_depth hops.

        Returns distinct entity names reachable from the given entity.
        Uses a recursive CTE similar to spreading activation.
        """
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    WITH RECURSIVE reachable(entity, depth) AS (
                        SELECT $1::text, 0
                        UNION
                        SELECT
                            CASE
                                WHEN LOWER(er.source_entity) = LOWER(r.entity)
                                THEN er.target_entity
                                ELSE er.source_entity
                            END,
                            r.depth + 1
                        FROM reachable r
                        JOIN entity_relations er
                            ON LOWER(er.source_entity) = LOWER(r.entity)
                            OR LOWER(er.target_entity) = LOWER(r.entity)
                        WHERE r.depth < $2
                    )
                    SELECT DISTINCT entity
                    FROM reachable
                    WHERE LOWER(entity) != LOWER($1)
                    LIMIT $3
                    """,
                    entity_name,
                    max_depth,
                    limit,
                )
            return [row["entity"] for row in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to traverse entity relations")
            raise StorageError(f"Failed to traverse entity relations: {exc}") from exc

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


def _row_to_entity_relation(row: dict[str, object]) -> EntityRelation:
    """Convert a database row to EntityRelation model."""
    return EntityRelation(
        id=str(row["id"]),
        source_entity=str(row["source_entity"]),
        relation=str(row["relation"]),
        target_entity=str(row["target_entity"]),
        context=str(row.get("context", "")),
        weight=float(row.get("weight", 0.5)),  # type: ignore[arg-type]
        source_trace_id=row.get("source_trace_id"),  # type: ignore[arg-type]
        created_at=row["created_at"],  # type: ignore[arg-type]
    )
