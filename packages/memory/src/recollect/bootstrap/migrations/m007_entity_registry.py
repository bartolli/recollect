"""m007_entity_registry -- per-user entities and their names behind mentions.

Existing-DB twin of the SCHEMA_SQL objects (fresh DBs get them from m001):
each statement no-ops where SCHEMA_SQL already created it. The entities
and entity_names DDL is pool.ENTITIES_SQL by name, so the name_key
expression stays single-sourced. The backfill resolves every mention with a
NULL entity_id as store_trace_entities would, replayed in trace created_at
order; a fresh or backfilled DB matches nothing.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from recollect.pool import ENTITIES_SQL, ENTITY_NAME_KEY_SQL

logger = logging.getLogger(__name__)

_ADD_COLUMN = """
ALTER TABLE trace_entities
ADD COLUMN IF NOT EXISTS entity_id TEXT REFERENCES entities(id)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_trace_entities_entity
ON trace_entities(entity_id)
"""

# One name_key recurs across owners, so every join back to legacy matches
# user_id too; GROUP BY entity_id keeps two keys of one merged entity from
# upserting its row twice.
_REGISTER_LEGACY_SQL = f"""
WITH legacy AS (
    SELECT te.entity_name, te.entity_type, te.trace_id,
           mt.user_id, mt.created_at,
           {ENTITY_NAME_KEY_SQL.format("te.entity_name")} AS name_key
    FROM trace_entities te
    JOIN memory_traces mt ON mt.id = te.trace_id
    WHERE te.entity_id IS NULL
),
keyed AS MATERIALIZED (
    SELECT l.name_key, l.user_id,
           (array_agg(l.entity_name
               ORDER BY l.created_at, l.trace_id, l.entity_name))[1] AS name,
           MIN(l.created_at) AS created_at,
           gen_random_uuid()::text AS new_id
    FROM legacy l
    GROUP BY l.name_key, l.user_id
),
named AS (
    INSERT INTO entity_names (entity_id, user_id, name, created_at)
    SELECT k.new_id, k.user_id, k.name, k.created_at
    FROM keyed k
    ON CONFLICT (name_key, user_id) DO UPDATE SET name = entity_names.name
    RETURNING entity_id, user_id, name_key
)
INSERT INTO entities (id, user_id, entity_type, created_at)
SELECT n.entity_id,
       n.user_id,
       COALESCE(
           (array_agg(l.entity_type
               ORDER BY l.created_at, l.trace_id, l.entity_name)
               FILTER (WHERE l.entity_type <> 'unknown'))[1],
           'unknown'
       ),
       MIN(l.created_at)
FROM named n
JOIN legacy l
  ON l.name_key = n.name_key AND l.user_id IS NOT DISTINCT FROM n.user_id
GROUP BY n.entity_id, n.user_id
ON CONFLICT (id) DO UPDATE SET entity_type = EXCLUDED.entity_type
WHERE entities.entity_type = 'unknown' AND EXCLUDED.entity_type <> 'unknown'
"""  # noqa: S608 -- the interpolated key expression is a module constant

_LINK_LEGACY_SQL = f"""
UPDATE trace_entities te SET entity_id = en.entity_id
FROM memory_traces mt, entity_names en
WHERE te.entity_id IS NULL
  AND mt.id = te.trace_id
  AND en.name_key = {ENTITY_NAME_KEY_SQL.format("te.entity_name")}
  AND en.user_id IS NOT DISTINCT FROM mt.user_id
"""  # noqa: S608 -- the interpolated key expression is a module constant


class _M007EntityRegistry:
    name = "m007_entity_registry"
    description = (
        "Add the entities registry, its entity_names (unique per name_key, "
        "user_id), and trace_entities.entity_id (FK, no delete action) with "
        "its index; backfill entity_id on every mention written before it."
    )

    async def up(self, conn: asyncpg.Connection[Any]) -> None:
        await conn.execute(ENTITIES_SQL)
        await conn.execute(_ADD_COLUMN)
        await conn.execute(_CREATE_INDEX)
        await conn.execute(_REGISTER_LEGACY_SQL)
        await conn.execute(_LINK_LEGACY_SQL)
        logger.info("m007: entities registry ready")


m007_entity_registry = _M007EntityRegistry()
