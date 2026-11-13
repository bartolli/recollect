"""m007_entity_registry -- per-user entities behind trace_entities mentions.

Existing-DB twin of the SCHEMA_SQL objects (fresh DBs get them from m001):
each statement no-ops where SCHEMA_SQL already created it. The entities
DDL is pool.ENTITIES_SQL by name, so the name_key expression stays single-
sourced. No backfill: legacy mentions keep a NULL entity_id.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from recollect.pool import ENTITIES_SQL

logger = logging.getLogger(__name__)

_ADD_COLUMN = """
ALTER TABLE trace_entities
ADD COLUMN IF NOT EXISTS entity_id TEXT REFERENCES entities(id)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_trace_entities_entity
ON trace_entities(entity_id)
"""


class _M007EntityRegistry:
    name = "m007_entity_registry"
    description = (
        "Add the entities registry (unique per name_key, user_id) and "
        "trace_entities.entity_id (FK, no delete action) with its index."
    )

    async def up(self, conn: asyncpg.Connection[Any]) -> None:
        await conn.execute(ENTITIES_SQL)
        await conn.execute(_ADD_COLUMN)
        await conn.execute(_CREATE_INDEX)
        logger.info("m007: entities registry ready")


m007_entity_registry = _M007EntityRegistry()
