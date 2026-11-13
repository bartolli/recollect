"""m008_relation_assertions -- entity_relations becomes per-trace assertions.

Existing-DB twin of the SCHEMA_SQL shape (fresh DBs get it from m001):
each statement no-ops or rewrites to the same shape on re-apply. The FK
keeps the name an inline REFERENCES generates, so fresh and twin match.
No backfill: existing rows keep NULL endpoint ids.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_ADD_COLUMNS = """
ALTER TABLE entity_relations
ADD COLUMN IF NOT EXISTS source_entity_id TEXT REFERENCES entities(id),
ADD COLUMN IF NOT EXISTS target_entity_id TEXT REFERENCES entities(id)
"""

_DROP_TRACE_FK = """
ALTER TABLE entity_relations
DROP CONSTRAINT IF EXISTS entity_relations_source_trace_id_fkey
"""

_ADD_TRACE_FK = """
ALTER TABLE entity_relations
ADD CONSTRAINT entity_relations_source_trace_id_fkey
FOREIGN KEY (source_trace_id) REFERENCES memory_traces(id) ON DELETE CASCADE
"""

_DROP_INDEXES = """
DROP INDEX IF EXISTS idx_entity_rel_pair;
DROP INDEX IF EXISTS idx_entity_rel_source;
DROP INDEX IF EXISTS idx_entity_rel_target;
"""

_CREATE_INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_rel_assertion
ON entity_relations(source_trace_id, source_entity, relation, target_entity);
CREATE INDEX IF NOT EXISTS idx_entity_rel_source_id
ON entity_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_rel_target_id
ON entity_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_trace_entities_trace
ON trace_entities(trace_id);
"""


class _M008RelationAssertions:
    name = "m008_relation_assertions"
    description = (
        "entity_relations: source_entity_id and target_entity_id (FK, no "
        "delete action) with their indexes, source_trace_id FK cascades, "
        "unique per (source_trace_id, source_entity, relation, "
        "target_entity) replaces the global triple index; "
        "trace_entities(trace_id) index for endpoint resolution."
    )

    async def up(self, conn: asyncpg.Connection[Any]) -> None:
        await conn.execute(_ADD_COLUMNS)
        await conn.execute(_DROP_TRACE_FK)
        await conn.execute(_ADD_TRACE_FK)
        await conn.execute(_DROP_INDEXES)
        await conn.execute(_CREATE_INDEXES)
        logger.info("m008: entity_relations assertions ready")


m008_relation_assertions = _M008RelationAssertions()
