"""m005_trace_status — memory_traces.status for the archive lifecycle.

Existing-DB twin of the SCHEMA_SQL column (fresh DBs get it from m001):
ADD COLUMN IF NOT EXISTS no-ops where SCHEMA_SQL already created it.
Constant DEFAULT makes the ALTER metadata-only on PG11+ -- no rewrite.
Partial index covers the decay scan (active, unconsolidated traces).
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_ADD_COLUMN = """
ALTER TABLE memory_traces
ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_traces_active_unconsolidated
ON memory_traces(created_at)
WHERE status = 'active' AND consolidated = FALSE
"""


class _M005TraceStatus:
    name = "m005_trace_status"
    description = (
        "Add memory_traces.status (TEXT NOT NULL DEFAULT 'active') and the "
        "active-unconsolidated partial index."
    )

    async def up(self, conn: asyncpg.Connection[Any]) -> None:
        await conn.execute(_ADD_COLUMN)
        await conn.execute(_CREATE_INDEX)
        logger.info("m005: memory_traces.status ready")


m005_trace_status = _M005TraceStatus()
