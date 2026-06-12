"""m006_decay_anchor -- memory_traces.last_decayed_at for telescoping decay.

Existing-DB twin of the SCHEMA_SQL column (fresh DBs get it from m001):
ADD COLUMN IF NOT EXISTS no-ops where SCHEMA_SQL already created it.
Nullable, no default -- metadata-only ALTER; NULL means never decayed.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_ADD_COLUMN = """
ALTER TABLE memory_traces
ADD COLUMN IF NOT EXISTS last_decayed_at TIMESTAMPTZ
"""


class _M006DecayAnchor:
    name = "m006_decay_anchor"
    description = (
        "Add memory_traces.last_decayed_at (TIMESTAMPTZ NULL) -- the "
        "telescoping decay-window stamp."
    )

    async def up(self, conn: asyncpg.Connection[Any]) -> None:
        await conn.execute(_ADD_COLUMN)
        logger.info("m006: memory_traces.last_decayed_at ready")


m006_decay_anchor = _M006DecayAnchor()
