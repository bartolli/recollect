"""m001_initial — wraps SCHEMA_SQL + 3 HNSW vector indexes.

Idempotent via CREATE ... IF NOT EXISTS. First-time run on an existing DB
(memory_v3, recollect_macos) is a no-op DDL-wise; the runner records the
applied_migrations row so subsequent bootstraps skip it.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from recollect.pool import (
    CONCEPT_VECTOR_INDEX_SQL,
    FACT_VECTOR_INDEX_SQL,
    SCHEMA_SQL,
    VECTOR_INDEX_SQL,
)

logger = logging.getLogger(__name__)


class _M001Initial:
    name = "m001_initial"
    description = "Initial schema: 10 tables, 27 indexes, 3 HNSW vector indexes."

    async def up(self, conn: asyncpg.Connection[Any]) -> None:
        await conn.execute(SCHEMA_SQL)
        try:
            await conn.execute(VECTOR_INDEX_SQL)
            await conn.execute(FACT_VECTOR_INDEX_SQL)
            await conn.execute(CONCEPT_VECTOR_INDEX_SQL)
        except asyncpg.UndefinedObjectError:
            logger.exception(
                "pgvector HNSW unavailable; vector indexes skipped"
            )


m001_initial = _M001Initial()
