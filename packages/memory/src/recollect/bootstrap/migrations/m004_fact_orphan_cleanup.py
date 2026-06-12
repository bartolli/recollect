"""m004_fact_orphan_cleanup — drop orphaned fact-owner concept embeddings.

Duplicate-mention fact writes embedded context_tags for fact ids that were
never stored (the new PersonaFact object on the exact-duplicate path).
concept_embeddings carries no FK on its polymorphic (owner_type, owner_id),
so the rows accumulated unread. The write path now gates tag embedding on
the actually-stored fact; this migration removes the backlog. Idempotent:
re-running matches zero rows.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_DELETE_ORPHANS = """
DELETE FROM concept_embeddings ce
WHERE ce.owner_type = 'fact'
  AND NOT EXISTS (
    SELECT 1 FROM persona_facts pf WHERE pf.id = ce.owner_id
  )
"""


class _M004FactOrphanCleanup:
    name = "m004_fact_orphan_cleanup"
    description = (
        "Delete concept_embeddings rows whose owner_type='fact' owner id "
        "has no persona_facts row (never-stored duplicate-path facts)."
    )

    async def up(self, conn: asyncpg.Connection[Any]) -> None:
        result = await conn.execute(_DELETE_ORPHANS)
        logger.info("m004: removed orphaned fact concept embeddings (%s)", result)


m004_fact_orphan_cleanup = _M004FactOrphanCleanup()
