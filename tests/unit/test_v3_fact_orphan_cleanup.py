"""m004: one-time cleanup of orphaned fact-owner concept embeddings.

Pre-slice-2 duplicate-mention writes embedded tags for never-stored fact
ids; concept_embeddings has no FK on its polymorphic owner, so the rows
accumulated unread.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from recollect.bootstrap.migrations import (
    default_registry,
    m004_fact_orphan_cleanup,
)


class TestM004FactOrphanCleanup:
    def test_registered_after_m003(self) -> None:
        names = default_registry().names()
        assert names.index("m004_fact_orphan_cleanup") == (
            names.index("m003_embedding_contract") + 1
        )

    async def test_up_deletes_only_ownerless_fact_rows(self) -> None:
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 285")
        await m004_fact_orphan_cleanup.up(conn)
        sql = conn.execute.await_args.args[0]
        assert "owner_type = 'fact'" in sql
        assert "NOT EXISTS" in sql
        assert "persona_facts" in sql
