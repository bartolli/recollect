"""m003_embedding_contract — record stored-vector contract.

Single-row table holds (model, task_prefix_version, applied_at). CHECK
constraint enforces singletonness. Subsequent connects compare the row
to the running provider via verify_embedding_contract().

`get_current_contract` is injectable for testing; default reads from a
fresh FastEmbedProvider instance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import asyncpg

from recollect.embeddings import FastEmbedProvider

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS embedding_contract (
    id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id = TRUE),
    model TEXT NOT NULL,
    task_prefix_version TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _default_contract() -> tuple[str, str]:
    return FastEmbedProvider().contract()


class _M003EmbeddingContract:
    name = "m003_embedding_contract"
    description = (
        "Create embedding_contract table; record (model, task_prefix_version)."
    )

    def __init__(
        self,
        get_current_contract: Callable[[], tuple[str, str]] | None = None,
    ) -> None:
        self._get_contract = get_current_contract or _default_contract

    async def up(self, conn: asyncpg.Connection[Any]) -> None:
        await conn.execute(_CREATE_TABLE)
        model, version = self._get_contract()
        await conn.execute(
            "INSERT INTO embedding_contract (model, task_prefix_version) "
            "VALUES ($1, $2) ON CONFLICT (id) DO NOTHING",
            model,
            version,
        )


m003_embedding_contract = _M003EmbeddingContract()
