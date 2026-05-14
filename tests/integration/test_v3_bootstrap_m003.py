"""Integration test: m003_embedding_contract on scratch DBs.

m003 creates the embedding_contract table and records the current
provider contract (model, task_prefix_version). The contract is readable
via storage and used by the write-path verification guard.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.bootstrap.migrations import default_registry
from recollect.bootstrap.runner import apply
from recollect.core import CognitiveMemory
from recollect.embeddings import FastEmbedProvider
from recollect.exceptions import EmbeddingContractError
from recollect.storage_ops import get_embedding_contract

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    _ScratchPool = asyncpg.Pool[asyncpg.Record]
    _ScratchFixture = tuple[asyncpg.Pool[asyncpg.Record], str]

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]


def _scratch_url(db_name: str) -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def _admin_url() -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path="/postgres"))


@pytest.fixture()
async def scratch() -> AsyncGenerator[_ScratchFixture, None]:
    db_name = f"bootstrap_m003_{uuid.uuid4().hex[:10]}"
    dsn = _scratch_url(db_name)
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)
    try:
        yield pool, dsn
    finally:
        await pool.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


class TestM003EmbeddingContract:
    async def test_contract_row_inserted_on_apply(
        self, scratch: _ScratchFixture,
    ) -> None:
        pool, _ = scratch
        await apply(pool, default_registry())
        provider = FastEmbedProvider()
        stored = await get_embedding_contract(pool)
        assert stored == provider.contract()

    async def test_contract_table_is_single_row(
        self, scratch: _ScratchFixture,
    ) -> None:
        pool, _ = scratch
        await apply(pool, default_registry())
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM embedding_contract"
            )
            assert count == 1
            with pytest.raises(asyncpg.IntegrityConstraintViolationError):
                await conn.execute(
                    "INSERT INTO embedding_contract (model, task_prefix_version) "
                    "VALUES ('other', 'other')"
                )


class TestConnectVerifiesContract:
    async def test_connect_raises_on_mismatched_stored_contract(
        self, scratch: _ScratchFixture,
    ) -> None:
        # Simulates re-embed-required: DB carries vectors from an older
        # provider; running a newer provider must refuse to start.
        pool, dsn = scratch
        await apply(pool, default_registry())
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE embedding_contract SET model = 'stale-model' WHERE id"
            )

        mem = CognitiveMemory()
        with pytest.raises(EmbeddingContractError, match="stale-model"):
            await mem.connect(dsn)
