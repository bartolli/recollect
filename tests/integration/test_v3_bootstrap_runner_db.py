"""Integration test: advisory-lock serialization on a real DB.

Two concurrent apply() coroutines must run sequentially because of the
session-scoped advisory lock. After both return, exactly one bootstrap
applied migrations; the other observed all-applied.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.bootstrap.migrations import default_registry
from recollect.bootstrap.runner import apply

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    _ScratchPool = asyncpg.Pool[asyncpg.Record]

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
async def scratch_pool() -> AsyncGenerator[_ScratchPool, None]:
    db_name = f"bootstrap_lock_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(_scratch_url(db_name), min_size=2, max_size=6)
    try:
        yield pool
    finally:
        await pool.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


class TestConcurrentBootstrap:
    async def test_two_concurrent_applies_serialize(
        self,
        scratch_pool: _ScratchPool,
    ) -> None:
        r1, r2 = await asyncio.gather(
            apply(scratch_pool, default_registry()),
            apply(scratch_pool, default_registry()),
        )
        # exactly one bootstrap applied migrations; the other observed them
        applied_counts = sorted([len(r1.applied), len(r2.applied)])
        skipped_counts = sorted([len(r1.skipped), len(r2.skipped)])
        registered = len(default_registry().names())
        assert applied_counts == [0, registered]
        assert skipped_counts == [0, registered]

    async def test_full_apply_idempotent_on_real_db(
        self,
        scratch_pool: _ScratchPool,
    ) -> None:
        first = await apply(scratch_pool, default_registry())
        second = await apply(scratch_pool, default_registry())
        assert first.applied == default_registry().names()
        assert second.applied == []
        assert second.skipped == default_registry().names()
