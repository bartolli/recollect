"""Integration: telescoping decay write + m006 decay-anchor column.

apply_decay_factor clamps the factor and stamps last_decayed_at in one
statement; apply_strength_factor never touches the stamp; m006 restores
the column on legacy DBs whose SCHEMA_SQL predates it.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.bootstrap.migrations import m006_decay_anchor
from recollect.core import CognitiveMemory
from recollect.datetime_utils import now_utc
from recollect.models import MemoryTrace

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [pytest.mark.slow, pytest.mark.asyncio]

_COLUMN_COUNT = (
    "SELECT count(*) FROM information_schema.columns "
    "WHERE table_name = 'memory_traces' AND column_name = 'last_decayed_at'"
)


def _scratch_url(db_name: str) -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def _admin_url() -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path="/postgres"))


@pytest.fixture()
async def scratch_db() -> AsyncGenerator[str, None]:
    db_name = f"decayanchor_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    try:
        yield _scratch_url(db_name)
    finally:
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


@pytest.fixture()
async def mem(scratch_db: str) -> AsyncGenerator[CognitiveMemory, None]:
    m = CognitiveMemory()
    await m.connect(scratch_db)
    try:
        yield m
    finally:
        await m.close()


class TestDecayWrite:
    async def test_stamps_and_clamps_in_one_statement(
        self, mem: CognitiveMemory
    ) -> None:
        trace = MemoryTrace(content="decaying", strength=0.5)
        await mem.storage.traces.store_trace(trace)
        stamp = now_utc()
        await mem.storage.traces.apply_decay_factor(trace.id, 0.9, stamp)
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.strength == pytest.approx(0.5 * 0.9)
        assert stored.last_decayed_at == stamp

    async def test_decay_factor_clamps_in_sql(self, mem: CognitiveMemory) -> None:
        trace = MemoryTrace(content="ceiling", strength=0.9)
        await mem.storage.traces.store_trace(trace)
        await mem.storage.traces.apply_decay_factor(trace.id, 1.5, now_utc())
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.strength == 1.0

    async def test_boost_write_leaves_stamp_untouched(
        self, mem: CognitiveMemory
    ) -> None:
        # A stamped boost would shrink the next decay window.
        trace = MemoryTrace(content="boosted", strength=0.5)
        await mem.storage.traces.store_trace(trace)
        await mem.storage.traces.apply_strength_factor(trace.id, 1.2)
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.last_decayed_at is None


class TestM006DecayAnchor:
    async def test_fresh_twin_and_legacy_restore_idempotent(
        self, scratch_db: str
    ) -> None:
        m = CognitiveMemory()
        await m.connect(scratch_db)
        await m.close()
        conn = await asyncpg.connect(scratch_db)
        try:
            assert await conn.fetchval(_COLUMN_COUNT) == 1
            await conn.execute(
                "ALTER TABLE memory_traces DROP COLUMN last_decayed_at"
            )
            await m006_decay_anchor.up(conn)
            await m006_decay_anchor.up(conn)
            assert await conn.fetchval(_COLUMN_COUNT) == 1
        finally:
            await conn.close()
