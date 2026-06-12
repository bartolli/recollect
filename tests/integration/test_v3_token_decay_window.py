"""Integration: decay_inactive honors an inactivity window.

last_activated_at is written on every reinforcement but was never
consulted: every consolidation pass decayed ALL active tokens, including
ones reinforced seconds earlier.
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.core import CognitiveMemory
from recollect.datetime_utils import now_utc
from recollect.models import RecallToken

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

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
async def mem() -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"tok_decay_{uuid.uuid4().hex[:10]}"
    dsn = _scratch_url(db_name)
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    m = CognitiveMemory()
    await m.connect(dsn)
    try:
        yield m
    finally:
        await m.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


async def _seed_tokens(mem: CognitiveMemory) -> tuple[str, str]:
    """Create one fresh and one stale (2h inactive) token, both strength 1.0."""
    fresh = await mem.storage.recall_tokens.create_token(
        RecallToken(label="fresh (token) | just reinforced | skip decay")
    )
    stale = await mem.storage.recall_tokens.create_token(
        RecallToken(label="stale (token) | long inactive | decays")
    )
    pool = await mem.storage.pool.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE recall_tokens SET last_activated_at = NOW() - INTERVAL '2 hours' "
            "WHERE id = $1",
            stale,
        )
    return fresh, stale


async def _strength(mem: CognitiveMemory, token_id: str) -> float:
    pool = await mem.storage.pool.get_pool()
    async with pool.acquire() as conn:
        value = await conn.fetchval(
            "SELECT strength FROM recall_tokens WHERE id = $1", token_id
        )
    return float(value)


class TestDecayInactivityWindow:
    async def test_recently_activated_token_skips_decay(
        self, mem: CognitiveMemory,
    ) -> None:
        fresh, stale = await _seed_tokens(mem)
        cutoff = now_utc() - timedelta(hours=1)
        decayed = await mem.storage.recall_tokens.decay_inactive(
            0.5, inactive_before=cutoff
        )
        assert decayed == 1
        assert await _strength(mem, fresh) == 1.0
        assert await _strength(mem, stale) == 0.5

    async def test_consolidate_passes_inactivity_cutoff(
        self, mem: CognitiveMemory,
    ) -> None:
        # Scenario: a token reinforced moments before consolidation is
        # untouched; the long-inactive one decays by decay_factor (0.9).
        fresh, stale = await _seed_tokens(mem)
        await mem.consolidate()
        assert await _strength(mem, fresh) == 1.0
        assert await _strength(mem, stale) == pytest.approx(0.9)
