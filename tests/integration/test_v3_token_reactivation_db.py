"""Integration: archived-token reactivation contract in the token store.

Pins the SQL both write-time paths (extend, revise) lean on:
get_activated_trace_ids filters status='active', so an archived token
never propagates its stamps; reinforce_tokens on an archived row sets
strength = significance and status = 'active', restoring propagation.
Ordering property: a significance write (update_token) before the
reinforce lands the reset on the revised value.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace, RecallToken

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
    db_name = f"tok_react_{uuid.uuid4().hex[:10]}"
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


async def _seed_archived_token(mem: CognitiveMemory) -> tuple[str, str, str]:
    """Two stamped traces under one token archived at full strength.

    Strength stays 1.0 so invisibility is attributable to status alone,
    not the strength_threshold floor.
    """
    seed = MemoryTrace(content="seed trace", strength=0.5)
    linked = MemoryTrace(content="linked trace", strength=0.5)
    await mem.storage.traces.store_trace(seed)
    await mem.storage.traces.store_trace(linked)
    token_id = await mem.storage.recall_tokens.create_token(
        RecallToken(label="Sarah | old situation | old implication", significance=0.7)
    )
    await mem.storage.recall_tokens.stamp_traces(token_id, [seed.id, linked.id])
    pool = await mem.storage.pool.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE recall_tokens SET status = 'archived' WHERE id = $1",
            token_id,
        )
    return token_id, seed.id, linked.id


async def _token_row(mem: CognitiveMemory, token_id: str) -> tuple[str, float]:
    pool = await mem.storage.pool.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, strength FROM recall_tokens WHERE id = $1", token_id
        )
    return str(row["status"]), float(row["strength"])


class TestArchivedTokenReactivation:
    async def test_archived_token_does_not_propagate(
        self, mem: CognitiveMemory
    ) -> None:
        _token_id, seed_id, _linked_id = await _seed_archived_token(mem)
        rows = await mem.storage.recall_tokens.get_activated_trace_ids([seed_id])
        assert rows == []

    async def test_reinforce_after_significance_write_restores_propagation(
        self, mem: CognitiveMemory
    ) -> None:
        # The revise path's call order: update_token (new significance),
        # then reinforce_tokens -- the reset must land on the new value.
        token_id, seed_id, linked_id = await _seed_archived_token(mem)
        await mem.storage.recall_tokens.update_token(
            token_id, "Sarah | risk resolved | no action needed", 0.3
        )
        await mem.storage.recall_tokens.reinforce_tokens([token_id], 0.1)

        status, strength = await _token_row(mem, token_id)
        assert status == "active"
        assert strength == pytest.approx(0.3)

        rows = await mem.storage.recall_tokens.get_activated_trace_ids([seed_id])
        propagated = [r[0] for r in rows]
        assert propagated == [linked_id]
