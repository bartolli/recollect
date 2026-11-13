"""Integration: entity resolution and the erase sweep under a concurrent batch.

One batch runs inside a held transaction; the racing statement starts,
blocks on that batch's row locks, and resumes after the commit.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace, TraceEntity
from recollect.stores.entity_index import PgEntityIndex

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from recollect.pool import PoolManager

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [pytest.mark.slow, pytest.mark.asyncio]


def _url(db_name: str) -> str:
    return urlunparse(urlparse(DB_URL)._replace(path=f"/{db_name}"))


async def _admin(sql: str) -> None:
    conn = await asyncpg.connect(_url("postgres"))
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@pytest.fixture()
async def db_url() -> AsyncGenerator[str, None]:
    db_name = f"entityrace_{uuid.uuid4().hex[:10]}"
    await _admin(f'CREATE DATABASE "{db_name}"')
    try:
        yield _url(db_name)
    finally:
        await _admin(f'DROP DATABASE "{db_name}" WITH (FORCE)')


@pytest.fixture()
async def mem(db_url: str) -> AsyncGenerator[CognitiveMemory, None]:
    m = CognitiveMemory()
    await m.connect(db_url)
    try:
        yield m
    finally:
        await m.close()


class _OneConnection:
    """Pool stand-in: every acquire yields the one held connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def get_pool(self) -> _OneConnection:
        return self

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        yield self._conn


@asynccontextmanager
async def _held_batch(
    db_url: str, trace_id: str, name: str
) -> AsyncIterator[asyncpg.transaction.Transaction]:
    conn = await asyncpg.connect(db_url)
    try:
        tx = conn.transaction()
        await tx.start()
        held = PgEntityIndex(cast("PoolManager", _OneConnection(conn)))
        await held.store_trace_entities(
            trace_id, [TraceEntity(entity_name=name, trace_id=trace_id)]
        )
        yield tx
    finally:
        await conn.close()


async def _await_lock_waiter(mem: CognitiveMemory) -> None:
    pool = await mem.storage.pool.get_pool()
    for _ in range(200):
        if await pool.fetchval(
            "SELECT count(*) FROM pg_stat_activity"
            " WHERE datname = current_database() AND wait_event_type = 'Lock'"
        ):
            return
        await asyncio.sleep(0.025)
    raise AssertionError("the racing statement never waited on a lock")


async def _stored(mem: CognitiveMemory, *names: str) -> MemoryTrace:
    trace = MemoryTrace(content="entity mention", user_id="u-a")
    await mem.storage.traces.store_trace(trace)
    await mem.storage.entities.store_trace_entities(
        trace.id, [TraceEntity(entity_name=n, trace_id=trace.id) for n in names]
    )
    return trace


async def test_concurrent_batches_converge_and_type_the_new_entity(
    mem: CognitiveMemory, db_url: str
) -> None:
    first, second = await _stored(mem), await _stored(mem)
    async with _held_batch(db_url, first.id, "Borealis") as tx:
        racing = asyncio.create_task(
            mem.storage.entities.store_trace_entities(
                second.id,
                [
                    TraceEntity(
                        entity_name="borealis",
                        entity_type="project",
                        trace_id=second.id,
                    )
                ],
            )
        )
        await _await_lock_waiter(mem)
        await tx.commit()
        await racing

    entities = await mem.entities(user_id="u-a")
    assert [(e.names, e.entity_type, e.trace_count) for e in entities] == [
        (["Borealis"], "project", 2)
    ]


async def test_sweep_keeps_the_name_a_concurrent_mention_resolved_through(
    mem: CognitiveMemory, db_url: str
) -> None:
    erased, kept = await _stored(mem, "Borealis"), await _stored(mem)
    async with _held_batch(db_url, kept.id, "borealis") as tx:
        erasing = asyncio.create_task(mem.erase(erased.id))
        await _await_lock_waiter(mem)
        await tx.commit()
        assert await erasing is True

    assert await mem.storage.entities.get_traces_by_entity("BOREALIS") == [kept.id]
    entities = await mem.entities(user_id="u-a")
    assert [(e.names, e.trace_count) for e in entities] == [(["Borealis"], 1)]
