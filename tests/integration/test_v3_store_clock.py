"""Integration: the probe store clock ages a scratch store.

Aging moves stored time, never the process clock: every lifecycle
timestamp shifts back by the same exact interval, so "now - stamp"
grows on every clock the SDK reads (Python and SQL NOW()).
"""

from __future__ import annotations

import os
import random
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from probe_cli.aging import Aging, age_store
from probe_cli.store_clock import EXCLUDED_COLUMNS, SHIFTED_COLUMNS, StoreClock
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get("DATABASE_URL", "")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
    pytest.mark.skipif(not DB_URL, reason="DATABASE_URL unset"),
]

# Stamps and shift straddle the 2026-11-01 US daylight-saving boundary:
# day-valued interval arithmetic in a DST session zone is off by one hour.
T0 = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
FROZEN = datetime(2026, 11, 12, 18, 0, tzinfo=UTC)
DST_ZONE = "America/New_York"
VEC = "[" + ",".join(["0.1"] * 768) + "]"
_ROW_KEY = {"token_stamps": "token_id, trace_id"}
_TIME_TYPES = [
    "timestamp with time zone",
    "timestamp without time zone",
    "date",
    "interval",
    "time with time zone",
    "time without time zone",
]


def _url(db_name: str) -> str:
    return urlunparse(urlparse(DB_URL)._replace(path=f"/{db_name}"))


async def _admin(sql: str) -> None:
    conn = await asyncpg.connect(_url("postgres"))
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@pytest.fixture()
async def db_name() -> AsyncGenerator[str, None]:
    name = f"probe_clock_{uuid.uuid4().hex[:10]}"
    await _admin(f'CREATE DATABASE "{name}"')
    try:
        yield name
    finally:
        await _admin(f'DROP DATABASE "{name}" WITH (FORCE)')


@pytest.fixture()
async def mem(db_name: str) -> AsyncGenerator[CognitiveMemory, None]:
    memory = CognitiveMemory()
    await memory.connect(_url(db_name))
    try:
        yield memory
    finally:
        await memory.close()


@pytest.fixture()
async def store(db_name: str) -> AsyncGenerator[asyncpg.Pool, None]:
    """Bootstrapped scratch store; pool sessions run in a DST zone."""
    memory = CognitiveMemory()
    await memory.connect(_url(db_name))
    await memory.close()
    pool = await asyncpg.create_pool(
        _url(db_name), min_size=1, max_size=2, server_settings={"timezone": DST_ZONE}
    )
    try:
        yield pool
    finally:
        await pool.close()


async def _seed_rows(pool: asyncpg.Pool) -> None:
    h = timedelta(hours=1)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO sessions (id, user_id, created_at, ended_at)"
            " VALUES ('s1', 'u', $1, NULL), ('s2', 'u', $2, $3)",
            T0,
            T0 + h,
            T0 + 2 * h,
        )
        await conn.execute(
            "INSERT INTO memory_traces (id, content, created_at, last_activation,"
            " last_retrieval, last_decayed_at) VALUES"
            " ('t1', 'a', $1, NULL, NULL, NULL), ('t2', 'b', $2, $3, $4, $5)",
            T0,
            T0 + h,
            T0 + 2 * h,
            T0 + 3 * h,
            T0 + 4 * h,
        )
        await conn.execute(
            "INSERT INTO associations (id, source_trace_id, target_trace_id,"
            " created_at, last_activation) VALUES ('a1', 't1', 't2', $1, $2)",
            T0,
            T0 + h,
        )
        await conn.execute(
            "INSERT INTO persona_facts (id, subject, predicate, object, content,"
            " user_id, created_at, updated_at)"
            " VALUES ('f1', 's', 'p', 'o', 'c', 'u', $1, $2)",
            T0,
            T0 + h,
        )
        await conn.execute(
            "INSERT INTO entity_relations (id, source_entity, relation,"
            " target_entity, created_at) VALUES ('r1', 'a', 'rel', 'b', $1)",
            T0,
        )
        await conn.execute(
            "INSERT INTO concept_embeddings (id, concept, owner_type, owner_id,"
            " embedding, created_at) VALUES ('c1', 'x', 'trace', 't1', $1::vector, $2)",
            VEC,
            T0,
        )
        await conn.execute(
            "INSERT INTO recall_tokens (id, label, created_at, last_activated_at)"
            " VALUES ('k1', 'l', $1, $2)",
            T0,
            T0 + h,
        )
        await conn.execute(
            "INSERT INTO token_stamps (token_id, trace_id, stamped_at)"
            " VALUES ('k1', 't1', $1)",
            T0,
        )


async def _stamps(pool: asyncpg.Pool) -> dict[tuple[str, str], list[Any]]:
    out: dict[tuple[str, str], list[Any]] = {}
    async with pool.acquire() as conn:
        for table, column in SHIFTED_COLUMNS:
            key = _ROW_KEY.get(table, "id")
            rows = await conn.fetch(
                f"SELECT {column} AS v FROM {table} ORDER BY {key}"  # noqa: S608
            )
            out[(table, column)] = [r["v"] for r in rows]
    return out


async def _trace_order(pool: asyncpg.Pool) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id FROM memory_traces ORDER BY created_at")
    return [r["id"] for r in rows]


async def test_shift_is_exact_seconds_across_a_dst_boundary(
    store: asyncpg.Pool,
) -> None:
    await _seed_rows(store)
    before = await _stamps(store)
    order = await _trace_order(store)
    shift = timedelta(days=30)

    await StoreClock(store, now=lambda: FROZEN).advance_to(shift)

    after = await _stamps(store)
    assert len(before) == 15
    for key, values in before.items():
        assert values, f"{key} has no seeded row"
        expected = [None if v is None else v - shift for v in values]
        assert after[key] == expected, key
    assert after[("memory_traces", "created_at")][0] == datetime(
        2026, 10, 13, 12, 0, tzinfo=UTC
    )
    assert after[("memory_traces", "last_retrieval")][0] is None
    assert await _trace_order(store) == order


async def test_every_time_column_is_shifted_or_excluded(store: asyncpg.Pool) -> None:
    async with store.acquire() as conn:
        rows = await conn.fetch(
            "SELECT c.table_name, c.column_name FROM information_schema.columns c"
            " JOIN information_schema.tables t USING (table_schema, table_name)"
            " WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE'"
            " AND c.data_type = ANY($1::text[])",
            _TIME_TYPES,
        )
    schema = {(r["table_name"], r["column_name"]) for r in rows}
    shifted, excluded = set(SHIFTED_COLUMNS), set(EXCLUDED_COLUMNS)

    assert not shifted & excluded
    assert shifted | excluded == schema


async def test_shift_refuses_a_database_outside_the_probe_prefix() -> None:
    db_name = f"clock_guard_{uuid.uuid4().hex[:10]}"
    await _admin(f'CREATE DATABASE "{db_name}"')
    pool = await asyncpg.create_pool(_url(db_name), min_size=1, max_size=2)
    try:
        await pool.execute("CREATE TABLE memory_traces (created_at TIMESTAMPTZ)")
        await pool.execute("INSERT INTO memory_traces VALUES ($1)", T0)

        clock = StoreClock(pool, now=lambda: FROZEN)
        with pytest.raises(ValueError, match="probe_"):
            await clock.advance_to(timedelta(days=1))
        with pytest.raises(ValueError, match="probe_"):
            await clock.settle()

        assert await pool.fetchval("SELECT created_at FROM memory_traces") == T0
    finally:
        await pool.close()
        await _admin(f'DROP DATABASE "{db_name}" WITH (FORCE)')


class _SlowWallClock:
    """Every read costs ten seconds: stepping takes real time."""

    def __init__(self) -> None:
        self.first: datetime | None = None
        self.last = FROZEN

    def __call__(self) -> datetime:
        self.last += timedelta(seconds=10)
        self.first = self.first or self.last
        return self.last


@pytest.mark.parametrize("steps", [1, 48])
async def test_emulated_age_is_independent_of_step_count(
    store: asyncpg.Pool, steps: int
) -> None:
    await _seed_rows(store)
    wall = _SlowWallClock()
    clock = StoreClock(store, now=wall)
    day = timedelta(days=1)

    for i in range(1, steps + 1):
        await clock.advance_to(day * i / steps)

    assert wall.first is not None
    created = (await _stamps(store))[("memory_traces", "created_at")][0]
    age_at_anchor = wall.first - T0
    assert wall.last - created == age_at_anchor + day


async def test_advance_refuses_an_instant_already_passed(store: asyncpg.Pool) -> None:
    await _seed_rows(store)
    before = await _stamps(store)
    clock = StoreClock(store, now=_SlowWallClock())

    with pytest.raises(ValueError, match="already passed"):
        await clock.advance_to(timedelta(seconds=5))

    assert await _stamps(store) == before


def _unit_vector(rng: random.Random) -> list[float]:
    raw = [rng.gauss(0.0, 1.0) for _ in range(768)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


async def _raw_ranks(
    mem: CognitiveMemory, queries: list[list[float]]
) -> list[list[tuple[str, float]]]:
    out: list[list[tuple[str, float]]] = []
    for query in queries:
        ranked = await mem.storage.vectors.search_semantic(query, 20)
        out.append([(t.content, round(sim, 9)) for t, sim in ranked])
    return out


async def test_raw_semantic_ranks_survive_repeated_shifts(
    db_name: str, mem: CognitiveMemory
) -> None:
    # Each shift is a non-HOT update (created_at is indexed): every row
    # gains one dead HNSW entry per shift, and an index scan over them
    # returns a handful of live rows. At fixture scale the planner prefers
    # an exact scan, so it is pinned to the vector index to meet that path.
    for setting in ("enable_seqscan", "enable_bitmapscan", "enable_sort"):
        await _admin(f'ALTER DATABASE "{db_name}" SET {setting} = off')
    await mem.close()
    await mem.connect(_url(db_name))
    rng = random.Random(1889)  # noqa: S311 -- determinism, not crypto
    for i in range(40):
        await mem.storage.traces.store_trace(
            MemoryTrace(content=f"trace-{i}", embedding=_unit_vector(rng), strength=0.8)
        )
    queries = [_unit_vector(rng) for _ in range(5)]
    pool = await mem.storage.pool.get_pool()
    plan = await pool.fetch(
        "EXPLAIN SELECT id FROM memory_traces WHERE strength >= 0.1"
        " ORDER BY embedding <=> $1::vector LIMIT 20",
        VEC,
    )
    assert "idx_traces_embedding" in "\n".join(row[0] for row in plan)
    before = await _raw_ranks(mem, queries)
    clock = StoreClock(pool, now=lambda: FROZEN)

    for step in range(1, 61):
        await clock.advance_to(timedelta(minutes=30) * step)
    await clock.settle()

    assert all(len(ranked) == 20 for ranked in before)
    assert await _raw_ranks(mem, queries) == before


async def test_age_store_steps_a_real_store_to_its_age(mem: CognitiveMemory) -> None:
    await mem.storage.traces.store_trace(
        MemoryTrace(content="aged", embedding=_unit_vector(random.Random(7)))  # noqa: S311
    )
    pool = await mem.storage.pool.get_pool()
    seeded_at = await pool.fetchval("SELECT created_at FROM memory_traces")

    clock = StoreClock(pool)
    aging = Aging(age_days=2, passes_per_day=2)

    passes = await age_store(mem, clock, aging)

    assert passes == 4
    aged_at = await pool.fetchval("SELECT created_at FROM memory_traces")
    # Real stepping time is absorbed: the shift is two days less what elapsed.
    assert timedelta(days=2) - timedelta(seconds=30) < seeded_at - aged_at
    assert seeded_at - aged_at <= timedelta(days=2)
    # The final shift and the settle are not: the store sits past its age.
    assert timedelta(0) < clock.emulated_offset() - aging.span < timedelta(seconds=30)


class _WorkClock:
    """Wall clock that moves only while the database works."""

    def __init__(self) -> None:
        self.now = FROZEN

    def __call__(self) -> datetime:
        return self.now


class _TickingConnection:
    """Each executed statement costs one second of wall time."""

    def __init__(self, conn: Any, wall: _WorkClock) -> None:
        self._conn = conn
        self._wall = wall

    def transaction(self) -> Any:
        return self._conn.transaction()

    async def fetchval(self, query: str) -> Any:
        return await self._conn.fetchval(query)

    async def fetch(self, query: str) -> Any:
        return await self._conn.fetch(query)

    async def execute(self, query: str, *args: object) -> Any:
        self._wall.now += timedelta(seconds=1)
        return await self._conn.execute(query, *args)


class _TickingPool:
    def __init__(self, pool: asyncpg.Pool, wall: _WorkClock) -> None:
        self._pool = pool
        self._wall = wall

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[_TickingConnection, None]:
        async with self._pool.acquire() as conn:
            yield _TickingConnection(conn, self._wall)


async def test_time_spent_in_database_work_is_residual_age(
    store: asyncpg.Pool,
) -> None:
    await _seed_rows(store)
    seeded = (await _stamps(store))[("memory_traces", "created_at")][0]
    wall = _WorkClock()
    pool = cast("asyncpg.Pool[asyncpg.Record]", _TickingPool(store, wall))
    clock = StoreClock(pool, now=wall)
    day = timedelta(days=1)
    # One statement per shifted table, in the shift and in the settle.
    work = timedelta(seconds=len({table for table, _ in SHIFTED_COLUMNS}))

    await clock.advance_to(day)
    # The sample precedes the shift: its own duration is not absorbed.
    assert clock.emulated_offset() == day + work

    await clock.advance_to(2 * day)
    # The next advance absorbs the earlier residue and leaves its own.
    assert clock.emulated_offset() == 2 * day + work
    aged = (await _stamps(store))[("memory_traces", "created_at")][0]
    assert seeded - aged == 2 * day - work

    await clock.settle()
    assert clock.emulated_offset() == 2 * day + 2 * work


async def test_advance_refuses_an_unclassified_time_column(store: asyncpg.Pool) -> None:
    # A migration that adds a time column must not be aged around silently.
    await _seed_rows(store)
    before = await _stamps(store)
    await store.execute(
        "ALTER TABLE memory_traces ADD COLUMN strength_as_of TIMESTAMPTZ"
    )

    with pytest.raises(ValueError, match=r"memory_traces\.strength_as_of"):
        await StoreClock(store, now=lambda: FROZEN).advance_to(timedelta(days=1))

    assert await _stamps(store) == before
