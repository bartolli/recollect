"""Integration: explicit entity merge over the per-user name relation.

merge_entities moves every name row and mention of the source onto the
target and deletes the source; later mentions of either form resolve to
the merged entity.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import EntityNotFoundError
from recollect.models import MemoryTrace, TraceEntity

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [pytest.mark.slow, pytest.mark.asyncio]

SHORT = "auto-injector"
LONG = "epinephrine auto-injector"


def _url(db_name: str) -> str:
    return urlunparse(urlparse(DB_URL)._replace(path=f"/{db_name}"))


async def _admin(sql: str) -> None:
    conn = await asyncpg.connect(_url("postgres"))
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@pytest.fixture()
async def mem() -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"entitymerge_{uuid.uuid4().hex[:10]}"
    await _admin(f'CREATE DATABASE "{db_name}"')
    m = CognitiveMemory()
    await m.connect(_url(db_name))
    try:
        yield m
    finally:
        await m.close()
        await _admin(f'DROP DATABASE "{db_name}" WITH (FORCE)')


async def _trace(
    mem: CognitiveMemory, *mentions: tuple[str, str], user_id: str | None = "u-a"
) -> MemoryTrace:
    trace = MemoryTrace(content="entity mention", user_id=user_id)
    await mem.storage.traces.store_trace(trace)
    await mem.storage.entities.store_trace_entities(
        trace.id,
        [
            TraceEntity(entity_name=name, entity_type=etype, trace_id=trace.id)
            for name, etype in mentions
        ],
    )
    return trace


async def _fetch(mem: CognitiveMemory, sql: str, *args: Any) -> list[asyncpg.Record]:
    pool = await mem.storage.pool.get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(sql, *args)


async def _entity_of(mem: CognitiveMemory, name: str) -> str:
    rows = await _fetch(mem, "SELECT entity_id FROM entity_names WHERE name = $1", name)
    return str(rows[0]["entity_id"])


async def _state(mem: CognitiveMemory) -> list[list[tuple[Any, ...]]]:
    tables = (
        "SELECT id, user_id, entity_type FROM entities ORDER BY id",
        "SELECT entity_id, name FROM entity_names ORDER BY name",
        "SELECT entity_name, trace_id, entity_id FROM trace_entities"
        " ORDER BY entity_name, trace_id",
    )
    return [[tuple(r) for r in await _fetch(mem, sql)] for sql in tables]


async def _merged_pair(mem: CognitiveMemory) -> tuple[MemoryTrace, MemoryTrace, str]:
    a = await _trace(mem, (SHORT, "unknown"))
    b = await _trace(mem, (LONG, "product"))
    into = await _entity_of(mem, LONG)
    await mem.merge_entities(await _entity_of(mem, SHORT), into)
    return a, b, into


class TestMergeMovesIdentity:
    async def test_merge_carries_both_names_and_mentions(
        self, mem: CognitiveMemory
    ) -> None:
        await _trace(mem, (SHORT, "unknown"))
        await _trace(mem, (LONG, "product"))
        source, into = await _entity_of(mem, SHORT), await _entity_of(mem, LONG)
        assert source != into

        merged = await mem.merge_entities(source, into)

        assert (merged.id, merged.names, merged.trace_count) == (into, [SHORT, LONG], 2)
        assert await mem.storage.entities.get_entity(source) is None
        mentions = await _fetch(mem, "SELECT DISTINCT entity_id FROM trace_entities")
        assert [m["entity_id"] for m in mentions] == [into]
        listed = await mem.entities(user_id="u-a")
        assert [(e.id, e.names, e.trace_count) for e in listed] == [
            (into, [SHORT, LONG], 2)
        ]

    async def test_new_mentions_of_either_form_resolve_to_merged(
        self, mem: CognitiveMemory
    ) -> None:
        a, b, into = await _merged_pair(mem)
        c = await _trace(mem, ("Auto-Injector ", "unknown"))
        sql = "SELECT entity_id FROM trace_entities WHERE trace_id = $1"
        assert [m["entity_id"] for m in await _fetch(mem, sql, c.id)] == [into]
        for form in (SHORT, LONG):
            linked = await mem.storage.entities.get_traces_by_entity(
                form, user_id="u-a"
            )
            assert sorted(linked) == sorted([a.id, b.id, c.id])

    async def test_batch_with_both_names_stores_one_identity(
        self, mem: CognitiveMemory
    ) -> None:
        _, _, into = await _merged_pair(mem)
        d = await _trace(mem, (LONG, "unknown"), (SHORT, "unknown"))
        mentions = await _fetch(
            mem,
            "SELECT entity_name, entity_id FROM trace_entities WHERE trace_id = $1"
            " ORDER BY entity_name",
            d.id,
        )
        assert [(m["entity_name"], m["entity_id"]) for m in mentions] == [
            (SHORT, into),
            (LONG, into),
        ]
        assert len(await _fetch(mem, "SELECT id FROM entities")) == 1


class TestMergeRefusals:
    async def test_refusals_leave_state_unchanged(self, mem: CognitiveMemory) -> None:
        await _trace(mem, (SHORT, "unknown"))
        await _trace(mem, (SHORT, "unknown"), user_id="u-b")
        owned = await _fetch(mem, "SELECT user_id, id FROM entities")
        mine, theirs = (dict(owned)[user] for user in ("u-a", "u-b"))
        before = await _state(mem)

        with pytest.raises(ValueError, match="itself"):
            await mem.merge_entities(mine, mine)
        with pytest.raises(EntityNotFoundError):
            await mem.merge_entities("no-such-entity", mine)
        with pytest.raises(ValueError, match="owners"):
            await mem.merge_entities(mine, theirs)
        store = mem.storage.entities
        assert await store.merge_entities(mine, mine) is False
        assert await store.merge_entities(mine, "no-such-entity") is False
        assert await store.merge_entities(theirs, mine) is False

        assert await _state(mem) == before


class TestMergeType:
    async def test_unknown_target_takes_typed_source_type(
        self, mem: CognitiveMemory
    ) -> None:
        await _trace(mem, (SHORT, "product"))
        await _trace(mem, (LONG, "unknown"))
        merged = await mem.merge_entities(
            await _entity_of(mem, SHORT), await _entity_of(mem, LONG)
        )
        assert merged.entity_type == "product"

    async def test_typed_target_keeps_its_type(self, mem: CognitiveMemory) -> None:
        await _trace(mem, (SHORT, "device"))
        await _trace(mem, (LONG, "product"))
        merged = await mem.merge_entities(
            await _entity_of(mem, SHORT), await _entity_of(mem, LONG)
        )
        assert merged.entity_type == "product"


class TestEraseAfterMerge:
    async def test_erase_drops_unused_names_then_the_entity(
        self, mem: CognitiveMemory
    ) -> None:
        a, b, into = await _merged_pair(mem)

        assert await mem.erase(a.id) is True
        remaining = await mem.storage.entities.get_entity(into)
        assert remaining is not None
        assert (remaining.names, remaining.trace_count) == ([LONG], 1)

        assert await mem.erase(b.id) is True
        assert await mem.storage.entities.get_entity(into) is None
        assert await _fetch(mem, "SELECT name FROM entity_names") == []
