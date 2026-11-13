"""Integration: entity_relations as per-trace assertions between entities.

store_relation keeps one row per asserting trace and resolves each
endpoint through that trace's own mentions; get_edges aggregates the
visible assertions touching one entity.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.bootstrap.migrations import m008_relation_assertions
from recollect.core import CognitiveMemory
from recollect.exceptions import StorageError
from recollect.models import EntityEdge, EntityRelation, MemoryTrace, TraceEntity

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

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
async def scratch_db() -> AsyncGenerator[str, None]:
    db_name = f"entityrel_{uuid.uuid4().hex[:10]}"
    await _admin(f'CREATE DATABASE "{db_name}"')
    try:
        yield _url(db_name)
    finally:
        await _admin(f'DROP DATABASE "{db_name}" WITH (FORCE)')


@pytest.fixture()
async def mem(scratch_db: str) -> AsyncGenerator[CognitiveMemory, None]:
    m = CognitiveMemory()
    await m.connect(scratch_db)
    try:
        yield m
    finally:
        await m.close()


async def _trace(
    mem: CognitiveMemory, *names: str, user_id: str | None = "u-a"
) -> MemoryTrace:
    trace = MemoryTrace(content="relation source", user_id=user_id)
    await mem.storage.traces.store_trace(trace)
    await mem.storage.entities.store_trace_entities(
        trace.id, [TraceEntity(entity_name=n, trace_id=trace.id) for n in names]
    )
    return trace


async def _assert(
    mem: CognitiveMemory,
    trace: MemoryTrace,
    source: str,
    target: str,
    *,
    relation: str = "works_at",
    weight: float = 0.5,
) -> None:
    await mem.storage.entity_relations.store_relation(
        EntityRelation(
            source_entity=source,
            relation=relation,
            target_entity=target,
            context=f"{source} {relation} {target}",
            weight=weight,
            source_trace_id=trace.id,
        )
    )


async def _fetch(mem: CognitiveMemory, sql: str, *args: Any) -> list[asyncpg.Record]:
    pool = await mem.storage.pool.get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(sql, *args)


async def _entity_of(mem: CognitiveMemory, name: str, user_id: str = "u-a") -> str:
    rows = await _fetch(
        mem,
        "SELECT entity_id FROM entity_names WHERE name = $1 AND user_id = $2",
        name,
        user_id,
    )
    return str(rows[0]["entity_id"])


async def _ids(mem: CognitiveMemory, trace: MemoryTrace) -> list[tuple[Any, ...]]:
    rows = await _fetch(
        mem,
        "SELECT source_entity, source_entity_id, target_entity, target_entity_id"
        " FROM entity_relations WHERE source_trace_id = $1 ORDER BY target_entity",
        trace.id,
    )
    return [tuple(r) for r in rows]


class TestEndpointResolution:
    async def test_endpoints_resolve_through_the_trace_own_mentions(
        self, mem: CognitiveMemory
    ) -> None:
        other = await _trace(mem, "Borealis")
        trace = await _trace(mem, "Alex", "Acme")
        await _assert(mem, trace, "Alex", "Acme")
        await _assert(mem, trace, "Alex", "Borealis", relation="leads")
        await _assert(mem, trace, "Alex", "Paris", relation="lives_in")
        alex, acme = await _entity_of(mem, "Alex"), await _entity_of(mem, "Acme")
        assert await _ids(mem, trace) == [
            ("Alex", alex, "Acme", acme),
            ("Alex", alex, "Borealis", None),
            ("Alex", alex, "Paris", None),
        ]
        assert await _ids(mem, other) == []
        entities = await _fetch(mem, "SELECT count(*) AS n FROM entities")
        assert entities[0]["n"] == 3

    async def test_case_and_whitespace_variants_resolve(
        self, mem: CognitiveMemory
    ) -> None:
        trace = await _trace(mem, "Alex", "Acme Corp")
        await _assert(mem, trace, " alex", "ACME\t corp ")
        alex, acme = await _entity_of(mem, "Alex"), await _entity_of(mem, "Acme Corp")
        assert await _ids(mem, trace) == [(" alex", alex, "ACME\t corp ", acme)]

    async def test_repeated_triple_in_one_trace_keeps_the_greater_weight(
        self, mem: CognitiveMemory
    ) -> None:
        trace = await _trace(mem, "Alex", "Acme")
        await _assert(mem, trace, "Alex", "Acme", weight=0.4)
        await _assert(mem, trace, "Alex", "Acme", weight=0.9)
        await _assert(mem, trace, "Alex", "Acme", weight=0.6)
        rows = await _fetch(mem, "SELECT weight FROM entity_relations")
        assert [r["weight"] for r in rows] == [0.9]

    async def test_missing_trace_fails_and_stores_nothing(
        self, mem: CognitiveMemory
    ) -> None:
        ghost = MemoryTrace(id="no-such-trace", content="never stored")
        with pytest.raises(StorageError):
            await _assert(mem, ghost, "Alex", "Acme")
        assert await _fetch(mem, "SELECT id FROM entity_relations") == []


def _shape(edges: list[EntityEdge]) -> list[tuple[str, str, str, float, int]]:
    return [(e.source, e.relation, e.target, e.weight, e.trace_count) for e in edges]


class TestAssertionsPerTrace:
    async def test_second_trace_adds_a_row_and_counts_once_more(
        self, mem: CognitiveMemory
    ) -> None:
        first = await _trace(mem, "Alex", "Acme")
        second = await _trace(mem, "alex", "Acme")
        await _assert(mem, first, "Alex", "Acme", weight=0.6)
        await _assert(mem, second, "Alex", "Acme", weight=0.8)
        rows = await _fetch(
            mem,
            "SELECT source_trace_id, context, weight FROM entity_relations"
            " ORDER BY weight",
        )
        assert [tuple(r) for r in rows] == [
            (first.id, "Alex works_at Acme", 0.6),
            (second.id, "Alex works_at Acme", 0.8),
        ]
        edges = await mem.entity_edges(await _entity_of(mem, "Alex"))
        assert _shape(edges) == [("Alex", "works_at", "Acme", 0.8, 2)]
        assert edges[0].target_entity_id == await _entity_of(mem, "Acme")

    async def test_users_assert_separately(self, mem: CognitiveMemory) -> None:
        mine = await _trace(mem, "Alex", "Acme", user_id="u-a")
        theirs = await _trace(mem, "Alex", "Acme", user_id="u-b")
        await _assert(mem, mine, "Alex", "Acme")
        await _assert(mem, theirs, "Alex", "Acme")
        rows = await _fetch(mem, "SELECT source_trace_id FROM entity_relations")
        assert sorted(r["source_trace_id"] for r in rows) == sorted(
            [mine.id, theirs.id]
        )
        for user in ("u-a", "u-b"):
            alex = await _entity_of(mem, "Alex", user)
            for scope in (user, None):
                edges = await mem.entity_edges(alex, user_id=scope)
                assert _shape(edges) == [("Alex", "works_at", "Acme", 0.5, 1)]
        other_scope = await mem.entity_edges(
            await _entity_of(mem, "Alex"), user_id="u-b"
        )
        assert other_scope == []


class TestEdgeReader:
    async def test_both_ends_literal_targets_limit_and_order(
        self, mem: CognitiveMemory
    ) -> None:
        one = await _trace(mem, "Alex", "Acme", "Bo")
        two = await _trace(mem, "Alex", "Acme", "Bo")
        await _assert(mem, one, "Alex", "Acme", weight=0.3)
        await _assert(mem, two, "Alex", "Acme", weight=0.4)
        await _assert(mem, one, "Bo", "Alex", relation="mentors", weight=0.9)
        await _assert(mem, one, "Alex", "Paris", relation="lives_in", weight=0.9)
        await _assert(mem, one, "Alex", "Bo", relation="knows", weight=0.9)
        await _assert(mem, one, "Bo", "Acme", relation="works_at", weight=1.0)
        alex = await _entity_of(mem, "Alex")
        edges = await mem.entity_edges(alex)
        assert _shape(edges) == [
            ("Alex", "works_at", "Acme", 0.4, 2),
            ("Alex", "knows", "Bo", 0.9, 1),
            ("Alex", "lives_in", "Paris", 0.9, 1),
            ("Bo", "mentors", "Alex", 0.9, 1),
        ]
        literal = edges[2]
        assert (literal.source_entity_id, literal.target_entity_id) == (alex, None)
        assert edges[3].target_entity_id == alex
        assert await mem.entity_edges(alex, limit=2) == edges[:2]


class TestForgetHidesAssertions:
    async def test_forgotten_traces_drop_out_of_edges_but_keep_rows(
        self, mem: CognitiveMemory
    ) -> None:
        first = await _trace(mem, "Alex", "Acme")
        second = await _trace(mem, "Alex", "Acme")
        await _assert(mem, first, "Alex", "Acme")
        await _assert(mem, second, "Alex", "Acme")
        alex = await _entity_of(mem, "Alex")

        await mem.forget(first.id)
        assert _shape(await mem.entity_edges(alex)) == [
            ("Alex", "works_at", "Acme", 0.5, 1)
        ]
        await mem.forget(second.id)
        assert await mem.entity_edges(alex) == []
        rows = await _fetch(mem, "SELECT id FROM entity_relations")
        assert len(rows) == 2


class TestEraseRemovesAssertions:
    async def test_erase_cascades_assertions_and_the_sweep_still_runs(
        self, mem: CognitiveMemory
    ) -> None:
        erased = await _trace(mem, "Solo", "Shared")
        kept = await _trace(mem, "shared", "Other")
        await _assert(mem, erased, "Solo", "Shared", relation="visits")
        await _assert(mem, kept, "shared", "Other", relation="near")

        assert await mem.erase(erased.id) is True

        rows = await _fetch(
            mem, "SELECT source_trace_id, context FROM entity_relations"
        )
        assert [tuple(r) for r in rows] == [(kept.id, "shared near Other")]
        names = sorted(e.name for e in await mem.entities(user_id="u-a"))
        assert names == ["Other", "Shared"]


SHORT = "auto-injector"
LONG = "epinephrine auto-injector"


class TestMergeRepointsAssertions:
    async def test_merge_moves_both_ends_and_aggregates_edges(
        self, mem: CognitiveMemory
    ) -> None:
        a = await _trace(mem, SHORT, "Alex")
        b = await _trace(mem, LONG, "Alex")
        await _assert(mem, a, "Alex", SHORT, relation="carries", weight=0.6)
        await _assert(mem, b, "Alex", LONG, relation="carries", weight=0.7)
        await _assert(mem, a, SHORT, "Alex", relation="prescribed_to")
        source, into = await _entity_of(mem, SHORT), await _entity_of(mem, LONG)

        await mem.merge_entities(source, into)

        left = await _fetch(
            mem,
            "SELECT id FROM entity_relations"
            " WHERE $1 IN (source_entity_id, target_entity_id)",
            source,
        )
        assert left == []
        edges = await mem.entity_edges(into)
        assert _shape(edges) == [
            ("Alex", "carries", SHORT, 0.7, 2),
            (SHORT, "prescribed_to", "Alex", 0.5, 1),
        ]
        assert (edges[0].target_entity_id, edges[1].source_entity_id) == (into, into)


# Column order is omitted: a dropped and re-added column takes a new attnum.
_RELATION_SHAPE = (
    "SELECT column_name, data_type, is_nullable, column_default"
    " FROM information_schema.columns WHERE table_name = 'entity_relations'"
    " ORDER BY column_name",
    "SELECT indexname, indexdef FROM pg_indexes"
    " WHERE tablename = 'entity_relations' ORDER BY indexname",
    "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint"
    " WHERE conrelid = 'entity_relations'::regclass ORDER BY conname",
)

_OLD_SHAPE = """
DROP INDEX idx_entity_rel_assertion;
DROP INDEX idx_entity_rel_source_id;
DROP INDEX idx_entity_rel_target_id;
DROP INDEX idx_trace_entities_trace;
ALTER TABLE entity_relations
    DROP COLUMN source_entity_id, DROP COLUMN target_entity_id,
    DROP CONSTRAINT entity_relations_source_trace_id_fkey,
    ADD FOREIGN KEY (source_trace_id) REFERENCES memory_traces(id) ON DELETE SET NULL;
CREATE INDEX idx_entity_rel_source ON entity_relations(source_entity);
CREATE INDEX idx_entity_rel_target ON entity_relations(target_entity);
CREATE UNIQUE INDEX idx_entity_rel_pair
    ON entity_relations(source_entity, relation, target_entity);
INSERT INTO memory_traces (id, content) VALUES ('t-old', 'legacy');
INSERT INTO entity_relations (id, source_entity, relation, target_entity,
    source_trace_id) VALUES ('r-old', 'Alex', 'works_at', 'Acme', 't-old'),
    ('r-orphan', 'Alex', 'lives_in', 'Paris', NULL);
"""


async def _relation_shape(conn: asyncpg.Connection) -> list[list[tuple[object, ...]]]:
    return [[tuple(r) for r in await conn.fetch(sql)] for sql in _RELATION_SHAPE]


class TestM008RelationAssertions:
    async def test_twin_brings_old_shape_to_fresh_and_reapplies(
        self, scratch_db: str
    ) -> None:
        m = CognitiveMemory()
        await m.connect(scratch_db)
        await m.close()
        conn = await asyncpg.connect(scratch_db)
        try:
            fresh = await _relation_shape(conn)
            _, indexes, constraints = fresh
            assert [i[0] for i in indexes] == [
                "entity_relations_pkey",
                "idx_entity_rel_assertion",
                "idx_entity_rel_source_id",
                "idx_entity_rel_target_id",
            ]
            assert (
                "entity_relations_source_trace_id_fkey",
                "FOREIGN KEY (source_trace_id) REFERENCES memory_traces(id)"
                " ON DELETE CASCADE",
            ) in constraints
            await conn.execute(_OLD_SHAPE)

            await m008_relation_assertions.up(conn)
            await m008_relation_assertions.up(conn)

            assert await _relation_shape(conn) == fresh
            assert await conn.fetchval(
                "SELECT indexdef FROM pg_indexes"
                " WHERE indexname = 'idx_trace_entities_trace'"
            )
            rows = await conn.fetch(
                "SELECT id, source_entity_id, target_entity_id FROM entity_relations"
                " ORDER BY id"
            )
            assert [tuple(r) for r in rows] == [
                ("r-old", None, None),
                ("r-orphan", None, None),
            ]
            await conn.execute("DELETE FROM memory_traces WHERE id = 't-old'")
            left = await conn.fetch("SELECT id FROM entity_relations")
            assert [r["id"] for r in left] == ["r-orphan"]
        finally:
            await conn.close()
