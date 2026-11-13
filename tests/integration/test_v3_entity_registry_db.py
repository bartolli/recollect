"""Integration: per-user entity identity behind trace_entities.

Every mention resolves through one entity_names row per (name_key, user_id)
inside store_trace_entities; linking reads match through that identity, so
case and whitespace variants of one name link.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.bootstrap.migrations import m007_entity_registry
from recollect.core import CognitiveMemory
from recollect.exceptions import StorageError
from recollect.models import KnownEntity, MemoryTrace, TraceEntity

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [pytest.mark.slow, pytest.mark.asyncio]


def _scratch_url(db_name: str) -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def _admin_url() -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path="/postgres"))


@pytest.fixture()
async def scratch_db() -> AsyncGenerator[str, None]:
    db_name = f"entityreg_{uuid.uuid4().hex[:10]}"
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


async def _rows(mem: CognitiveMemory, sql: str) -> list[asyncpg.Record]:
    pool = await mem.storage.pool.get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(sql)


class TestIdentityResolution:
    async def test_surface_variants_share_one_entity(
        self, mem: CognitiveMemory
    ) -> None:
        first = await _trace(mem, ("spaCy", "unknown"))
        second = await _trace(mem, ("  spacy ", "unknown"))
        entities = await _rows(mem, "SELECT id FROM entities")
        assert len(entities) == 1
        names = await _rows(mem, "SELECT entity_id, name FROM entity_names")
        assert [(n["entity_id"], n["name"]) for n in names] == [
            (entities[0]["id"], "spaCy")
        ]
        mentions = await _rows(
            mem,
            "SELECT trace_id, entity_id FROM trace_entities",
        )
        assert {m["trace_id"] for m in mentions} == {first.id, second.id}
        assert {m["entity_id"] for m in mentions} == {entities[0]["id"]}

    async def test_users_partition_identity(self, mem: CognitiveMemory) -> None:
        a = await _trace(mem, ("Alex", "person"), user_id="u-a")
        await _trace(mem, ("Alex", "person"), user_id="u-b")
        await _trace(mem, ("Alex", "person"), user_id=None)
        await _trace(mem, ("alex", "person"), user_id=None)
        owners = await _rows(mem, "SELECT user_id FROM entities")
        assert sorted(o["user_id"] or "" for o in owners) == ["", "u-a", "u-b"]
        name_owners = await _rows(
            mem,
            "SELECT en.user_id IS NOT DISTINCT FROM e.user_id AS same"
            " FROM entity_names en JOIN entities e ON e.id = en.entity_id",
        )
        assert [n["same"] for n in name_owners] == [True, True, True]
        linked = await mem.storage.entities.get_traces_by_entity("Alex", user_id="u-a")
        assert linked == [a.id]

    async def test_first_typed_mention_sets_registry_type(
        self, mem: CognitiveMemory
    ) -> None:
        await _trace(mem, ("Acme", "unknown"))
        await _trace(mem, ("acme", "organization"))
        await _trace(mem, ("ACME", "product"))
        types = await _rows(mem, "SELECT entity_type FROM entities")
        assert [t["entity_type"] for t in types] == ["organization"]
        mentions = await _rows(
            mem, "SELECT entity_name, entity_type FROM trace_entities"
        )
        assert {(m["entity_name"], m["entity_type"]) for m in mentions} == {
            ("Acme", "unknown"),
            ("acme", "organization"),
            ("ACME", "product"),
        }

    async def test_batch_repeating_a_key_yields_one_entity(
        self, mem: CognitiveMemory
    ) -> None:
        trace = await _trace(
            mem, ("spaCy", "unknown"), ("spaCy", "unknown"), ("SPACY", "product")
        )
        entities = await _rows(
            mem,
            "SELECT en.name, e.entity_type FROM entities e"
            " JOIN entity_names en ON en.entity_id = e.id",
        )
        assert [(e["name"], e["entity_type"]) for e in entities] == [
            ("spaCy", "product")
        ]
        mentions = await _rows(mem, "SELECT entity_name FROM trace_entities")
        assert sorted(m["entity_name"] for m in mentions) == ["SPACY", "spaCy"]
        linked = await mem.storage.entities.get_traces_by_entity("spacy")
        assert linked == [trace.id]

    async def test_missing_trace_fails_and_stores_nothing(
        self, mem: CognitiveMemory
    ) -> None:
        with pytest.raises(StorageError):
            await mem.storage.entities.store_trace_entities(
                "no-such-trace",
                [TraceEntity(entity_name="Ghost", trace_id="no-such-trace")],
            )
        counts = await _rows(
            mem,
            "SELECT (SELECT count(*) FROM entities),"
            " (SELECT count(*) FROM entity_names),"
            " (SELECT count(*) FROM trace_entities)",
        )
        assert tuple(counts[0]) == (0, 0, 0)


class TestIdentityLinking:
    async def test_normalized_lookup_finds_variant_mention(
        self, mem: CognitiveMemory
    ) -> None:
        trace = await _trace(
            mem,
            ("spaCy ", "product"),
            ("Oak   Street", "place"),
            ("\tMarchetti\nand Sons\n", "organization"),
        )
        linked = await mem.storage.entities.get_traces_by_entity("spacy", user_id="u-a")
        assert linked == [trace.id]
        for variant in ("oak street", "marchetti and sons"):
            assert await mem.storage.entities.get_traces_by_entity(variant) == [
                trace.id
            ]

    async def test_mention_without_identity_does_not_match(
        self, mem: CognitiveMemory
    ) -> None:
        await _trace(mem, ("Legacy", "unknown"))
        pool = await mem.storage.pool.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE trace_entities SET entity_id = NULL")
        assert await mem.storage.entities.get_traces_by_entity("Legacy") == []


class TestKnownEntities:
    async def test_lists_own_entities_most_mentioned_first(
        self, mem: CognitiveMemory
    ) -> None:
        await _trace(mem, ("Borealis", "project"), ("Alex", "person"))
        await _trace(mem, ("alex", "person"), ("ALEX ", "person"))
        await _trace(mem, ("Alex", "person"), user_id="u-b")
        entities = await mem.entities(user_id="u-a")
        assert all(isinstance(e, KnownEntity) for e in entities)
        assert [(e.names, e.name, e.trace_count) for e in entities] == [
            (["Alex"], "Alex", 2),
            (["Borealis"], "Borealis", 1),
        ]
        assert {e.user_id for e in entities} == {"u-a"}
        assert entities[0].entity_type == "person"


class TestEraseReleasesEntities:
    async def test_erase_drops_only_unmentioned_entities(
        self, mem: CognitiveMemory
    ) -> None:
        erased = await _trace(mem, ("Solo", "person"), ("Shared", "place"))
        await _trace(mem, ("shared", "place"))
        assert await mem.erase(erased.id) is True
        names = [e.name for e in await mem.entities(user_id="u-a")]
        assert names == ["Shared"]

    async def test_mentioned_entity_cannot_be_deleted(
        self, mem: CognitiveMemory
    ) -> None:
        await _trace(mem, ("Solo", "person"))
        pool = await mem.storage.pool.get_pool()
        async with pool.acquire() as conn:
            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await conn.execute("DELETE FROM entities")
        assert await mem.storage.entities.delete_unmentioned_entities() == 0

    async def test_forget_keeps_entities(self, mem: CognitiveMemory) -> None:
        trace = await _trace(mem, ("Solo", "person"))
        await mem.forget(trace.id)
        entities = await mem.entities(user_id="u-a")
        assert [(e.name, e.trace_count) for e in entities] == [("Solo", 1)]


_REGISTRY_SHAPE = (
    "SELECT table_name, column_name, data_type, is_nullable, column_default,"
    " generation_expression FROM information_schema.columns"
    " WHERE table_name IN ('entities', 'entity_names')"
    " OR (table_name = 'trace_entities' AND column_name = 'entity_id')"
    " ORDER BY table_name, column_name",
    "SELECT indexname, indexdef FROM pg_indexes WHERE indexname IN"
    " ('idx_entity_names_key_user', 'idx_entity_names_entity',"
    " 'idx_trace_entities_entity') ORDER BY indexname",
    "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint"
    " WHERE conrelid IN ('entities'::regclass, 'entity_names'::regclass,"
    " 'trace_entities'::regclass) ORDER BY conname",
)


async def _registry_shape(conn: asyncpg.Connection) -> list[list[tuple[object, ...]]]:
    return [[tuple(r) for r in await conn.fetch(sql)] for sql in _REGISTRY_SHAPE]


class TestM007EntityRegistry:
    async def test_twin_matches_fresh_schema_and_reapplies(
        self, scratch_db: str
    ) -> None:
        m = CognitiveMemory()
        await m.connect(scratch_db)
        await m.close()
        conn = await asyncpg.connect(scratch_db)
        try:
            fresh = await _registry_shape(conn)
            columns, indexes, _ = fresh
            assert [c[1] for c in columns if c[0] == "entities"] == [
                "created_at",
                "entity_type",
                "id",
                "user_id",
            ]
            assert len(indexes) == 3
            await m007_entity_registry.up(conn)
            assert await _registry_shape(conn) == fresh
            await conn.execute("ALTER TABLE trace_entities DROP COLUMN entity_id")
            await conn.execute("DROP TABLE entity_names")
            await conn.execute("DROP TABLE entities")
            await m007_entity_registry.up(conn)
            await m007_entity_registry.up(conn)
            assert await _registry_shape(conn) == fresh
        finally:
            await conn.close()
