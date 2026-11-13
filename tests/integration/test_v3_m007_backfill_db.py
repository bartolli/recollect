"""Integration: m007 backfills entity_id on mentions written before the registry.

The fixture builds the m006 shape through the runner, seeds legacy mentions,
then bootstraps main. Trace ids and insert order run against created_at, and
the oldest surface form sorts first under neither C nor en_US collation, so
only the chronological replay order passes.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.bootstrap.migration import MigrationRegistry
from recollect.bootstrap.migrations import default_registry, m007_entity_registry
from recollect.bootstrap.runner import apply
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import Entity, ExtractionResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    _Upgraded = tuple[asyncpg.Pool[asyncpg.Record], str]

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [pytest.mark.slow, pytest.mark.asyncio]

_EMB_DIM = 768

_TO_M006_SHAPE = """
ALTER TABLE entity_relations
    DROP COLUMN source_entity_id, DROP COLUMN target_entity_id;
ALTER TABLE trace_entities DROP COLUMN entity_id;
DROP TABLE entity_names;
DROP TABLE entities;
"""

_LEGACY_SEED = """
INSERT INTO memory_traces (id, content, user_id, created_at) VALUES
    ('t1-new', 'legacy', 'u-a', '2026-03-01T00:00:00Z'),
    ('t2-mid', 'legacy', 'u-a', '2026-02-01T00:00:00Z'),
    ('t3-old', 'legacy', 'u-a', '2026-01-01T00:00:00Z'),
    ('tb', 'legacy', 'u-b', '2026-01-15T00:00:00Z'),
    ('tn', 'legacy', NULL, '2026-01-20T00:00:00Z');
INSERT INTO trace_entities (entity_name, entity_type, trace_id) VALUES
    ('ACME', 'organization', 't1-new'),
    ('acme', 'product', 't2-mid'),
    ('oak   street', NULL, 't2-mid'),
    ('Acme', NULL, 't3-old'),
    ('Oak Street', 'unknown', 't3-old'),
    ('Oak Street', 'place', 'tb'),
    ('oak street', 'place', 'tn');
"""

_SNAPSHOT = (
    "SELECT * FROM entities ORDER BY id",
    "SELECT * FROM entity_names ORDER BY entity_id, name_key",
    "SELECT * FROM trace_entities ORDER BY trace_id, entity_name",
)


def _db_url(db_name: str) -> str:
    return urlunparse(urlparse(DB_URL)._replace(path=f"/{db_name}"))


def _pre_m007_registry() -> MigrationRegistry:
    registry = MigrationRegistry()
    for migration in default_registry():
        if migration.name == m007_entity_registry.name:
            break
        registry.register(migration)
    return registry


@pytest.fixture()
async def upgraded() -> AsyncGenerator[_Upgraded, None]:
    db_name = f"m007fill_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_db_url("postgres"))
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(_db_url(db_name), min_size=1, max_size=4)
    try:
        await apply(pool, _pre_m007_registry())
        async with pool.acquire() as conn:
            await conn.execute(_TO_M006_SHAPE)
            await conn.execute(_LEGACY_SEED)
        result = await apply(pool, default_registry())
        assert result.applied == ["m007_entity_registry", "m008_relation_assertions"]
        yield pool, _db_url(db_name)
    finally:
        await pool.close()
        admin = await asyncpg.connect(_db_url("postgres"))
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


def _memory(extracted: list[Entity]) -> CognitiveMemory:
    embeddings = AsyncMock()
    embeddings.generate_embedding = AsyncMock(return_value=[0.1] * _EMB_DIM)
    embeddings.dimensions = _EMB_DIM
    embeddings.contract = lambda: ("nomic-ai/nomic-embed-text-v1.5", "v0.7")
    extractor = AsyncMock()
    extractor.extract = AsyncMock(return_value=ExtractionResult(entities=extracted))
    config = MemoryConfig()
    # Both call the LLM after linking; neither feeds entity associations.
    config._set("persona.auto_extract", False)
    config._set("recall_tokens.enabled", False)
    return CognitiveMemory(embeddings=embeddings, extractor=extractor, config=config)


async def _rows(
    pool: asyncpg.Pool[asyncpg.Record], sql: str, *args: object
) -> list[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch(sql, *args)


class TestLegacyMentionsResolve:
    async def test_every_mention_resolves_per_key_and_owner(
        self, upgraded: _Upgraded
    ) -> None:
        pool, _ = upgraded
        mentions = await _rows(
            pool, "SELECT trace_id, entity_name, entity_id FROM trace_entities"
        )
        ids = {(m["trace_id"], m["entity_name"]): m["entity_id"] for m in mentions}
        assert None not in ids.values()
        assert ids["t3-old", "Oak Street"] == ids["t2-mid", "oak   street"]
        assert ids["t3-old", "Acme"] == ids["t2-mid", "acme"] == ids["t1-new", "ACME"]
        oak_by_owner = {ids["t3-old", "Oak Street"], ids["tb", "Oak Street"]}
        oak_by_owner.add(ids["tn", "oak street"])
        assert len(oak_by_owner) == 3
        owned = await _rows(
            pool,
            "SELECT bool_and(e.user_id IS NOT DISTINCT FROM mt.user_id"
            " AND en.user_id IS NOT DISTINCT FROM mt.user_id) AS same,"
            " (SELECT count(*) FROM entities) AS entities,"
            " (SELECT count(*) FROM entity_names) AS names"
            " FROM trace_entities te"
            " JOIN memory_traces mt ON mt.id = te.trace_id"
            " JOIN entities e ON e.id = te.entity_id"
            " JOIN entity_names en ON en.entity_id = e.id",
        )
        assert tuple(owned[0]) == (True, 4, 4)

    async def test_first_form_names_and_first_typed_mention_types(
        self, upgraded: _Upgraded
    ) -> None:
        pool, _ = upgraded
        registered = await _rows(
            pool,
            "SELECT en.name, e.entity_type, e.created_at = old.created_at"
            " AND en.created_at = old.created_at AS oldest"
            " FROM entities e JOIN entity_names en ON en.entity_id = e.id,"
            " (SELECT created_at FROM memory_traces WHERE id = 't3-old') old"
            " WHERE e.user_id = 'u-a' ORDER BY en.name_key",
        )
        assert [tuple(r) for r in registered] == [
            ("Acme", "product", True),
            ("Oak Street", "unknown", True),
        ]


class TestUpgradedStoreLinksNewToOld:
    async def test_new_trace_forms_entity_associations_with_legacy_traces(
        self, upgraded: _Upgraded
    ) -> None:
        pool, dsn = upgraded
        mem = _memory([Entity(name="OAK  street", entity_type="place")])
        await mem.connect(dsn)
        try:
            trace = await mem.experience("Walked down Oak Street", user_id="u-a")
            linked = await mem.storage.entities.get_traces_by_entity(
                "oak street", user_id="u-a"
            )
        finally:
            await mem.close()
        assert sorted(linked) == sorted(["t2-mid", "t3-old", trace.id])
        edges = await _rows(
            pool,
            "SELECT CASE WHEN source_trace_id = $1 THEN target_trace_id"
            " ELSE source_trace_id END AS other FROM associations"
            " WHERE association_type = 'entity'"
            " AND $1 IN (source_trace_id, target_trace_id)",
            trace.id,
        )
        assert sorted(e["other"] for e in edges) == ["t2-mid", "t3-old"]


class TestBackfillReapplies:
    async def test_second_run_changes_no_row(self, upgraded: _Upgraded) -> None:
        pool, _ = upgraded
        async with pool.acquire() as conn:
            before = [[tuple(r) for r in await conn.fetch(s)] for s in _SNAPSHOT]
            async with conn.transaction():
                await m007_entity_registry.up(conn)
            after = [[tuple(r) for r in await conn.fetch(s)] for s in _SNAPSHOT]
        assert [len(rows) for rows in before] == [4, 4, 7]
        assert after == before

    async def test_null_mention_of_registered_key_reuses_its_entity(
        self, upgraded: _Upgraded
    ) -> None:
        pool, _ = upgraded
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_traces (id, content, user_id, created_at)"
                " VALUES ('t4-late', 'late', 'u-a', '2026-04-01T00:00:00Z');"
                " INSERT INTO trace_entities (entity_name, entity_type, trace_id)"
                " VALUES ('ACME  ', 'person', 't4-late')"
            )
            async with conn.transaction():
                await m007_entity_registry.up(conn)
        acme = await _rows(
            pool,
            "SELECT te.trace_id, te.entity_id, e.entity_type,"
            " (SELECT count(*) FROM entity_names) AS names"
            " FROM trace_entities te LEFT JOIN entities e ON e.id = te.entity_id"
            " WHERE te.trace_id IN ('t3-old', 't4-late')"
            " AND te.entity_name ILIKE 'acme%' ORDER BY te.trace_id",
        )
        old, late = acme
        assert late["entity_id"] is not None
        assert late["entity_id"] == old["entity_id"]
        assert (late["entity_type"], late["names"]) == ("product", 4)
