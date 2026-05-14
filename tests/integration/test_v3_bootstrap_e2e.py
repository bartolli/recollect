"""Integration test: bootstrap.apply on a fresh scratch DB.

Verifies the full schema lands (10 tables, 3 HNSW vector indexes, 2
extensions, applied_migrations row count == registry size) and that a
second apply() call is a no-op (idempotency invariant).
"""

from __future__ import annotations

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

    _ScratchPool = tuple[asyncpg.Pool[asyncpg.Record], str]

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
    db_name = f"bootstrap_test_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(_scratch_url(db_name), min_size=1, max_size=4)
    try:
        yield pool, db_name
    finally:
        await pool.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


class TestFreshBootstrap:
    async def test_apply_creates_full_schema(
        self,
        scratch_pool: _ScratchPool,
    ) -> None:
        pool, _ = scratch_pool
        result = await apply(pool, default_registry())

        assert result.applied == default_registry().names()
        assert result.skipped == []
        assert result.schema_ahead == []

        async with pool.acquire() as conn:
            extensions = {
                r["extname"]
                for r in await conn.fetch(
                    "SELECT extname FROM pg_extension "
                    "WHERE extname IN ('vector', 'pg_trgm')"
                )
            }
            assert extensions == {"vector", "pg_trgm"}

            tables = {
                r["tablename"]
                for r in await conn.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
            }
            expected_tables = {
                "sessions", "memory_traces", "associations",
                "trace_entities", "trace_concepts", "persona_facts",
                "entity_relations", "concept_embeddings", "recall_tokens",
                "token_stamps", "applied_migrations",
            }
            assert expected_tables.issubset(tables)

            hnsw_count = await conn.fetchval(
                "SELECT COUNT(*) FROM pg_indexes "
                "WHERE schemaname = 'public' AND indexdef ILIKE '%hnsw%'"
            )
            assert hnsw_count == 3

            applied_names = [
                r["name"]
                for r in await conn.fetch(
                    "SELECT name FROM applied_migrations ORDER BY applied_at"
                )
            ]
            assert applied_names == default_registry().names()

    async def test_second_apply_is_noop(
        self,
        scratch_pool: _ScratchPool,
    ) -> None:
        pool, _ = scratch_pool
        first = await apply(pool, default_registry())
        second = await apply(pool, default_registry())

        assert first.applied == default_registry().names()
        assert second.applied == []
        assert second.skipped == default_registry().names()
        assert second.schema_ahead == []
