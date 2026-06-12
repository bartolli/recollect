"""Integration: status foundation -- trace status column, FactStatus widening.

Archive-lifecycle substrate on scratch DBs: persona_facts.status is
unconstrained TEXT at the DB tier, so 'archived' writes always succeeded;
the break was the closed FactStatus Literal on read. Trace status is new
column + model field + m005 for existing DBs.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.bootstrap.migration import MigrationRegistry
from recollect.bootstrap.migrations import (
    default_registry,
    m001_initial,
    m002_user_id_backfill,
    m003_embedding_contract,
    m004_fact_orphan_cleanup,
)
from recollect.bootstrap.runner import apply
from recollect.models import MemoryTrace, PersonaFact
from recollect.storage_context import create_storage_context

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    _ScratchFixture = tuple[asyncpg.Pool[asyncpg.Record], str]

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
async def scratch() -> AsyncGenerator[_ScratchFixture, None]:
    db_name = f"status_fnd_{uuid.uuid4().hex[:10]}"
    dsn = _scratch_url(db_name)
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)
    try:
        yield pool, dsn
    finally:
        await pool.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


class TestFactStatusWidening:
    async def test_archived_fact_reads_back(
        self, scratch: _ScratchFixture,
    ) -> None:
        # The DB tier accepts 'archived' today; only the read-side
        # Literal broke. get_persona_facts stays unfiltered (BD-4:
        # widen, do not drop) so archive substrates can read it.
        pool, dsn = scratch
        await apply(pool, default_registry())
        storage = create_storage_context(dsn)
        await storage.initialize()
        try:
            await storage.facts.store_persona_fact(
                PersonaFact(
                    subject="Alex",
                    predicate="noted",
                    object="archived example",
                    content="archived example",
                    status="promoted",
                    user_id="user-a",
                )
            )
            async with pool.acquire() as conn:
                await conn.execute("UPDATE persona_facts SET status = 'archived'")
            facts = await storage.facts.get_persona_facts()
            assert len(facts) == 1
            assert facts[0].status == "archived"
        finally:
            await storage.close()


class TestTraceStatusRoundTrip:
    async def test_status_survives_write_and_read(
        self, scratch: _ScratchFixture,
    ) -> None:
        pool, dsn = scratch
        await apply(pool, default_registry())
        storage = create_storage_context(dsn)
        await storage.initialize()
        try:
            trace = MemoryTrace(content="to be archived", status="archived")
            await storage.traces.store_trace(trace)
            fetched = await storage.traces.get_trace(trace.id)
            assert fetched is not None
            assert fetched.status == "archived"
        finally:
            await storage.close()

    async def test_status_defaults_active(
        self, scratch: _ScratchFixture,
    ) -> None:
        pool, dsn = scratch
        await apply(pool, default_registry())
        storage = create_storage_context(dsn)
        await storage.initialize()
        try:
            trace = MemoryTrace(content="ordinary trace")
            await storage.traces.store_trace(trace)
            fetched = await storage.traces.get_trace(trace.id)
            assert fetched is not None
            assert fetched.status == "active"
        finally:
            await storage.close()


class TestM005TraceStatus:
    async def test_fresh_bootstrap_records_m005_and_index(
        self, scratch: _ScratchFixture,
    ) -> None:
        pool, _ = scratch
        await apply(pool, default_registry())
        async with pool.acquire() as conn:
            applied = [
                r["name"]
                for r in await conn.fetch(
                    "SELECT name FROM applied_migrations ORDER BY name"
                )
            ]
            assert "m005_trace_status" in applied
            index_exists = await conn.fetchval(
                "SELECT COUNT(*) FROM pg_indexes WHERE indexname = "
                "'idx_traces_active_unconsolidated'"
            )
            assert index_exists == 1

    async def test_existing_db_upgrade_idempotent(
        self, scratch: _ScratchFixture,
    ) -> None:
        # Pre-status DB at m004: drop the SCHEMA_SQL-created column to
        # simulate prod, seed a surviving row, then upgrade and re-run.
        pool, _ = scratch
        pre_m005 = MigrationRegistry()
        pre_m005.register(m001_initial)
        pre_m005.register(m002_user_id_backfill)
        pre_m005.register(m003_embedding_contract)
        pre_m005.register(m004_fact_orphan_cleanup)
        await apply(pool, pre_m005)
        async with pool.acquire() as conn:
            await conn.execute("ALTER TABLE memory_traces DROP COLUMN status")
            await conn.execute(
                "INSERT INTO memory_traces (id, content) VALUES ('t1', 'old row')"
            )

        await apply(pool, default_registry())
        await apply(pool, default_registry())  # re-run: no-op

        async with pool.acquire() as conn:
            status = await conn.fetchval(
                "SELECT status FROM memory_traces WHERE id = 't1'"
            )
            assert status == "active"
            m005_count = await conn.fetchval(
                "SELECT COUNT(*) FROM applied_migrations "
                "WHERE name = 'm005_trace_status'"
            )
            assert m005_count == 1
