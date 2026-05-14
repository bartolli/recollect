"""Integration test: m002_user_id_backfill on scratch DBs.

Three cases:
  - NULL rows present + default uid set    -> backfilled + NOT NULL applied
  - NULL rows present + default uid empty  -> BootstrapError (no writes)
  - Empty table                            -> SET NOT NULL applies cleanly
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.bootstrap.migration import MigrationRegistry
from recollect.bootstrap.migrations.m001_initial import m001_initial
from recollect.bootstrap.migrations.m002_user_id_backfill import (
    _M002UserIdBackfill,
)
from recollect.bootstrap.runner import apply
from recollect.exceptions import BootstrapError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    _ScratchPool = asyncpg.Pool[asyncpg.Record]

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
    db_name = f"bootstrap_m002_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(_scratch_url(db_name), min_size=1, max_size=4)
    try:
        yield pool
    finally:
        await pool.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


def _registry_with_m002(default_uid: str) -> MigrationRegistry:
    r = MigrationRegistry()
    r.register(m001_initial)
    r.register(_M002UserIdBackfill(get_default_user_id=lambda: default_uid))
    return r


async def _insert_null_fact(conn: asyncpg.Connection[asyncpg.Record]) -> None:
    await conn.execute(
        """
        INSERT INTO persona_facts (id, subject, predicate, object, content)
        VALUES ($1, 'alice', 'likes', 'sushi', 'alice likes sushi')
        """,
        f"fact-{uuid.uuid4().hex[:8]}",
    )


class TestM002Backfill:
    async def test_backfills_null_user_id_when_default_set(
        self, scratch_pool: _ScratchPool,
    ) -> None:
        # m001 only, then insert NULL rows
        m001_only = MigrationRegistry()
        m001_only.register(m001_initial)
        await apply(scratch_pool, m001_only)
        async with scratch_pool.acquire() as conn:
            await _insert_null_fact(conn)
            await _insert_null_fact(conn)
            await _insert_null_fact(conn)

        result = await apply(scratch_pool, _registry_with_m002("alice"))
        assert result.applied == ["m002_user_id_backfill"]

        async with scratch_pool.acquire() as conn:
            null_count = await conn.fetchval(
                "SELECT COUNT(*) FROM persona_facts WHERE user_id IS NULL"
            )
            assert null_count == 0
            alice_count = await conn.fetchval(
                "SELECT COUNT(*) FROM persona_facts WHERE user_id = 'alice'"
            )
            assert alice_count == 3
            # NOT NULL constraint enforced: insert with NULL user_id must fail
            with pytest.raises(asyncpg.NotNullViolationError):
                await conn.execute(
                    """
                    INSERT INTO persona_facts
                        (id, subject, predicate, object, content, user_id)
                    VALUES ('post', 's', 'p', 'o', 'content', NULL)
                    """
                )

    async def test_raises_when_null_present_and_default_empty(
        self, scratch_pool: _ScratchPool,
    ) -> None:
        m001_only = MigrationRegistry()
        m001_only.register(m001_initial)
        await apply(scratch_pool, m001_only)
        async with scratch_pool.acquire() as conn:
            await _insert_null_fact(conn)

        with pytest.raises(BootstrapError, match="user_id"):
            await apply(scratch_pool, _registry_with_m002(""))

        async with scratch_pool.acquire() as conn:
            null_count = await conn.fetchval(
                "SELECT COUNT(*) FROM persona_facts WHERE user_id IS NULL"
            )
            assert null_count == 1  # untouched

    async def test_empty_table_applies_not_null_cleanly(
        self, scratch_pool: _ScratchPool,
    ) -> None:
        result = await apply(scratch_pool, _registry_with_m002(""))
        assert "m002_user_id_backfill" in result.applied
        async with scratch_pool.acquire() as conn:
            with pytest.raises(asyncpg.NotNullViolationError):
                await conn.execute(
                    """
                    INSERT INTO persona_facts
                        (id, subject, predicate, object, content, user_id)
                    VALUES ('p', 's', 'p', 'o', 'content', NULL)
                    """
                )
