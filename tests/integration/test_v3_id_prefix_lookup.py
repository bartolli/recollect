"""Integration: the id-prefix lookups behind short ids, on the store SQL.

A short id is judged within one user's whole store: every status counts,
another user's rows never do, and the prefix has no wildcard semantics.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace, PersonaFact

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)
MINE_ACTIVE = "aaaaaaaa-0000-4000-8000-000000000001"
MINE_FORGOTTEN = "aaaaaaaa-0000-4000-8000-000000000002"
THEIRS = "aaaaaaaa-0000-4000-8000-000000000003"
MINE_OTHER = "bbbbbbbb-0000-4000-8000-000000000004"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]


def _url(db_name: str) -> str:
    return urlunparse(urlparse(DB_URL)._replace(path=f"/{db_name}"))


@pytest.fixture()
async def mem() -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"id_prefix_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_url("postgres"))
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    m = CognitiveMemory(extractor=AsyncMock())
    await m.connect(_url(db_name))
    try:
        yield m
    finally:
        await m.close()
        admin = await asyncpg.connect(_url("postgres"))
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


async def test_trace_prefix_is_user_scoped_and_status_agnostic(
    mem: CognitiveMemory,
) -> None:
    for trace_id, user, status in (
        (MINE_ACTIVE, "u1", "active"),
        (MINE_FORGOTTEN, "u1", "forgotten"),
        (THEIRS, "u2", "active"),
        (MINE_OTHER, "u1", "active"),
    ):
        await mem.storage.traces.store_trace(
            MemoryTrace(id=trace_id, content=trace_id, user_id=user, status=status)
        )

    mine = await mem.traces_by_id_prefix("aaaaaaaa", user_id="u1")
    assert {t.id for t in mine} == {MINE_ACTIVE, MINE_FORGOTTEN}

    theirs = await mem.traces_by_id_prefix("aaaaaaaa", user_id="u2")
    assert [t.id for t in theirs] == [THEIRS]

    # A full id is still checked against the caller's user.
    assert await mem.traces_by_id_prefix(THEIRS, user_id="u1") == []
    # `_` and `%` are id characters here, never wildcards.
    assert await mem.traces_by_id_prefix("aaaaaaa_", user_id="u1") == []
    assert await mem.traces_by_id_prefix("%", user_id="u1") == []
    assert len(await mem.traces_by_id_prefix("aaaaaaaa", user_id="u1", limit=1)) == 1


async def test_fact_prefix_is_user_scoped_and_status_agnostic(
    mem: CognitiveMemory,
) -> None:
    for fact_id, user, status in (
        (MINE_ACTIVE, "u1", "pinned"),
        (MINE_FORGOTTEN, "u1", "archived"),
        (THEIRS, "u2", "pinned"),
    ):
        await mem.storage.facts.store_persona_fact(
            PersonaFact(
                id=fact_id,
                subject="Nadia",
                predicate="prefers",
                object=fact_id,
                content=f"Nadia prefers {fact_id}",
                status=status,
                user_id=user,
            )
        )

    mine = await mem.facts_by_id_prefix("aaaaaaaa", user_id="u1")
    assert {f.id for f in mine} == {MINE_ACTIVE, MINE_FORGOTTEN}
    assert await mem.facts_by_id_prefix(THEIRS, user_id="u1") == []
    assert await mem.facts_by_id_prefix("aaaaaaa_", user_id="u1") == []
