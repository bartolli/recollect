"""Integration: experience() persists the extractor's relation context."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import ExtractionResult, Relation

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [pytest.mark.slow, pytest.mark.asyncio]

HEDGED = Relation(
    source="user's low-activity hive",
    relation="requires",
    target="colony merge with user's other hive",
    context=(
        "The hive with almost no entrance activity is reported (not lab-confirmed)"
        " to have probably lost its queen, and merging it with the other hive is"
        " being considered as a remedy; confirm queen status before acting."
    ),
)
BARE = Relation(source="user", relation="practices", target="beekeeping")


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
    db_name = f"relctx_{uuid.uuid4().hex[:10]}"
    await _admin(f'CREATE DATABASE "{db_name}"')
    try:
        yield _url(db_name)
    finally:
        await _admin(f'DROP DATABASE "{db_name}" WITH (FORCE)')


@pytest.fixture()
async def mem(scratch_db: str) -> AsyncGenerator[CognitiveMemory, None]:
    extractor = AsyncMock()
    extractor.extract = AsyncMock(
        return_value=ExtractionResult(relations=[HEDGED, BARE])
    )
    cfg = MemoryConfig()
    cfg._set("recall_tokens.enabled", False)
    m = CognitiveMemory(extractor=extractor, config=cfg)
    await m.connect(scratch_db)
    try:
        yield m
    finally:
        await m.close()


async def test_rows_hold_extractor_context_or_the_triple(
    mem: CognitiveMemory,
) -> None:
    trace = await mem.experience("hive notes", user_id="u-a")
    pool = await mem.storage.pool.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT relation, context FROM entity_relations"
            " WHERE source_trace_id = $1 ORDER BY relation",
            trace.id,
        )
    assert [tuple(r) for r in rows] == [
        ("practices", "user practices beekeeping"),
        ("requires", HEDGED.context),
    ]
