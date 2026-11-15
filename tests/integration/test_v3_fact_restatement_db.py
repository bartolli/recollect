"""Integration: a restatement from another live trace is adopted on a real store."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Literal
from unittest.mock import AsyncMock
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import PersonaFact

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)
USER = "u1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]


def _url(db_name: str) -> str:
    return urlunparse(urlparse(DB_URL)._replace(path=f"/{db_name}"))


def _extraction(
    context: str,
    tags: list[str],
    *,
    fact_type: Literal["semantic", "episodic"] = "semantic",
) -> ExtractionResult:
    relation = Relation(
        source="user",
        relation="prefers",
        target="green tea",
        confidence=0.9,
        category="preference",
        context=context,
        context_tags=tags,
    )
    return ExtractionResult(
        fact_type=fact_type,
        concepts=["green tea"],
        relations=[relation],
        significance=0.6,
    )


@pytest.fixture()
def extractor() -> AsyncMock:
    x = AsyncMock()
    x.extract = AsyncMock(return_value=ExtractionResult())
    return x


@pytest.fixture()
async def mem(extractor: AsyncMock) -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"restatement_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_url("postgres"))
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    cfg = MemoryConfig()
    cfg._set("recall_tokens.enabled", False)
    m = CognitiveMemory(extractor=extractor, config=cfg)
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


async def _facts(mem: CognitiveMemory) -> list[PersonaFact]:
    return await mem.storage.facts.get_persona_facts(user_id=USER, limit=None)


async def _fact_concepts(mem: CognitiveMemory, fact_id: str) -> object:
    pool = await mem.storage.pool.get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT array_agg(concept) FROM concept_embeddings "
            "WHERE owner_type = 'fact' AND owner_id = $1",
            fact_id,
        )


async def _candidate_from_a(mem: CognitiveMemory, extractor: AsyncMock) -> PersonaFact:
    extractor.extract.return_value = _extraction(
        "The user prefers green tea.", ["old tag"]
    )
    await mem.experience("I prefer green tea", user_id=USER)
    (row,) = await _facts(mem)
    assert (row.status, row.mention_count) == ("candidate", 1)
    return row


async def test_extraction_restatement_is_adopted_and_promoted(
    mem: CognitiveMemory, extractor: AsyncMock
) -> None:
    row = await _candidate_from_a(mem, extractor)
    extractor.extract.return_value = _extraction(
        "The user prefers green tea, without sugar.", ["no sugar"]
    )
    c = await mem.experience("I prefer green tea, without sugar", user_id=USER)

    (adopted,) = await _facts(mem)
    assert adopted.id == row.id
    assert adopted.content == "The user prefers green tea, without sugar."
    assert adopted.source_trace_id == c.id
    assert adopted.embedding is not None
    assert adopted.embedding != row.embedding
    assert (adopted.mention_count, adopted.status) == (2, "promoted")
    assert await _fact_concepts(mem, row.id) == ["no sugar"]


async def test_pin_restatement_is_adopted_and_pinned(
    mem: CognitiveMemory, extractor: AsyncMock
) -> None:
    row = await _candidate_from_a(mem, extractor)
    extractor.extract.return_value = _extraction(
        "The user prefers green tea, without sugar.",
        ["no sugar"],
        fact_type="episodic",
    )
    c = await mem.experience("I prefer green tea, without sugar", user_id=USER)
    assert [f.id for f in await _facts(mem)] == [row.id]

    (pinned,) = await mem.pin(c.id)

    assert (pinned.id, pinned.status) == (row.id, "pinned")
    assert pinned.content == "The user prefers green tea, without sugar."
    (stored,) = await _facts(mem)
    assert (stored.id, stored.status, stored.mention_count) == (row.id, "pinned", 1)
    assert stored.content == "The user prefers green tea, without sugar."
    assert stored.source_trace_id == c.id
    assert stored.embedding is not None
    assert stored.embedding != row.embedding
    assert await _fact_concepts(mem, row.id) == ["no sugar"]
