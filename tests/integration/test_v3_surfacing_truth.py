"""Integration: surfacing reads are status-filtered in SQL, not window-cut.

A pinned safety fact older than 50 newer candidate rows must reach every
surfacing path -- the recency window on the fact read hid it from all of them.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import MemoryTrace, PersonaFact
from recollect_mcp.server import AppContext, _generate_primer, recall, reflect

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


@pytest.fixture()
def extractor() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
async def mem(extractor: AsyncMock) -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"surfacing_{uuid.uuid4().hex[:10]}"
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


async def _seed_buried_pin(mem: CognitiveMemory) -> PersonaFact:
    """One pinned health fact, then 55 newer candidates that bury it."""
    pinned = PersonaFact(
        subject="Alex",
        predicate="is_allergic_to",
        object="penicillin",
        content="Alex is allergic to penicillin",
        category="health",
        status="pinned",
        user_id=USER,
    )
    await _bury(mem, pinned)
    return pinned


async def _bury(mem: CognitiveMemory, oldest: PersonaFact) -> None:
    await mem.storage.facts.store_persona_fact(oldest)
    for i in range(55):
        await mem.storage.facts.store_persona_fact(
            PersonaFact(
                subject="Alex",
                predicate="is_interested_in",
                object=f"topic-{i}",
                content=f"Alex is interested in topic-{i}",
                status="candidate",
                user_id=USER,
            )
        )


def _ctx(mem: CognitiveMemory) -> MagicMock:
    fake = MagicMock()
    fake.request_context.lifespan_context = AppContext(
        memory=mem, worker=MagicMock(), user_id=USER
    )
    fake.headers = None
    return fake


async def test_status_filtered_read_ignores_the_recency_window(
    mem: CognitiveMemory,
) -> None:
    pinned = await _seed_buried_pin(mem)
    surfacing = await mem.facts(
        user_id=USER, statuses=("promoted", "pinned"), limit=None
    )
    assert [f.id for f in surfacing] == [pinned.id]
    assert len(await mem.facts(user_id=USER)) == 50


async def test_buried_pin_reaches_reflect_primer_and_unprimed_recall(
    mem: CognitiveMemory,
) -> None:
    await _seed_buried_pin(mem)
    ctx = _ctx(mem)
    assert "penicillin" in await _generate_primer(ctx.request_context.lifespan_context)
    assert "penicillin" in await recall("what should I cook tonight", ctx)
    assert "penicillin" in await reflect(ctx)
    # Primed now: the one-shot safety net is spent, so this recall can only
    # reach the fact through the gate's promoted+pinned supplement.
    assert "penicillin" in await recall("what should I cook tonight", ctx)


async def test_buried_pin_reaches_write_time_safety_surfacing(
    mem: CognitiveMemory,
) -> None:
    pinned = await _seed_buried_pin(mem)
    trace = MemoryTrace(content="dinner plans", pattern={"domains": ["food"]})
    surfaced = await mem.surface_relevant_facts(trace, user_id=USER)
    assert pinned.id in [f.id for f in surfaced]


async def test_restating_the_oldest_of_many_facts_increments_not_duplicates(
    mem: CognitiveMemory, extractor: AsyncMock
) -> None:
    oldest = PersonaFact(
        subject="Alex",
        predicate="works_at",
        object="Acme",
        content="Alex works at Acme",
        category="identity",
        status="candidate",
        user_id=USER,
    )
    await _bury(mem, oldest)
    extractor.extract = AsyncMock(
        return_value=ExtractionResult(
            concepts=["employment"],
            relations=[
                Relation(
                    source="Alex",
                    relation="works_at",
                    target="Acme",
                    confidence=0.9,
                    category="identity",
                    context="Alex works at Acme",
                )
            ],
            significance=0.5,
            fact_type="semantic",
        )
    )
    await mem.experience("Alex works at Acme", user_id=USER)

    acme = [
        f
        for f in await mem.facts("Alex", user_id=USER, limit=None)
        if f.object == "Acme"
    ]
    assert [f.id for f in acme] == [oldest.id]
    assert acme[0].mention_count == 2
    assert acme[0].status == "promoted"
