"""Integration: persona-fact lifecycle -- supersede, unpin, pin.

Lifecycle semantics on scratch DBs with the LLM faked at the boundary:
corrections must stay surfaced, retractions must leave surfacing, pins
must promote what extraction actually produced.
"""

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
from recollect.models import FactStatus, PersonaFact

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

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
async def mem() -> AsyncGenerator[tuple[CognitiveMemory, AsyncMock], None]:
    db_name = f"fact_lc_{uuid.uuid4().hex[:10]}"
    dsn = _scratch_url(db_name)
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    extractor = AsyncMock()
    cfg = MemoryConfig()
    cfg._set("recall_tokens.enabled", False)
    m = CognitiveMemory(extractor=extractor, config=cfg)
    await m.connect(dsn)
    try:
        yield m, extractor
    finally:
        await m.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


async def _seed_fact(
    mem: CognitiveMemory,
    *,
    obj: str,
    status: FactStatus,
    predicate: str = "lives_in",
) -> PersonaFact:
    fact = PersonaFact(
        subject="user",
        predicate=predicate,
        object=obj,
        content=f"user {predicate} {obj}",
        status=status,
        user_id="u1",
    )
    await mem.storage.facts.store_persona_fact(fact)
    return fact


def _wire_extraction(extractor: AsyncMock, relation: Relation) -> None:
    extractor.extract = AsyncMock(
        return_value=ExtractionResult(
            concepts=["relocation"],
            relations=[relation],
            significance=0.5,
            fact_type="semantic",
        )
    )


_BERLIN = Relation(
    source="user",
    relation="lives_in",  # enum-canonical; persists raw (alias map deleted)
    target="Berlin",
    confidence=0.9,
    category="identity",
    context="user moved to Berlin",
)


class TestSupersedeInheritsStatus:
    async def test_correction_stays_surfaced(
        self, mem: tuple[CognitiveMemory, AsyncMock],
    ) -> None:
        # The defect: a promoted fact superseded by a non-fast-track
        # correction left a candidate -- invisible to every recall surface.
        m, extractor = mem
        old = await _seed_fact(m, obj="Lisbon", status="promoted")
        _wire_extraction(extractor, _BERLIN)
        await m.experience("I moved to Berlin", user_id="u1")

        facts = {f.object: f for f in await m.facts(user_id="u1")}
        assert "Lisbon" not in facts  # superseded rows leave the active set
        berlin = facts["Berlin"]
        assert berlin.id != old.id
        assert berlin.status == "promoted"

    async def test_pinned_stays_pinned(
        self, mem: tuple[CognitiveMemory, AsyncMock],
    ) -> None:
        m, extractor = mem
        await _seed_fact(m, obj="Lisbon", status="pinned")
        _wire_extraction(extractor, _BERLIN)
        await m.experience("I moved to Berlin", user_id="u1")

        facts = {f.object: f for f in await m.facts(user_id="u1")}
        assert facts["Berlin"].status == "pinned"


class TestUnpinArchives:
    async def test_unpin_removes_from_surfacing(
        self, mem: tuple[CognitiveMemory, AsyncMock],
    ) -> None:
        from unittest.mock import MagicMock

        from recollect_mcp.server import AppContext, reflect

        m, _ = mem
        fact = await _seed_fact(
            m, obj="shellfish", status="pinned", predicate="is_allergic_to"
        )
        assert await m.unpin(fact.id) is True

        facts = {f.id: f for f in await m.facts(user_id="u1")}
        assert facts[fact.id].status == "archived"

        app = AppContext(memory=m, worker=MagicMock(), user_id="u1")
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app
        reflected = await reflect(ctx)
        assert "shellfish" not in reflected

    async def test_unpin_unknown_fact_reports_not_found(
        self, mem: tuple[CognitiveMemory, AsyncMock],
    ) -> None:
        m, _ = mem
        assert await m.unpin("no-such-fact") is False


_CYCLING = Relation(
    source="user",
    relation="prefers",
    target="cycling",
    confidence=0.5,  # below persona.confidence_threshold: extraction skips it
    category="preference",
    context="user prefers cycling on weekends",
)


class TestPinPromotesRelations:
    async def test_pin_promotes_extracted_relations(
        self, mem: tuple[CognitiveMemory, AsyncMock],
    ) -> None:
        # The relation persists in trace.pattern even when the
        # confidence gate skipped it at write time -- pin is the user
        # saying "this matters", so it promotes what was extracted.
        m, extractor = mem
        _wire_extraction(extractor, _CYCLING)
        trace = await m.experience("I love cycling on weekends", user_id="u1")
        assert await m.facts(user_id="u1") == []

        pinned = await m.pin(trace.id)
        assert [f.object for f in pinned] == ["cycling"]

        facts = await m.facts(user_id="u1")
        assert len(facts) == 1
        fact = facts[0]
        assert (fact.subject, fact.predicate, fact.object) == (
            "user",
            "prefers",
            "cycling",
        )
        assert fact.category == "preference"
        assert fact.status == "pinned"
        assert fact.embedding is not None

    async def test_pin_empty_extraction_falls_back_to_noted(
        self, mem: tuple[CognitiveMemory, AsyncMock],
    ) -> None:
        m, extractor = mem
        extractor.extract = AsyncMock(
            return_value=ExtractionResult(concepts=["note"], significance=0.5)
        )
        trace = await m.experience("An unstructured remark", user_id="u1")

        pinned = await m.pin(trace.id)
        assert len(pinned) == 1
        fact = pinned[0]
        assert (fact.subject, fact.predicate) == ("user", "noted")
        assert fact.object == "An unstructured remark"
        assert fact.status == "pinned"
