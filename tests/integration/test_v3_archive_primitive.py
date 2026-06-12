"""Integration: archive primitive -- preservation, scan exclusion, erase.

Consolidator-archive flips status and preserves derived rows (the
concept_embeddings orphan leak closes structurally); the consolidation
scan excludes archived rows (batch starvation); erase() hard-deletes
with FK cascades.
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.core import CognitiveMemory
from recollect.datetime_utils import now_utc
from recollect.models import ConceptEmbedding, MemoryTrace, PersonaFact

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
    db_name = f"archive_{uuid.uuid4().hex[:10]}"
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


def _weak_old_trace() -> MemoryTrace:
    # Below threshold after decay AND past grace: the forgotten branch.
    return MemoryTrace(
        content="fading",
        strength=0.05,
        created_at=now_utc() - timedelta(hours=48),
    )


class TestConsolidatorArchive:
    async def test_archives_and_preserves_derived_rows(
        self, mem: CognitiveMemory, scratch_db: str
    ) -> None:
        trace = _weak_old_trace()
        await mem.storage.traces.store_trace(trace)
        await mem.storage.concept_embeddings.store_concept_embeddings(
            [
                ConceptEmbedding(
                    concept="fading thing",
                    owner_type="trace",
                    owner_id=trace.id,
                    embedding=[0.1] * 768,
                )
            ]
        )
        fact = PersonaFact(
            subject="user",
            predicate="noted",
            object="fading thing",
            content="user noted fading thing",
            source_trace_id=trace.id,
            user_id="u1",
        )
        await mem.storage.facts.store_persona_fact(fact)

        result = await mem.consolidate()

        assert result.forgotten == 1
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.status == "archived"
        conn = await asyncpg.connect(scratch_db)
        try:
            orphan_check = await conn.fetchval(
                "SELECT count(*) FROM concept_embeddings "
                "WHERE owner_type = 'trace' AND owner_id = $1",
                trace.id,
            )
        finally:
            await conn.close()
        assert orphan_check == 1
        facts = await mem.storage.facts.get_persona_facts(user_id="u1")
        assert any(f.id == fact.id for f in facts)

    async def test_scan_excludes_archived(self, mem: CognitiveMemory) -> None:
        archived = _weak_old_trace()
        pending = MemoryTrace(content="new", strength=0.3)
        await mem.storage.traces.store_trace(archived)
        await mem.storage.traces.store_trace(pending)
        assert await mem.storage.traces.archive_trace(archived.id) is True

        scanned = await mem.storage.traces.get_unconsolidated_traces()
        ids = {t.id for t in scanned}
        assert pending.id in ids
        assert archived.id not in ids

    async def test_archive_trace_false_on_already_archived(
        self, mem: CognitiveMemory
    ) -> None:
        trace = MemoryTrace(content="once", strength=0.3)
        await mem.storage.traces.store_trace(trace)
        assert await mem.storage.traces.archive_trace(trace.id) is True
        assert await mem.storage.traces.archive_trace(trace.id) is False


class TestForgetGuard:
    async def test_forget_archives_non_safety_fact(
        self, mem: CognitiveMemory
    ) -> None:
        trace = MemoryTrace(content="prefers window seats", strength=0.5)
        await mem.storage.traces.store_trace(trace)
        fact = PersonaFact(
            subject="user",
            predicate="prefers",
            object="window seats",
            category="preference",
            content="user prefers window seats",
            source_trace_id=trace.id,
            status="promoted",
            user_id="u1",
        )
        await mem.storage.facts.store_persona_fact(fact)

        result = await mem.forget(trace.id)

        assert result.archived_fact_ids == [fact.id]
        rows = await mem.storage.facts.get_facts_by_source_trace_id(trace.id)
        assert rows[0].status == "archived"

    async def test_hard_fact_retained_then_forced(
        self, mem: CognitiveMemory
    ) -> None:
        trace = MemoryTrace(content="allergic to shellfish", strength=0.5)
        await mem.storage.traces.store_trace(trace)
        fact = PersonaFact(
            subject="user",
            predicate="is_allergic_to",
            object="shellfish",
            category="health",
            content="user is allergic to shellfish",
            source_trace_id=trace.id,
            status="promoted",
            user_id="u1",
        )
        await mem.storage.facts.store_persona_fact(fact)

        result = await mem.forget(trace.id)
        assert result.archived_fact_ids == []
        assert [f.id for f in result.retained_facts] == [fact.id]
        rows = await mem.storage.facts.get_facts_by_source_trace_id(trace.id)
        assert rows[0].status == "promoted"

        # Forced re-forget: trace already archived -> not-found contract.
        # Suppression of the retained fact goes through a fresh guard
        # pass on a still-active trace, so seed a second trace.
        trace2 = MemoryTrace(content="allergy restated", strength=0.5)
        await mem.storage.traces.store_trace(trace2)
        fact2 = fact.model_copy(
            update={"id": "fact2-hard", "source_trace_id": trace2.id}
        )
        await mem.storage.facts.store_persona_fact(fact2)
        forced = await mem.forget(trace2.id, force=True)
        assert forced.archived_fact_ids == ["fact2-hard"]

    async def test_restated_retraction_creates_fresh_fact(
        self, mem: CognitiveMemory
    ) -> None:
        archived = PersonaFact(
            subject="user",
            predicate="lives_in",
            object="Lisbon",
            category="identity",
            content="user lives in Lisbon",
            status="archived",
            user_id="u1",
        )
        await mem.storage.facts.store_persona_fact(archived)
        restated = PersonaFact(
            subject="user",
            predicate="lives_in",
            object="Lisbon",
            category="identity",
            content="user lives in Lisbon",
            status="candidate",
            user_id="u1",
        )
        stored = await mem._store_or_promote_fact(restated)
        assert stored is not None
        facts = await mem.storage.facts.get_persona_facts("user")
        statuses = sorted(f.status for f in facts)
        assert statuses == ["archived", "candidate"]


class TestErase:
    async def test_erase_hard_deletes(self, mem: CognitiveMemory) -> None:
        trace = MemoryTrace(content="gone", strength=0.3)
        await mem.storage.traces.store_trace(trace)
        assert await mem.erase(trace.id) is True
        assert await mem.storage.traces.get_trace(trace.id) is None

    async def test_forget_then_recallable_until_reactivation_story(
        self, mem: CognitiveMemory
    ) -> None:
        # Locks the accepted interim: retrieval is status-agnostic, so an
        # archived trace still surfaces (it is reactivation substrate).
        vec = await mem._embeddings.generate_embedding("archived but present")
        trace = MemoryTrace(
            content="archived but present", embedding=vec, strength=0.8
        )
        await mem.storage.traces.store_trace(trace)
        await mem.forget(trace.id)
        query = await mem._embeddings.generate_embedding(
            "archived but present", task="search_query"
        )
        results = await mem.storage.vectors.search_semantic(query, limit=10)
        assert trace.id in {t.id for t, _ in results}
