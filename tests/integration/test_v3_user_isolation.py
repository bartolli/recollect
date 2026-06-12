"""Integration: user isolation across retrieval channels (two-user DB).

Seeds a deliberately contaminated topology -- cross-user association,
shared entity name, cross-user token stamp -- and asserts each channel
honors user_id when supplied while preserving the NULL-kwarg-means-no-
filter contract when not.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, NamedTuple
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.core import CognitiveMemory
from recollect.llm.types import TokenAssessment
from recollect.models import (
    Association,
    MemoryTrace,
    PersonaFact,
    RecallToken,
    TraceEntity,
)

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


class TwoUserDB(NamedTuple):
    reader: CognitiveMemory
    writer: CognitiveMemory
    dsn: str
    a1: MemoryTrace
    a2: MemoryTrace
    a3: MemoryTrace
    b1: MemoryTrace


@pytest.fixture()
async def two_user() -> AsyncGenerator[TwoUserDB, None]:
    db_name = f"user_iso_{uuid.uuid4().hex[:10]}"
    dsn = _scratch_url(db_name)
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    writer = CognitiveMemory()
    await writer.connect(dsn)
    a1 = await writer.experience("Alex is allergic to peanuts", user_id="user-a")
    a2 = await writer.experience(
        "Planning a birthday dinner for Alex", user_id="user-a"
    )
    a3 = await writer.experience(
        "Alex carries an epipen for the severe peanut allergy", user_id="user-a"
    )
    b1 = await writer.experience("Alex is allergic to shellfish", user_id="user-b")
    storage = writer.storage
    await storage.associations.store_association(
        Association(
            source_trace_id=a1.id,
            target_trace_id=a2.id,
            forward_strength=0.9,
            backward_strength=0.9,
        )
    )
    await storage.associations.store_association(
        Association(
            source_trace_id=a1.id,
            target_trace_id=b1.id,
            forward_strength=0.9,
            backward_strength=0.9,
        )
    )
    for trace in (a1, b1):
        await storage.entities.store_trace_entities(
            trace.id,
            [
                TraceEntity(
                    entity_name="Alex", entity_type="person", trace_id=trace.id
                )
            ],
        )
    token_id = await storage.recall_tokens.create_token(
        RecallToken(label="Alex (peer) | allergy context | cross-user bridge")
    )
    await storage.recall_tokens.stamp_traces(token_id, [a1.id, b1.id])
    # Reader has an empty working-memory buffer: storage channels only.
    reader = CognitiveMemory()
    await reader.connect(dsn)
    try:
        yield TwoUserDB(
            reader=reader, writer=writer, dsn=dsn, a1=a1, a2=a2, a3=a3, b1=b1
        )
    finally:
        await reader.close()
        await writer.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


class TestSpreadActivationIsolation:
    async def test_user_predicate_excludes_cross_user_traces(
        self, two_user: TwoUserDB,
    ) -> None:
        spreads = await two_user.reader.storage.vectors.spread_activation(
            two_user.a1.id, user_id="user-a"
        )
        ids = {t.id for t, _ in spreads}
        assert two_user.b1.id not in ids
        assert two_user.a2.id in ids

    async def test_no_user_id_spreads_across_users(
        self, two_user: TwoUserDB,
    ) -> None:
        # NULL-kwarg contract: single-user SDK behavior unchanged.
        spreads = await two_user.reader.storage.vectors.spread_activation(
            two_user.a1.id
        )
        ids = {t.id for t, _ in spreads}
        assert two_user.b1.id in ids


class TestEntityMatchIsolation:
    async def test_user_predicate_excludes_cross_user_traces(
        self, two_user: TwoUserDB,
    ) -> None:
        matches = await two_user.reader.storage.entities.match_entities(
            ["Alex"], user_id="user-a"
        )
        ids = {tid for tid, _ in matches}
        assert two_user.a1.id in ids
        assert two_user.b1.id not in ids


class TestTokenHopIsolation:
    async def test_user_predicate_excludes_cross_user_traces(
        self, two_user: TwoUserDB,
    ) -> None:
        # a1 and b1 share a token; the hop from a1 must not surface b1.
        rows = await two_user.reader.storage.recall_tokens.get_activated_trace_ids(
            [two_user.a1.id], user_id="user-a"
        )
        ids = {trace_id for trace_id, _, _, _, _, _ in rows}
        assert two_user.b1.id not in ids

    async def test_no_user_id_hops_across_users(
        self, two_user: TwoUserDB,
    ) -> None:
        rows = await two_user.reader.storage.recall_tokens.get_activated_trace_ids(
            [two_user.a1.id]
        )
        ids = {trace_id for trace_id, _, _, _, _, _ in rows}
        assert two_user.b1.id in ids


class TestFindGroupsIsolation:
    async def test_user_predicate_excludes_cross_user_stamps(
        self, two_user: TwoUserDB,
    ) -> None:
        groups = await two_user.reader.storage.recall_tokens.find_groups_for_traces(
            [two_user.a1.id, two_user.b1.id], user_id="user-a"
        )
        assert groups
        for group in groups:
            stamped = group["stamped_trace_ids"]
            assert isinstance(stamped, list)
            assert two_user.b1.id not in stamped

    async def test_no_user_id_returns_all_stamps(
        self, two_user: TwoUserDB,
    ) -> None:
        groups = await two_user.reader.storage.recall_tokens.find_groups_for_traces(
            [two_user.a1.id, two_user.b1.id]
        )
        all_stamped: set[str] = set()
        for group in groups:
            stamped = group["stamped_trace_ids"]
            assert isinstance(stamped, list)
            all_stamped.update(stamped)
        assert {two_user.a1.id, two_user.b1.id} <= all_stamped


class TestThinkAboutIsolation:
    async def test_user_scoped_recall_excludes_cross_user_traces(
        self, two_user: TwoUserDB,
    ) -> None:
        # Every channel is contaminated: association, entity, token.
        thoughts = await two_user.reader.think_about(
            "What is Alex allergic to?", user_id="user-a"
        )
        assert thoughts
        for thought in thoughts:
            assert thought.trace.user_id == "user-a"

    async def test_no_user_id_recall_unchanged(
        self, two_user: TwoUserDB,
    ) -> None:
        # Scenario: single-user SDK -- user_id=None means no filter.
        thoughts = await two_user.reader.think_about(
            "What is Alex allergic to?"
        )
        users = {t.trace.user_id for t in thoughts}
        assert "user-b" in users


class TestWriteTimeAssessmentScoping:
    async def test_related_traces_never_cross_users(
        self, two_user: TwoUserDB,
    ) -> None:
        # b1 is cosine-close to a1; ungated it enters the related set and
        # situational groups span users. LLM faked at the boundary.
        extractor = MagicMock()
        extractor._provider.complete_structured = AsyncMock(
            return_value=TokenAssessment(action="none")
        )
        mem = CognitiveMemory(extractor=extractor)
        await mem.connect(two_user.dsn)
        try:
            result = await mem.assess_situational(two_user.a1)
        finally:
            await mem.close()
        assert result is not None
        assert result.related_trace_ids
        assert two_user.b1.id not in result.related_trace_ids


async def _seed_fact(
    mem: CognitiveMemory, *, subject: str, content: str, user_id: str
) -> None:
    await mem.storage.facts.store_persona_fact(
        PersonaFact(
            subject=subject,
            predicate="noted",
            object=content,
            content=content,
            status="promoted",
            user_id=user_id,
        )
    )


class TestFactsIsolation:
    async def test_user_predicate_excludes_cross_user_facts(
        self, two_user: TwoUserDB,
    ) -> None:
        await _seed_fact(
            two_user.reader,
            subject="Alex",
            content="Alex is allergic to peanuts",
            user_id="user-a",
        )
        await _seed_fact(
            two_user.reader,
            subject="Brianna",
            content="Brianna prefers oat milk",
            user_id="user-b",
        )
        facts = await two_user.reader.facts(user_id="user-a")
        assert facts
        for fact in facts:
            assert fact.user_id == "user-a"


class TestMcpReadSurfaceParity:
    async def test_reflect_parity_with_recall(
        self, two_user: TwoUserDB,
    ) -> None:
        # Scenario: reflect and recall must agree -- the exact asymmetry
        # behind the persona_facts user_id drift postmortem.
        from recollect_mcp.server import AppContext, recall, reflect

        await _seed_fact(
            two_user.reader,
            subject="Alex",
            content="Alex is allergic to peanuts",
            user_id="user-a",
        )
        await _seed_fact(
            two_user.reader,
            subject="Brianna",
            content="Brianna prefers oat milk",
            user_id="user-b",
        )
        app = AppContext(
            memory=two_user.reader, worker=MagicMock(), user_id="user-a"
        )
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app

        reflected = await reflect(ctx)
        assert "Alex" in reflected
        assert "Brianna" not in reflected

        recalled = await recall("What is Alex allergic to?", ctx)
        assert "peanuts" in recalled
        assert "shellfish" not in recalled
        assert "Brianna" not in recalled


class TestWorkingMemoryIsolation:
    async def test_user_scoped_recall_filters_buffer(
        self, two_user: TwoUserDB,
    ) -> None:
        # The writer's buffer holds both users' traces; the buffer is the
        # only channel left that can leak b1 after the storage predicates.
        thoughts = await two_user.writer.think_about(
            "What is Alex allergic to?", user_id="user-a"
        )
        assert thoughts
        for thought in thoughts:
            assert thought.trace.user_id == "user-a"

    async def test_no_user_id_buffer_unchanged(
        self, two_user: TwoUserDB,
    ) -> None:
        thoughts = await two_user.writer.think_about(
            "What is Alex allergic to?"
        )
        users = {t.trace.user_id for t in thoughts}
        assert "user-b" in users
