"""Integration: ranking guards -- score clamp, honest trace guarantee.

Mechanical halves of the ranking-coherence story (probe-gated blend
changes live elsewhere): an anti-correlated candidate must degrade
instead of crashing recall, and min_trace_slots must reserve actual
memory traces, not persona-fact thoughts.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.embeddings import FastEmbedProvider
from recollect.models import MemoryTrace, PersonaFact, RecallToken

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]

QUERY = "What did Alex say about the launch?"


def _scratch_url(db_name: str) -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def _admin_url() -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path="/postgres"))


@pytest.fixture()
async def mem() -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"rank_grd_{uuid.uuid4().hex[:10]}"
    dsn = _scratch_url(db_name)
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    cfg = MemoryConfig()
    cfg._set("persona.max_facts_per_query", 10)
    # relevance strategy: facts are never pinned-flagged, so they compete
    # for the unpinned pool's reserved trace slots -- the defect surface.
    cfg._set("persona.ranking_strategy", "relevance")
    m = CognitiveMemory(config=cfg)
    await m.connect(dsn)
    try:
        yield m
    finally:
        await m.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


class TestNegativeCosineClamp:
    async def test_anti_correlated_candidate_degrades_not_crashes(
        self, mem: CognitiveMemory,
    ) -> None:
        # Merge floors search/entity/WM scores at zero, but
        # _fetch_token_traces assigns raw cosine directly: a token-bridged
        # trace too weak for search_semantic (strength < selection_threshold)
        # enters the pool with negative score and fails Thought(ge=0).
        provider = FastEmbedProvider()
        q = await provider.generate_embedding(QUERY, task="search_query")
        anchor = MemoryTrace(
            content="Alex presented the launch plan",
            embedding=q,
            user_id="u1",
        )
        anti = MemoryTrace(
            content="anti-correlated faded trace",
            embedding=[-x for x in q],
            strength=0.05,
            user_id="u1",
        )
        await mem.storage.traces.store_trace(anchor)
        await mem.storage.traces.store_trace(anti)
        token_id = await mem.storage.recall_tokens.create_token(
            RecallToken(label="launch (project) | planning thread | bridge")
        )
        await mem.storage.recall_tokens.stamp_traces(
            token_id, [anchor.id, anti.id]
        )

        thoughts = await mem.think_about(QUERY, user_id="u1")
        for thought in thoughts:
            assert thought.relevance >= 0.0


class TestTraceGuarantee:
    async def test_trace_slots_hold_traces(self, mem: CognitiveMemory) -> None:
        # Promoted-fact thoughts share the unpinned pool; with enough
        # outranking facts the reserved min_trace_slots all go to facts.
        provider = FastEmbedProvider()
        q = await provider.generate_embedding(QUERY, task="search_query")
        for i in range(10):
            await mem.storage.facts.store_persona_fact(
                PersonaFact(
                    subject=f"subject-{i}",
                    predicate="noted",
                    object=f"launch detail {i}",
                    content=f"launch detail {i}",
                    status="promoted",
                    embedding=q,
                    user_id="u1",
                )
            )
        for content in (
            "Alex presented the launch plan",
            "The launch date moved to Thursday",
        ):
            await mem.storage.traces.store_trace(
                MemoryTrace(
                    content=content,
                    embedding=await provider.generate_embedding(
                        content, task="search_document"
                    ),
                    user_id="u1",
                )
            )

        thoughts = await mem.think_about(QUERY, user_id="u1")
        trace_thoughts = [
            t for t in thoughts if not t.trace.pattern.get("persona_fact")
        ]
        assert len(trace_thoughts) >= 2
