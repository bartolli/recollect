"""Integration tests: full cognitive memory lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncGenerator

import pytest
from recollect.core import CognitiveMemory
from recollect.models import (
    ConsolidationResult,
    HealthStatus,
    MemoryStats,
    MemoryTrace,
    Thought,
)

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]


@pytest.fixture()
async def mem() -> AsyncGenerator[CognitiveMemory, None]:
    """Create a CognitiveMemory connected to a real database."""
    m = CognitiveMemory()
    await m.connect(DB_URL)
    yield m
    # Cleanup: remove traces created during the test
    try:
        traces = await m.timeline(limit=100)
        for t in traces:
            with contextlib.suppress(Exception):
                await m.forget(t.id)
    finally:
        await m.close()


class TestFullLifecycle:
    """End-to-end tests against a live PostgreSQL database."""

    async def test_experience_stores_and_retrieves(self, mem: CognitiveMemory) -> None:
        trace = await mem.experience("PostgreSQL is a relational database")
        assert isinstance(trace, MemoryTrace)
        assert trace.content == "PostgreSQL is a relational database"
        assert 0.0 <= trace.strength <= 1.0

        thoughts = await mem.think_about("relational database")
        assert len(thoughts) >= 1
        assert isinstance(thoughts[0], Thought)
        contents = [t.trace.content for t in thoughts]
        assert "PostgreSQL is a relational database" in contents

    async def test_multiple_experiences_timeline(self, mem: CognitiveMemory) -> None:
        contents = [
            "First experience for timeline",
            "Second experience for timeline",
            "Third experience for timeline",
        ]
        for c in contents:
            await mem.experience(c)
            await asyncio.sleep(0.05)  # ensure distinct timestamps

        recent = await mem.timeline(limit=3)
        assert len(recent) >= 3
        # Timeline is newest-first
        recent_contents = [t.content for t in recent[:3]]
        assert recent_contents[0] == "Third experience for timeline"
        assert recent_contents[2] == "First experience for timeline"

    async def test_forget_removes_trace(self, mem: CognitiveMemory) -> None:
        trace = await mem.experience("This memory will be forgotten")
        deleted = await mem.forget(trace.id)
        assert deleted is True

        # Verify trace is gone from storage (timeline queries storage)
        recent = await mem.timeline(limit=50)
        stored_ids = {t.id for t in recent}
        assert trace.id not in stored_ids

    async def test_reinforce_increases_strength(self, mem: CognitiveMemory) -> None:
        trace = await mem.experience("Reinforce this memory")
        original_strength = trace.strength

        reinforced = await mem.reinforce(trace.id, factor=1.5)
        assert isinstance(reinforced, MemoryTrace)
        assert reinforced.strength > original_strength
        assert reinforced.strength <= 1.0

    async def test_consolidate_lifecycle(self, mem: CognitiveMemory) -> None:
        for i in range(3):
            await mem.experience(f"Consolidation test trace {i}")

        result = await mem.consolidate()
        assert isinstance(result, ConsolidationResult)
        total = result.consolidated + result.forgotten + result.still_pending
        assert total >= 3

    async def test_stats_and_health(self, mem: CognitiveMemory) -> None:
        s = mem.stats()
        assert isinstance(s, MemoryStats)
        assert s.connected is True
        assert s.working_memory_capacity >= 5

        h = mem.health()
        assert isinstance(h, HealthStatus)
        assert h.status == "ok"

    async def test_experience_with_context(self, mem: CognitiveMemory) -> None:
        ctx = {"source": "integration-test", "priority": "high"}
        trace = await mem.experience("Context-aware experience", context=ctx)
        assert trace.context == ctx
        assert trace.context["source"] == "integration-test"
