"""Tests for the recall skip-reflect safety net (story-2, B1).

The unreflected recall path surfaces pinned + recall-safety-bypass
({health,dietary}) facts as IMPORTANT CONTEXT so safety-critical context is
never silently dropped when an agent skips reflect. Deduped against
gate-surfaced persona facts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.exceptions import StorageError
from recollect.models import MemoryTrace, PersonaFact, Thought
from recollect_mcp.server import AppContext, recall


def _ctx(memory: AsyncMock, user_id: str = "alex") -> MagicMock:
    fake = MagicMock()
    fake.request_context.lifespan_context = AppContext(
        memory=memory, worker=MagicMock(), user_id=user_id
    )
    return fake


@pytest.fixture
def memory() -> AsyncMock:
    m = AsyncMock()
    m.think_about.return_value = [
        Thought(
            trace=MemoryTrace(content="weekend plans"),
            relevance=0.7,
            token_count=8,
            reconstruction="weekend plans",
        )
    ]
    m.facts.return_value = []
    return m


async def test_unprimed_recall_surfaces_pinned_and_safety_facts(
    memory: AsyncMock,
) -> None:
    """Skip-reflect recall surfaces pinned + health/dietary; non-safety excluded."""
    memory.facts.return_value = [
        PersonaFact(
            subject="Angel",
            predicate="is_allergic_to",
            object="penicillin",
            content="Angel is allergic to penicillin",
            status="promoted",
            category="health",
        ),
        PersonaFact(
            subject="Angel",
            predicate="prefers",
            object="window seat",
            content="Angel prefers a window seat",
            status="pinned",
            category="preference",
        ),
        PersonaFact(
            subject="Angel",
            predicate="implements_rule",
            object="hypothesis independence",
            content="Angel implements the hypothesis-independence rule",
            status="promoted",
            category="constraint",
        ),
    ]
    result = await recall("any weekend ideas?", _ctx(memory))
    assert "IMPORTANT CONTEXT:" in result
    assert "penicillin" in result  # health surfaces below floor
    assert "window seat" in result  # pinned surfaces below floor
    assert "hypothesis independence" not in result  # promoted non-safety excluded


async def test_safety_net_dedups_gate_surfaced_fact(memory: AsyncMock) -> None:
    """A pinned fact the gate already surfaced appears exactly once, not twice."""
    memory.facts.return_value = [
        PersonaFact(
            subject="Angel",
            predicate="is_allergic_to",
            object="penicillin",
            content="Angel is allergic to penicillin",
            status="pinned",
            category="health",
        )
    ]
    memory.think_about.return_value = [
        Thought(
            trace=MemoryTrace(
                content="[IMPORTANT CONTEXT] Angel is_allergic_to penicillin",
                pattern={"persona_fact": True, "category": "health"},
            ),
            relevance=0.95,
            token_count=12,
            reconstruction=(
                "[IMPORTANT CONTEXT] Angel is_allergic_to penicillin"
                " -- Angel is allergic to penicillin"
            ),
        )
    ]
    result = await recall("any drug allergies?", _ctx(memory))
    assert result.count("Angel is_allergic_to penicillin") == 1


async def test_failed_first_recall_rearms_safety_net(memory: AsyncMock) -> None:
    """StorageError on the first recall re-arms the one-shot net."""
    memory.facts.return_value = [
        PersonaFact(
            subject="Angel",
            predicate="prefers",
            object="window seat",
            content="Angel prefers a window seat",
            status="pinned",
            category="preference",
        )
    ]
    ctx = _ctx(memory)
    memory.think_about.side_effect = StorageError("db down")
    result = await recall("first attempt", ctx)
    assert "Recall failed" in result
    assert ctx.request_context.lifespan_context.primed is False

    memory.think_about.side_effect = None
    result = await recall("second attempt", ctx)
    assert "IMPORTANT CONTEXT:" in result
    assert "window seat" in result  # net re-armed, delivered on retry


async def test_successful_first_recall_stays_one_shot(memory: AsyncMock) -> None:
    """The net fires on the first delivered recall only."""
    memory.facts.return_value = [
        PersonaFact(
            subject="Angel",
            predicate="prefers",
            object="window seat",
            content="Angel prefers a window seat",
            status="pinned",
            category="preference",
        )
    ]
    ctx = _ctx(memory)
    first = await recall("one", ctx)
    second = await recall("two", ctx)
    assert "window seat" in first
    assert "window seat" not in second
