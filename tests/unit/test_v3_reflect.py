"""Tests for the reflect MCP tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.models import MemoryTrace, PersonaFact, Session, Thought
from recollect_mcp.server import AppContext, recall, reflect


@pytest.fixture
def mock_memory() -> AsyncMock:
    memory = AsyncMock()
    memory.facts.return_value = [
        PersonaFact(
            subject="Alex",
            predicate="mother_of",
            object="Sarah",
            content="Alex's mother is Sarah who lives in Portland",
            status="pinned",
        )
    ]
    memory.start_session.return_value = Session(id="sess-123", user_id="alex")
    return memory


def _make_ctx(mock_memory: AsyncMock, user_id: str = "alex") -> MagicMock:
    fake_ctx = MagicMock()
    app = AppContext(memory=mock_memory, worker=MagicMock(), user_id=user_id)
    fake_ctx.request_context.lifespan_context = app
    return fake_ctx


@pytest.fixture
def ctx(mock_memory: AsyncMock) -> MagicMock:
    return _make_ctx(mock_memory, user_id="alex")


async def test_reflect_returns_primer_and_facts(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    """Reflect returns both primer (relational graph) and formatted facts."""
    result = await reflect(ctx)

    assert "KNOWN FACTS AND RELATIONSHIPS" in result
    assert "PERSONA FACTS" in result
    assert "Alex" in result
    assert "mother_of" in result
    assert "Sarah" in result
    mock_memory.facts.assert_awaited()


async def test_reflect_empty_facts(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    """Reflect with no facts returns empty-state messages."""
    mock_memory.facts.return_value = []

    result = await reflect(ctx)

    assert "No known" in result
    assert "No persona facts" in result


async def test_reflect_filters_active_only(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    """Reflect includes pinned and promoted facts but excludes candidates."""
    mock_memory.facts.return_value = [
        PersonaFact(
            subject="Alex",
            predicate="allergic_to",
            object="peanut",
            content="Severe peanut allergy",
            status="pinned",
        ),
        PersonaFact(
            subject="Alex",
            predicate="likes",
            object="jazz",
            content="Alex enjoys jazz music",
            status="promoted",
        ),
        PersonaFact(
            subject="Alex",
            predicate="visited",
            object="Rome",
            content="Alex visited Rome last summer",
            status="candidate",
        ),
    ]

    result = await reflect(ctx)

    assert "peanut" in result
    assert "jazz" in result
    assert "Rome" not in result


async def test_reflect_idempotent(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    """Calling reflect twice returns identical results."""
    first = await reflect(ctx)
    second = await reflect(ctx)

    assert first == second


async def test_recall_without_reflect_prepends_primer(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    """First recall without reflect should auto-prime with primer."""
    mock_memory.think_about.return_value = [
        Thought(
            trace=MemoryTrace(content="dinner recipe"),
            relevance=0.8,
            token_count=10,
            reconstruction="dinner recipe",
        )
    ]
    result = await recall("dinner ideas", ctx)
    assert "KNOWN FACTS AND RELATIONSHIPS" in result
    assert "dinner recipe" in result
    app = ctx.request_context.lifespan_context
    assert app.primed is True


async def test_recall_after_reflect_no_primer(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    """Recall after reflect should NOT prepend primer."""
    mock_memory.think_about.return_value = [
        Thought(
            trace=MemoryTrace(content="dinner recipe"),
            relevance=0.8,
            token_count=10,
            reconstruction="dinner recipe",
        )
    ]
    await reflect(ctx)
    result = await recall("dinner ideas", ctx)
    # Result should have the recall content but NOT start with primer
    # (primer was already delivered via reflect)
    assert "dinner recipe" in result
    # The recall result should be just the formatted thoughts, no primer prefix
    assert not result.startswith("KNOWN FACTS")


async def test_reflect_sets_primed_flag(ctx: MagicMock) -> None:
    """Calling reflect should set primed = True."""
    app = ctx.request_context.lifespan_context
    assert app.primed is False
    await reflect(ctx)
    assert app.primed is True
