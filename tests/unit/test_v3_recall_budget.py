"""MCP recall: the complete response, safety context included, fits token_budget."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from recollect.core import CognitiveMemory, _estimate_tokens
from recollect.models import MemoryTrace, PersonaFact
from recollect_mcp.server import (
    _BUDGET_EXHAUSTED,
    _MIN_RECALL_BUDGET,
    _NO_MATCH,
    _SAFETY_PENDING,
    AppContext,
    _format_thoughts,
    recall,
)

if TYPE_CHECKING:
    from unittest.mock import AsyncMock

_PRINTED_ID = re.compile(r"^- \[([0-9a-f]{8})\] ", re.MULTILINE)
_LONG_FACT = "Alex carries an epinephrine pen. " * 12


def _ctx(memory: CognitiveMemory) -> MagicMock:
    fake = MagicMock()
    fake.request_context.lifespan_context = AppContext(
        memory=memory, worker=MagicMock(), user_id="alex"
    )
    fake.headers = None
    return fake


def _pinned(content: str) -> PersonaFact:
    return PersonaFact(
        subject="Alex",
        predicate="is_allergic_to",
        object="peanuts",
        content=content,
        category="health",
        confidence=0.9,
        status="pinned",
    )


def _credited(mock: AsyncMock) -> list[str]:
    return sorted(c.args[0][:8] for c in mock.await_args_list)


@pytest.fixture()
def memory(mock_storage: MagicMock, mock_embeddings: AsyncMock) -> CognitiveMemory:
    return CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)


async def test_printed_text_fits_budget_and_credit_matches_print(
    memory: CognitiveMemory,
    mock_vector_index: AsyncMock,
    mock_fact_store: AsyncMock,
    mock_trace_store: AsyncMock,
) -> None:
    pinned = _pinned("Alex is allergic to peanuts")
    dietary = PersonaFact(
        subject="Alex",
        predicate="avoids",
        object="gluten",
        content="Alex avoids gluten",
        category="dietary",
        status="promoted",
    )
    mock_fact_store.get_persona_facts.return_value = [pinned, dietary]
    mock_fact_store.search_facts_semantic.return_value = [(pinned, 0.9)]
    mock_vector_index.search_semantic.return_value = [
        (MemoryTrace(id=f"{i:08x}-trace", content="m" * 760), 0.6 - i * 0.02)
        for i in range(7)
    ]

    text = await recall("what should Alex eat", _ctx(memory), token_budget=800)

    assert _estimate_tokens(text) <= 800
    assert "peanuts" in text
    assert "gluten" in text
    printed = sorted(_PRINTED_ID.findall(text))
    assert 0 < len(printed) < 7
    assert _credited(mock_trace_store.mark_retrieved) == printed
    assert _credited(mock_trace_store.apply_strength_factor) == printed


async def test_budget_below_minimum_rejected_before_the_net_is_claimed(
    memory: CognitiveMemory,
    mock_embeddings: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    mock_fact_store.get_persona_facts.return_value = [_pinned("Alex avoids peanuts")]
    ctx = _ctx(memory)

    # ToolError: its text reaches the client; any other exception is replaced
    # by a generic message and the minimum would stay on the server.
    with pytest.raises(ToolError, match=f"at least {_MIN_RECALL_BUDGET}"):
        await recall("dinner", ctx, token_budget=_MIN_RECALL_BUDGET - 1)
    with pytest.raises(ToolError, match="non-empty"):
        await recall("  ", ctx, token_budget=200)

    assert not ctx.request_context.lifespan_context.primed
    mock_embeddings.generate_embedding.assert_not_awaited()
    assert f"at least {_MIN_RECALL_BUDGET}" in (recall.__doc__ or "")
    assert "peanuts" in await recall("dinner", ctx, token_budget=200)


async def test_safety_net_that_does_not_fit_stays_pending(
    memory: CognitiveMemory,
    mock_vector_index: AsyncMock,
    mock_fact_store: AsyncMock,
    mock_trace_store: AsyncMock,
) -> None:
    mock_fact_store.get_persona_facts.return_value = [_pinned(_LONG_FACT)]
    mock_vector_index.search_semantic.return_value = [
        (MemoryTrace(content="weekend plans"), 0.8)
    ]
    ctx = _ctx(memory)

    with patch.object(
        memory, "think_about_window", wraps=memory.think_about_window
    ) as spy:
        text = await recall("weekend ideas", ctx, token_budget=50)
        assert text == _SAFETY_PENDING
        assert _estimate_tokens(text) <= 50
        spy.assert_not_awaited()
        mock_trace_store.mark_retrieved.assert_not_awaited()

        retry = await recall("weekend ideas", ctx, token_budget=2000)
    assert _LONG_FACT.strip() in retry


async def test_matched_memories_none_fitting_reads_budget_exhausted(
    memory: CognitiveMemory,
    mock_vector_index: AsyncMock,
    mock_trace_store: AsyncMock,
) -> None:
    mock_vector_index.search_semantic.return_value = [
        (MemoryTrace(content="x" * 2000), 0.9)
    ]

    text = await recall("anything", _ctx(memory), token_budget=_MIN_RECALL_BUDGET)

    assert text == _BUDGET_EXHAUSTED
    assert _estimate_tokens(text) <= _MIN_RECALL_BUDGET
    mock_trace_store.mark_retrieved.assert_not_awaited()
    mock_trace_store.apply_strength_factor.assert_not_awaited()


async def test_no_candidates_reads_no_match(memory: CognitiveMemory) -> None:
    assert await recall("anything", _ctx(memory)) == _NO_MATCH


async def test_safety_net_fits_and_no_trace_fits_prints_the_net_alone(
    memory: CognitiveMemory,
    mock_vector_index: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    fact = _pinned("Alex avoids peanuts")
    mock_fact_store.get_persona_facts.return_value = [fact]
    mock_vector_index.search_semantic.return_value = [
        (MemoryTrace(content="x" * 2000), 0.9)
    ]

    text = await recall("anything", _ctx(memory), token_budget=_MIN_RECALL_BUDGET)

    assert text == _format_thoughts([], safety_facts=[fact])
    assert _estimate_tokens(text) <= _MIN_RECALL_BUDGET
