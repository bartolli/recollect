"""MCP recall: every accepted token_budget bounds the returned text, no exception."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from recollect.core import CognitiveMemory, _estimate_tokens
from recollect.exceptions import EmbeddingError, MemorySDKError, StorageError
from recollect.models import MemoryTrace, PersonaFact
from recollect_mcp.server import (
    _MIN_RECALL_BUDGET,
    _RECALL_FAILED,
    _STORAGE_FAILED,
    AppContext,
    recall,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from unittest.mock import AsyncMock

_BUDGETS = [_MIN_RECALL_BUDGET, _MIN_RECALL_BUDGET + 1, 40, 100, 250, 800, 100_000]


def _ctx(memory: CognitiveMemory) -> MagicMock:
    fake = MagicMock()
    fake.request_context.lifespan_context = AppContext(
        memory=memory, worker=MagicMock(), user_id="alex"
    )
    fake.headers = None
    return fake


def _fact(obj: str, content: str, *, category: str, status: str) -> PersonaFact:
    return PersonaFact(
        subject="Alex",
        predicate="relates_to",
        object=obj,
        content=content,
        category=category,
        confidence=0.9,
        status=status,
    )


def _pinned(content: str) -> PersonaFact:
    return _fact("peanuts", content, category="health", status="pinned")


def _no_match(vectors: AsyncMock, facts: AsyncMock) -> None:
    return None


def _exhausted(vectors: AsyncMock, facts: AsyncMock) -> None:
    vectors.search_semantic.return_value = [(MemoryTrace(content="x" * 2000), 0.9)]


def _safety_pending(vectors: AsyncMock, facts: AsyncMock) -> None:
    facts.get_persona_facts.return_value = [_pinned("Alex carries a pen. " * 30)]
    vectors.search_semantic.return_value = [(MemoryTrace(content="weekend"), 0.8)]


def _safety_fits(vectors: AsyncMock, facts: AsyncMock) -> None:
    facts.get_persona_facts.return_value = [_pinned("Alex avoids peanuts")]
    vectors.search_semantic.return_value = [(MemoryTrace(content="x" * 2000), 0.9)]


def _mixed(vectors: AsyncMock, facts: AsyncMock) -> None:
    pinned = _pinned("Alex is allergic to peanuts")
    dietary = _fact("gluten", "Alex avoids gluten", category="dietary", status="pinned")
    jazz = _fact("jazz", "Alex likes jazz", category="preference", status="promoted")
    facts.get_persona_facts.return_value = [pinned, dietary]
    facts.search_facts_semantic.return_value = [(pinned, 0.9), (jazz, 0.85)]
    vectors.search_semantic.return_value = [
        (MemoryTrace(content="m" * 60 * (i + 1)), 0.8 - i * 0.05) for i in range(7)
    ]


@pytest.mark.parametrize("budget", _BUDGETS)
@pytest.mark.parametrize(
    "arrange", [_no_match, _exhausted, _safety_pending, _safety_fits, _mixed]
)
async def test_returned_text_within_every_accepted_budget(
    arrange: Callable[[AsyncMock, AsyncMock], None],
    budget: int,
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_vector_index: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    arrange(mock_vector_index, mock_fact_store)
    memory = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)

    text = await recall("what should Alex eat", _ctx(memory), token_budget=budget)

    assert _estimate_tokens(text) <= budget


@pytest.mark.parametrize(
    ("error", "notice"),
    [
        (StorageError("s" * 2000), _STORAGE_FAILED),
        (EmbeddingError("e" * 2000), _RECALL_FAILED),
    ],
)
async def test_failure_notice_is_bounded_and_keeps_the_net_pending(
    error: MemorySDKError,
    notice: str,
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_vector_index: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    mock_fact_store.get_persona_facts.return_value = [_pinned("Alex avoids peanuts")]
    mock_vector_index.search_semantic.side_effect = error
    ctx = _ctx(CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings))

    text = await recall("dinner", ctx, token_budget=_MIN_RECALL_BUDGET)

    # The exception text stays in the log: it is unbounded and not the client's.
    assert text == notice
    assert _estimate_tokens(text) <= _MIN_RECALL_BUDGET
    mock_vector_index.search_semantic.side_effect = None
    assert "peanuts" in await recall("dinner", ctx, token_budget=200)
