"""Retrieval credit follows the window think_about returns, and only it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace, PersonaFact

if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock

    from recollect.models import Thought


@pytest.fixture()
def mem(mock_storage: MagicMock, mock_embeddings: AsyncMock) -> CognitiveMemory:
    return CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)


def _fact(i: int) -> PersonaFact:
    return PersonaFact(
        id=f"fact-{i}",
        subject="Alex",
        predicate="likes",
        object=f"thing {i}",
        content=f"Alex likes thing {i}",
        category="preference",
        confidence=0.9,
        status="promoted",
    )


def _written(mock: AsyncMock) -> list[str]:
    return [c.args[0] for c in mock.await_args_list]


def _trace_ids(thoughts: list[Thought]) -> list[str]:
    return [t.trace.id for t in thoughts if not t.trace.pattern.get("persona_fact")]


async def test_credit_follows_the_returned_window(
    mem: CognitiveMemory,
    mock_vector_index: AsyncMock,
    mock_fact_store: AsyncMock,
    mock_trace_store: AsyncMock,
) -> None:
    traces = [MemoryTrace(id=f"t{i}", content=f"memory {i}") for i in range(7)]
    mock_vector_index.search_semantic.return_value = [
        (t, 0.5 - i * 0.01) for i, t in enumerate(traces)
    ]
    mock_fact_store.search_facts_semantic.return_value = [
        (_fact(i), 0.95) for i in range(5)
    ]
    window = await mem.think_about("what does Alex like")
    assert len(window) == 7
    assert _trace_ids(window) == ["t0", "t1"]
    assert sorted(_written(mock_trace_store.apply_strength_factor)) == ["t0", "t1"]
    assert sorted(_written(mock_trace_store.mark_retrieved)) == ["t0", "t1"]


async def test_forgotten_candidate_is_neither_delivered_nor_written(
    mem: CognitiveMemory,
    mock_vector_index: AsyncMock,
    mock_trace_store: AsyncMock,
) -> None:
    gone = MemoryTrace(id="gone", content="retracted", status="forgotten")
    live = MemoryTrace(id="live", content="kept")
    mock_vector_index.search_semantic.return_value = [(gone, 0.95), (live, 0.5)]
    window = await mem.think_about("anything")
    assert _trace_ids(window) == ["live"]
    for mock in (
        mock_trace_store.apply_strength_factor,
        mock_trace_store.mark_retrieved,
        mock_trace_store.reactivate_trace,
    ):
        assert "gone" not in _written(mock)


async def test_oversized_first_trace_does_not_end_selection(
    mem: CognitiveMemory,
    mock_vector_index: AsyncMock,
    mock_trace_store: AsyncMock,
) -> None:
    big = MemoryTrace(id="big", content="x" * 2000)
    small = MemoryTrace(id="small", content="fits")
    mock_vector_index.search_semantic.return_value = [(big, 0.9), (small, 0.8)]
    window = await mem.think_about("anything", token_budget=100)
    assert _trace_ids(window) == ["small"]
    assert _written(mock_trace_store.mark_retrieved) == ["small"]


async def test_no_trace_fits_returns_fitting_fact_and_credits_nothing(
    mem: CognitiveMemory,
    mock_vector_index: AsyncMock,
    mock_fact_store: AsyncMock,
    mock_trace_store: AsyncMock,
) -> None:
    mock_vector_index.search_semantic.return_value = [
        (MemoryTrace(id=f"big{i}", content="x" * 2000), 0.9 - i * 0.1) for i in range(3)
    ]
    mock_fact_store.search_facts_semantic.return_value = [(_fact(0), 0.95)]
    window = await mem.think_about("what does Alex like", token_budget=100)
    assert _trace_ids(window) == []
    assert [t.trace.pattern.get("persona_fact") for t in window] == [True]
    mock_trace_store.apply_strength_factor.assert_not_awaited()
    mock_trace_store.mark_retrieved.assert_not_awaited()
    mock_trace_store.reactivate_trace.assert_not_awaited()


async def test_window_reports_an_oversized_trace_as_withheld(
    mem: CognitiveMemory,
    mock_vector_index: AsyncMock,
) -> None:
    big = MemoryTrace(id="big", content="x" * 2000)
    small = MemoryTrace(id="small", content="fits")
    mock_vector_index.search_semantic.return_value = [(big, 0.9), (small, 0.8)]
    window = await mem.think_about_window("anything", token_budget=100)
    assert _trace_ids(window.thoughts) == ["small"]
    assert window.withheld == 1


async def test_window_full_by_rank_withholds_nothing(
    mem: CognitiveMemory,
    mock_vector_index: AsyncMock,
) -> None:
    mock_vector_index.search_semantic.return_value = [
        *(
            (MemoryTrace(id=f"t{i}", content=f"memory {i}"), 0.9 - i * 0.05)
            for i in range(7)
        ),
        (MemoryTrace(id="big", content="x" * 2000), 0.1),
    ]
    window = await mem.think_about_window("anything", token_budget=100)
    assert _trace_ids(window.thoughts) == [f"t{i}" for i in range(7)]
    assert window.withheld == 0


async def test_think_about_returns_the_window_thoughts_credited_once(
    mem: CognitiveMemory,
    mock_vector_index: AsyncMock,
    mock_trace_store: AsyncMock,
) -> None:
    mock_vector_index.search_semantic.return_value = [
        (MemoryTrace(id=f"t{i}", content=f"memory {i}"), 0.9 - i * 0.1)
        for i in range(3)
    ]
    window = await mem.think_about_window("anything")
    mock_trace_store.mark_retrieved.reset_mock()
    thoughts = await mem.think_about("anything")
    assert _trace_ids(thoughts) == _trace_ids(window.thoughts) == ["t0", "t1", "t2"]
    assert sorted(_written(mock_trace_store.mark_retrieved)) == ["t0", "t1", "t2"]
