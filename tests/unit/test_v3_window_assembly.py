"""_assemble_window: slot policy, and the token budget over the whole window."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from recollect.core import CognitiveMemory, _estimate_tokens
from recollect.models import MemoryTrace, Thought

if TYPE_CHECKING:
    from collections.abc import Callable
    from unittest.mock import AsyncMock, MagicMock

_UNBOUND = 1_000_000


@pytest.fixture()
def mem(mock_storage: MagicMock, mock_embeddings: AsyncMock) -> CognitiveMemory:
    return CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)


def _fact(content: str, relevance: float, *, pinned: bool = True) -> Thought:
    return Thought(
        trace=MemoryTrace(content=content, pattern={"persona_fact": True}),
        relevance=relevance,
        token_count=_estimate_tokens(content),
        reconstructed=True,
        reconstruction=content,
        pinned=pinned,
    )


def _cand(content: str, score: float) -> tuple[MemoryTrace, float]:
    return MemoryTrace(content=content), score


def _assemble(
    mem: CognitiveMemory,
    candidates: list[tuple[MemoryTrace, float]],
    facts: list[Thought],
    *,
    token_budget: int = _UNBOUND,
    render: Callable[[list[Thought]], str] | None = None,
) -> list[Thought]:
    return mem._assemble_window(
        candidates, facts, token_budget=token_budget, render=render
    ).thoughts


def _contents(window: list[Thought]) -> list[str | None]:
    return [t.trace.content for t in window]


def _is_trace(t: Thought) -> bool:
    return not t.trace.pattern.get("persona_fact")


class TestSlotPolicy:
    """Budget not binding: today's pinned / trace-guarantee / relevance policy."""

    def test_traces_guaranteed_when_facts_dominate(self, mem: CognitiveMemory) -> None:
        facts = [_fact(f"fact{i}", 0.9 - i * 0.05) for i in range(5)]
        traces = [_cand(f"trace{i}", 0.7 - i * 0.1) for i in range(2)]
        result = _assemble(mem, traces, facts)
        assert len([t for t in result if _is_trace(t)]) >= 2

    def test_facts_fill_when_few_traces(self, mem: CognitiveMemory) -> None:
        facts = [_fact(f"fact{i}", 0.9 - i * 0.05) for i in range(6)]
        result = _assemble(mem, [_cand("trace0", 0.8)], facts)
        assert len(result) == 7
        assert len([t for t in result if _is_trace(t)]) == 1

    def test_high_relevance_trace_survives(self, mem: CognitiveMemory) -> None:
        facts = [_fact(f"fact{i}", 0.5) for i in range(6)]
        traces = [
            _cand("strong_trace", 0.95),
            _cand("weak_trace", 0.3),
            _cand("mid_trace", 0.6),
        ]
        result = _assemble(mem, traces, facts)
        trace_contents = [t.trace.content for t in result if _is_trace(t)]
        assert "strong_trace" in trace_contents
        assert "mid_trace" in trace_contents

    def test_result_sorted_by_relevance(self, mem: CognitiveMemory) -> None:
        result = _assemble(mem, [_cand("trace", 0.9)], [_fact("fact", 0.5)])
        assert result[0].relevance >= result[1].relevance

    def test_max_retrievals_caps_traces_in_rank_order(
        self, mem: CognitiveMemory
    ) -> None:
        traces = [_cand(f"trace{i}", 0.9 - i * 0.05) for i in range(9)]
        assert _contents(_assemble(mem, traces, [])) == [f"trace{i}" for i in range(7)]

    def test_ties_keep_traces_ahead_of_facts(self, mem: CognitiveMemory) -> None:
        result = _assemble(
            mem,
            [_cand("t1", 0.9), _cand("t2", 0.8), _cand("t3", 0.5)],
            [_fact("fact", 0.5, pinned=False)],
        )
        assert _contents(result) == ["t1", "t2", "t3", "fact"]


class TestTokenBudget:
    def test_oversized_first_trace_does_not_end_selection(
        self, mem: CognitiveMemory
    ) -> None:
        traces = [_cand("x" * 2000, 0.9), _cand("fits", 0.8)]
        result = _assemble(mem, traces, [], token_budget=100)
        assert _contents(result) == ["fits"]

    def test_budget_covers_facts_with_traces(self, mem: CognitiveMemory) -> None:
        fact = _fact("f" * 120, 0.99)
        traces = [_cand(f"{i}" * 160, 0.5 - i * 0.1) for i in range(3)]
        result = _assemble(mem, traces, [fact], token_budget=100)
        assert sum(t.token_count for t in result) <= 100
        assert _contents(result) == ["f" * 120, "0" * 160]

    def test_no_trace_fits_keeps_fitting_fact(self, mem: CognitiveMemory) -> None:
        traces = [_cand("x" * 2000, 0.9), _cand("y" * 2000, 0.8)]
        result = _assemble(mem, traces, [_fact("fact", 0.4)], token_budget=100)
        assert _contents(result) == ["fact"]


def _render(window: list[Thought]) -> str:
    return "Related memories:\n" + "\n".join(f"- {t.reconstruction}" for t in window)


def _sized(label: str) -> str:
    return label.ljust(38, ".")


class TestRender:
    """render given: its estimate is the cost; drops go lowest priority first."""

    def _items(self) -> tuple[list[tuple[MemoryTrace, float]], list[Thought]]:
        traces = [_cand(_sized(f"t{r}"), r / 10) for r in (9, 8, 7)]
        # Pinned at low relevance: first by priority, last by relevance.
        return traces, [_fact(_sized("fact"), 0.3)]

    def test_estimate_of_render_within_budget(self, mem: CognitiveMemory) -> None:
        traces, facts = self._items()
        result = _assemble(mem, traces, facts, token_budget=30, render=_render)
        assert _estimate_tokens(_render(result)) <= 30
        assert _contents(result) == [_sized("t9"), _sized("fact")]

    def test_without_render_the_same_budget_holds_more(
        self, mem: CognitiveMemory
    ) -> None:
        traces, facts = self._items()
        result = _assemble(mem, traces, facts, token_budget=30)
        assert _contents(result) == [_sized("t9"), _sized("t8"), _sized("fact")]
