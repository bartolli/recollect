"""Reactivation -- floor gate on delivered traces, lost races, forgotten drop.

_drop_forgotten removes forgotten candidates before assembly, with no
write. _credit_delivered revives a delivered archived trace whose fused
score clears retrieval.reactivation_floor (DB write + in-turn display
copy); below the floor it is delivered unrevived. An undelivered
candidate is never revived.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory, _drop_forgotten
from recollect.models import MemoryTrace, Thought

_EMB_DIM = 768


def _fake_embedding(seed: float = 0.1) -> list[float]:
    return [seed + i * 0.001 for i in range(_EMB_DIM)]


@pytest.fixture()
def mem(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_extractor: AsyncMock,
) -> CognitiveMemory:
    return CognitiveMemory(
        storage=mock_storage,
        embeddings=mock_embeddings,
        extractor=mock_extractor,
    )


def _archived(strength: float = 0.05, significance: float = 0.6) -> MemoryTrace:
    return MemoryTrace(
        id="arch-1",
        content="archived",
        strength=strength,
        significance=significance,
        status="archived",
        consolidated=True,
    )


def _delivered(trace: MemoryTrace, relevance: float) -> list[Thought]:
    return [Thought(trace=trace, relevance=relevance)]


class TestCreditDelivered:
    async def test_above_floor_revives(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        out = await mem._credit_delivered(_delivered(_archived(), 0.5))
        mock_trace_store.reactivate_trace.assert_awaited_once()
        kwargs = mock_trace_store.reactivate_trace.await_args.kwargs
        assert kwargs["min_strength"] == pytest.approx(0.6)
        revived = out[0].trace
        assert revived.status == "active"
        assert revived.consolidated is False
        assert revived.strength == pytest.approx(0.6)
        assert out[0].relevance == pytest.approx(0.5)

    async def test_below_floor_delivered_unrevived_with_credit(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        out = await mem._credit_delivered(_delivered(_archived(), 0.2))
        mock_trace_store.reactivate_trace.assert_not_awaited()
        mock_trace_store.mark_retrieved.assert_awaited_once_with("arch-1")
        assert out[0].trace.status == "archived"

    async def test_active_trace_untouched(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        trace = MemoryTrace(content="live", strength=0.5)
        out = await mem._credit_delivered(_delivered(trace, 0.9))
        mock_trace_store.reactivate_trace.assert_not_awaited()
        assert out[0].trace is trace

    async def test_lost_race_keeps_trace_as_read(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        # Concurrent hit already revived it: UPDATE 0 -> no display copy.
        mock_trace_store.reactivate_trace.return_value = False
        trace = _archived()
        out = await mem._credit_delivered(_delivered(trace, 0.5))
        assert out[0].trace is trace


def test_forgotten_dropped_regardless_of_score() -> None:
    # Explicit retraction: a high-relevance recall must not undo
    # the forget by reviving or even surfacing the trace.
    gone = MemoryTrace(content="retracted", strength=0.8, status="forgotten")
    kept = _archived()
    assert _drop_forgotten([(gone, 0.95), (kept, 0.5)]) == [(kept, 0.5)]


class TestThinkAboutWiring:
    async def test_delivered_archived_candidate_reactivates(
        self,
        mem: CognitiveMemory,
        mock_vector_index: AsyncMock,
        mock_trace_store: AsyncMock,
    ) -> None:
        trace = MemoryTrace(
            content="archived but relevant",
            embedding=_fake_embedding(),
            strength=0.5,
            significance=0.3,
            status="archived",
        )
        mock_vector_index.search_semantic.return_value = [(trace, 0.9)]
        await mem.think_about("archived but relevant")
        mock_trace_store.reactivate_trace.assert_awaited_once()

    async def test_undelivered_archived_candidate_stays_archived(
        self,
        mem: CognitiveMemory,
        mock_vector_index: AsyncMock,
        mock_trace_store: AsyncMock,
    ) -> None:
        archived = _archived().model_copy(update={"content": "x" * 2000})
        live = MemoryTrace(id="live", content="fits")
        mock_vector_index.search_semantic.return_value = [
            (archived, 0.9),
            (live, 0.5),
        ]
        thoughts = await mem.think_about("anything", token_budget=100)
        assert [t.trace.id for t in thoughts] == ["live"]
        mock_trace_store.reactivate_trace.assert_not_awaited()
        boosts = mock_trace_store.apply_strength_factor.await_args_list
        assert "arch-1" not in [c.args[0] for c in boosts]
