"""Reactivation hook -- floor gate, channel-agnostic revival, lost races.

_reactivate_archived_candidates runs on the merged candidate list: any
archived candidate whose blended score clears retrieval.reactivation_floor
flips active (DB write + in-turn display copy); below the floor it passes
through unchanged. think_about wires the hook before selection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace

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


class TestReactivationHook:
    async def test_above_floor_revives(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        trace = _archived()
        out = await mem._reactivate_archived_candidates([(trace, 0.5)])
        mock_trace_store.reactivate_trace.assert_awaited_once()
        kwargs = mock_trace_store.reactivate_trace.await_args.kwargs
        assert kwargs["min_strength"] == pytest.approx(0.6)
        revived, score = out[0]
        assert revived.status == "active"
        assert revived.consolidated is False
        assert revived.strength == pytest.approx(0.6)
        assert score == pytest.approx(0.5)

    async def test_below_floor_passes_through(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        trace = _archived()
        out = await mem._reactivate_archived_candidates([(trace, 0.2)])
        mock_trace_store.reactivate_trace.assert_not_awaited()
        assert out[0][0].status == "archived"

    async def test_active_candidate_untouched(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        trace = MemoryTrace(content="live", strength=0.5)
        out = await mem._reactivate_archived_candidates([(trace, 0.9)])
        mock_trace_store.reactivate_trace.assert_not_awaited()
        assert out[0][0] is trace

    async def test_lost_race_keeps_candidate_as_read(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        # Concurrent hit already revived it: UPDATE 0 -> no display copy.
        mock_trace_store.reactivate_trace.return_value = False
        trace = _archived()
        out = await mem._reactivate_archived_candidates([(trace, 0.5)])
        assert out[0][0] is trace


class TestThinkAboutWiring:
    async def test_semantic_archived_candidate_reactivates(
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
