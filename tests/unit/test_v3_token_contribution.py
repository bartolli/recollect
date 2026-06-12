"""Contribution-gated Hebbian reinforcement.

A token earns strength only by propagating at least one non-seed trace
through the hop; seed-adjacency alone reinforces nothing. _token_hop
derives contributing token IDs from the single get_activated_trace_ids
query -- the absorbed get_tokens_for_traces second query is gone.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace

_D = 768


def _emb(s: float = 0.1) -> list[float]:
    return [s + i * 0.001 for i in range(_D)]


def _hop_row(
    trace_id: str, token_id: str, anchor_id: str
) -> tuple[str, str, str, float, float, str]:
    return (trace_id, token_id, "label", 0.8, 0.5, anchor_id)


class TestContributionGate:
    @pytest.mark.asyncio()
    async def test_only_propagating_token_reinforced(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        """Seed-only tokens yield zero hop rows (SQL contract) and gain nothing."""
        seed = MemoryTrace(id="seed-1", content="t", embedding=_emb())
        mock_storage.recall_tokens.get_activated_trace_ids = AsyncMock(
            return_value=[_hop_row("hop-1", "tok-prop", "seed-1")]
        )
        m = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        await m._activate_recall_tokens(_emb(), [(seed, 0.6)])
        mock_storage.recall_tokens.reinforce_tokens.assert_awaited_once()
        reinforced = mock_storage.recall_tokens.reinforce_tokens.call_args[0][0]
        assert reinforced == ["tok-prop"]

    @pytest.mark.asyncio()
    async def test_no_second_token_query(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        seed = MemoryTrace(id="seed-1", content="t", embedding=_emb())
        mock_storage.recall_tokens.get_activated_trace_ids = AsyncMock(
            return_value=[_hop_row("hop-1", "tok-prop", "seed-1")]
        )
        m = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        await m._activate_recall_tokens(_emb(), [(seed, 0.6)])
        mock_storage.recall_tokens.get_tokens_for_traces.assert_not_called()

    @pytest.mark.asyncio()
    async def test_excluded_rows_earn_no_credit(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        """Rows whose targets are already propagated credit nothing this hop."""
        mock_storage.recall_tokens.get_activated_trace_ids = AsyncMock(
            return_value=[_hop_row("hop-1", "tok-x", "seed-1")]
        )
        m = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        props, contributing = await m._token_hop(
            ["seed-1"], {"seed-1": 0.6}, 0.85, 0.1, exclude_ids=["hop-1"]
        )
        assert props == {}
        assert contributing == set()
