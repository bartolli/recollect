"""Witness-bounded propagation: rescued mass cannot outrank its testimony."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace


def _trace(trace_id: str) -> MemoryTrace:
    return MemoryTrace(content=f"trace-{trace_id}", significance=0.0)


def _fused(
    scores: dict[str, float],
    traces: dict[str, MemoryTrace],
    **kwargs: object,
) -> dict[str, float]:
    result = CognitiveMemory._compute_fused_scores(
        traces,
        scores,
        {},
        {},
        0.0,
        0.0,
        significance_weight=0.0,
        valence_weight=0.0,
        **kwargs,  # type: ignore[arg-type]
    )
    return {t.id: s for t, s in result}


class TestWitnessBound:
    def test_boosted_member_caps_at_witness_evidence(self) -> None:
        # t3q06 shape: member base 0.50 + prop 0.58*0.5 = 0.79 unbounded,
        # while the vouching seed's own evidence is 0.596 -- the bound
        # holds the member at the testimony's level.
        member = _trace("member")
        out = _fused(
            {member.id: 0.50},
            {member.id: member},
            token_bonuses={member.id: 0.58},
            propagation_blend=0.5,
            witness_evidence={member.id: 0.596},
            witness_bound_margin=0.0,
        )
        assert abs(out[member.id] - 0.596) < 1e-9

    def test_bound_never_lowers_own_evidence(self) -> None:
        # A member already above its witness keeps its own score:
        # propagation contributes zero under the bound, never a penalty.
        member = _trace("strong-member")
        out = _fused(
            {member.id: 0.70},
            {member.id: member},
            token_bonuses={member.id: 0.58},
            propagation_blend=0.5,
            witness_evidence={member.id: 0.596},
            witness_bound_margin=0.0,
        )
        assert abs(out[member.id] - 0.70) < 1e-9

    def test_negative_margin_disables_bound(self) -> None:
        # Sentinel: margin < 0 restores the unbounded formula exactly,
        # witness evidence present or not.
        member = _trace("unbounded")
        out = _fused(
            {member.id: 0.50},
            {member.id: member},
            token_bonuses={member.id: 0.58},
            propagation_blend=0.5,
            witness_evidence={member.id: 0.596},
            witness_bound_margin=-1.0,
        )
        assert abs(out[member.id] - (0.50 + 0.58 * 0.5)) < 1e-9

    def test_margin_grants_headroom_above_witness(self) -> None:
        member = _trace("headroom")
        out = _fused(
            {member.id: 0.50},
            {member.id: member},
            token_bonuses={member.id: 0.58},
            propagation_blend=0.5,
            witness_evidence={member.id: 0.596},
            witness_bound_margin=0.1,
        )
        assert abs(out[member.id] - 0.696) < 1e-9


class TestHopWitnessEvidence:
    @pytest.fixture()
    def mem(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_extractor: AsyncMock,
    ) -> CognitiveMemory:
        return CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=mock_extractor,
        )

    async def test_witness_is_best_vouching_seed_evidence(
        self, mem: CognitiveMemory, mock_storage: MagicMock
    ) -> None:
        # Two seeds vouch the same absent member; the witness level is the
        # stronger testimony, independent of which row wins the prop max.
        s1 = MemoryTrace(id="seed-1", content="s1", embedding=[0.1] * 8)
        s2 = MemoryTrace(id="seed-2", content="s2", embedding=[0.1] * 8)
        mock_storage.recall_tokens.get_activated_trace_ids.return_value = [
            ("member-1", "tok-1", "label", 0.4, 0.5, "seed-1"),
            ("member-1", "tok-2", "label", 0.9, 0.9, "seed-2"),
        ]
        props, witness = await mem._activate_recall_tokens(
            [0.1] * 8, [(s1, 0.6), (s2, 0.4)]
        )
        assert props["member-1"] == pytest.approx(
            0.4 * 0.85 * 0.9 * 0.9, abs=1e-6
        )
        assert witness["member-1"] == pytest.approx(0.6, abs=1e-9)

    async def test_seeds_stay_excluded_from_witness(
        self, mem: CognitiveMemory, mock_storage: MagicMock
    ) -> None:
        s1 = MemoryTrace(id="seed-1", content="s1", embedding=[0.1] * 8)
        mock_storage.recall_tokens.get_activated_trace_ids.return_value = [
            ("seed-1", "tok-1", "label", 0.8, 0.5, "seed-1"),
        ]
        props, witness = await mem._activate_recall_tokens(
            [0.1] * 8, [(s1, 0.6)]
        )
        assert props == {}
        assert witness == {}
