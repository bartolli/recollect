"""Unit tests for situational recall token integration."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import TokenAssessment
from recollect.models import MemoryTrace, RecallToken

_EMB_DIM = 768
_RELATED = MemoryTrace(
    id="related-1", content="Sarah is feeling better",
    embedding=[0.2 + i * 0.001 for i in range(_EMB_DIM)],
)
_GROUP = [{"token_id": "tok-1", "label": "Sarah | health checkup | recovery progress",
    "strength": 1.0, "significance": 0.6, "stamped_trace_ids": ["related-1"]}]
_ARCHIVED_GROUP = [{"token_id": "tok-archived", "label": "Sarah | old | old imp",
    "strength": 0.005, "significance": 0.7, "stamped_trace_ids": ["related-1"],
    "status": "archived"}]
_ACTIVE_GROUP = [{"token_id": "tok-active", "label": "Sarah | current | cur imp",
    "strength": 0.8, "significance": 0.6, "stamped_trace_ids": ["related-1"],
    "status": "active"}]


def _emb(seed: float = 0.1) -> list[float]:
    return [seed + i * 0.001 for i in range(_EMB_DIM)]


def _setup(mock_storage, mock_extractor, assessment, *, groups=None):
    """Configure mocks for write-time token assessment tests."""
    mock_storage.vectors.search_semantic.return_value = [(_RELATED, 0.5)]
    mock_storage.recall_tokens.find_groups_for_traces = AsyncMock(
        return_value=groups or []
    )
    mock_extractor._provider = AsyncMock()
    mock_extractor._provider.complete_structured = AsyncMock(return_value=assessment)


@pytest.fixture()
def mem(mock_storage, mock_embeddings, mock_extractor):
    return CognitiveMemory(
        storage=mock_storage, embeddings=mock_embeddings, extractor=mock_extractor,
    )


class TestWriteTimeAssessment:
    async def test_create_action(self, mem, mock_storage, mock_extractor):
        _setup(mock_storage, mock_extractor, TokenAssessment(
            action="create", linked_indices=[1],
            person_ref="Sarah", situation="health checkup", implication="mother Sarah",
        ))
        await mem.experience("Sarah's health checkup went well")
        mock_storage.recall_tokens.create_token.assert_awaited_once()
        mock_storage.recall_tokens.stamp_traces.assert_awaited_once()
        stamped_ids = mock_storage.recall_tokens.stamp_traces.call_args[0][1]
        assert "related-1" in stamped_ids
        assert len(stamped_ids) == 2

    async def test_extend_action(self, mem, mock_storage, mock_extractor):
        _setup(mock_storage, mock_extractor, TokenAssessment(
            action="extend", group_number=1, implication="new concept",
        ), groups=_GROUP)
        await mem.experience("Sarah mentioned a new treatment")
        mock_storage.recall_tokens.stamp_traces.assert_awaited_once()
        stamped = mock_storage.recall_tokens.stamp_traces.call_args[0][1]
        assert len(stamped) == 1
        mock_storage.recall_tokens.update_token_label.assert_awaited_once()

    async def test_revise_action(self, mem, mock_storage, mock_extractor):
        _setup(mock_storage, mock_extractor, TokenAssessment(
            action="revise", group_number=1,
            situation="updated", implication="new state", significance=0.3,
        ), groups=_GROUP)
        await mem.experience("Sarah's diagnosis was revised")
        mock_storage.recall_tokens.update_token.assert_awaited_once()
        args = mock_storage.recall_tokens.update_token.call_args[0]
        assert args[0] == "tok-1"
        assert "new state" in args[1]
        assert args[2] == pytest.approx(0.3)
        mock_storage.recall_tokens.stamp_traces.assert_awaited_once()

    async def test_none_action(self, mem, mock_storage, mock_extractor):
        _setup(mock_storage, mock_extractor, TokenAssessment(action="none"))
        await mem.experience("Unrelated thought")
        mock_storage.recall_tokens.create_token.assert_not_awaited()

    async def test_disabled(self, mock_storage, mock_embeddings, mock_extractor):
        cfg = MemoryConfig()
        cfg._config["recall_tokens"]["enabled"] = False
        mem = CognitiveMemory(
            storage=mock_storage, embeddings=mock_embeddings,
            extractor=mock_extractor, config=cfg,
        )
        await mem.experience("Anything")
        mock_storage.recall_tokens.create_token.assert_not_awaited()


class TestQueryTimeActivation:
    async def test_propagation_formula(self, mem, mock_storage):
        t1 = MemoryTrace(id="seed-1", content="test", embedding=_emb())
        mock_storage.recall_tokens.get_activated_trace_ids.return_value = [
            ("activated-1", "mother-sarah", 0.8, 0.5, "seed-1"),
        ]
        mock_storage.recall_tokens.get_tokens_for_traces.return_value = [
            (RecallToken(id="tok-1", label="mother-sarah"), "seed-1"),
        ]
        result = await mem._activate_recall_tokens(_emb(), [(t1, 0.6)])
        assert "activated-1" in result
        expected = 0.6 * 0.85 * 0.8 * 0.5  # anchor * hop_decay * strength * sig
        assert result["activated-1"] == pytest.approx(expected, abs=0.001)

    async def test_disabled(self, mock_storage, mock_embeddings, mock_extractor):
        cfg = MemoryConfig()
        cfg._config["recall_tokens"]["enabled"] = False
        mem = CognitiveMemory(
            storage=mock_storage, embeddings=mock_embeddings,
            extractor=mock_extractor, config=cfg,
        )
        t1 = MemoryTrace(id="seed-1", content="test", embedding=_emb())
        result = await mem._activate_recall_tokens(_emb(), [(t1, 0.6)])
        assert result == {}


class TestScoringBlend:
    def test_additive_blend(self):
        t1 = MemoryTrace(id="t1", content="test")
        result = CognitiveMemory._compute_fused_scores(
            {"t1": t1}, {"t1": 0.7},
            activation_levels={}, entity_sims={},
            spread_bonus=0.1, entity_bonus=0.1,
            token_bonuses={"t1": 0.08},
        )
        _trace, score = result[0]
        expected = 0.7 + 0.015 + 0.0 + 0.04  # base + sig + val + token
        assert score == pytest.approx(expected, abs=0.001)

    def test_additive_blend_with_zero_base(self):
        t1 = MemoryTrace(id="t1", content="test")
        result = CognitiveMemory._compute_fused_scores(
            {"t1": t1}, {"t1": 0.0},
            activation_levels={}, entity_sims={},
            spread_bonus=0.1, entity_bonus=0.1,
            token_bonuses={"t1": 0.08},
        )
        _trace, score = result[0]
        assert score == pytest.approx(0.015 + 0.04, abs=0.001)  # sig + token
        assert score > 0.0


class TestStaticHelpers:
    def test_append_implication(self):
        label = "Sarah | health checkup | recovery progress"
        result = CognitiveMemory._append_implication(label, "new symptom")
        assert result == "Sarah | health checkup | recovery progress, new symptom"

    def test_format_existing_groups(self):
        related: list[tuple[MemoryTrace, float]] = [
            (MemoryTrace(id="related-1", content="trace 1"), 0.5),
        ]
        result = CognitiveMemory._format_existing_groups(related, _GROUP)
        assert "G1:" in result
        assert "Sarah | health checkup" in result
        assert "significance: 0.6" in result

    def test_format_existing_groups_empty(self):
        assert CognitiveMemory._format_existing_groups([], []) == "None"


class TestForgetCleanup:
    async def test_forget_deletes_token_stamps(self, mem, mock_storage):
        mock_storage.traces.delete_trace.return_value = True
        await mem.forget("trace-abc-123")
        mock_storage.recall_tokens.delete_by_trace.assert_awaited_once_with(
            "trace-abc-123"
        )


class TestTokenArchiving:
    async def test_decay_archives_not_deletes(self, mem, mock_storage):
        """Verify decay_inactive is called (archiving behavior is in the store)."""
        await mem.consolidate()
        mock_storage.recall_tokens.decay_inactive.assert_awaited_once()

    async def test_extend_reactivates_archived_token(self, mem, mock_storage, mock_extractor):
        _setup(mock_storage, mock_extractor, TokenAssessment(
            action="extend", group_number=1, implication="new relevance",
        ), groups=_ARCHIVED_GROUP)
        await mem.experience("Sarah's old situation resurfaced")
        mock_storage.recall_tokens.reinforce_tokens.assert_awaited()
        assert mock_storage.recall_tokens.reinforce_tokens.call_args[0][0] == ["tok-archived"]

    async def test_active_extend_does_not_reinforce(self, mem, mock_storage, mock_extractor):
        _setup(mock_storage, mock_extractor, TokenAssessment(
            action="extend", group_number=1, implication="more info",
        ), groups=_ACTIVE_GROUP)
        await mem.experience("Sarah mentioned something new")
        mock_storage.recall_tokens.reinforce_tokens.assert_not_awaited()
