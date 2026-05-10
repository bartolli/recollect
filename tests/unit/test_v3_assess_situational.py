"""Unit tests for CognitiveMemory.assess_situational."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import SituationalAssessment, TokenAssessment
from recollect.models import MemoryTrace

_EMB_DIM = 768


def _emb(seed: float = 0.1) -> list[float]:
    return [seed + i * 0.001 for i in range(_EMB_DIM)]


_RELATED = MemoryTrace(id="related-1", content="seed trace", embedding=_emb(0.2))
_GROUP = [
    {
        "token_id": "tok-1",
        "label": "Sarah | health checkup | recovery progress",
        "strength": 1.0,
        "significance": 0.6,
        "stamped_trace_ids": ["related-1"],
        "status": "active",
    }
]


def _wire(mock_storage, mock_extractor, assessment, *, groups=None, related=None):
    mock_storage.vectors.search_semantic.return_value = (
        related if related is not None else [(_RELATED, 0.5)]
    )
    mock_storage.recall_tokens.find_groups_for_traces = AsyncMock(
        return_value=groups or []
    )
    mock_extractor._provider = AsyncMock()
    mock_extractor._provider.complete_structured = AsyncMock(return_value=assessment)


@pytest.fixture()
def mem(mock_storage, mock_embeddings, mock_extractor):
    return CognitiveMemory(
        storage=mock_storage,
        embeddings=mock_embeddings,
        extractor=mock_extractor,
    )


@pytest.fixture()
def trace():
    return MemoryTrace(id="probe-eval-1", content="new memory", embedding=_emb(0.3))


class TestAssessSituationalReturnShape:
    async def test_returns_wrapper_with_no_groups(
        self, mem, mock_storage, mock_extractor, trace
    ):
        _wire(mock_storage, mock_extractor, TokenAssessment(action="none"))
        result = await mem.assess_situational(trace)
        assert isinstance(result, SituationalAssessment)
        assert result.assessment.action == "none"
        assert result.related_trace_ids == ["related-1"]
        assert result.candidate_token_ids == []

    async def test_returns_wrapper_with_groups(
        self, mem, mock_storage, mock_extractor, trace
    ):
        _wire(
            mock_storage,
            mock_extractor,
            TokenAssessment(
                action="extend",
                group_number=1,
                implication="new concept",
            ),
            groups=_GROUP,
        )
        result = await mem.assess_situational(trace)
        assert result is not None
        assert result.assessment.action == "extend"
        assert result.assessment.group_number == 1
        assert result.candidate_token_ids == ["tok-1"]
        assert result.related_trace_ids == ["related-1"]


class TestAssessSituationalSkipPaths:
    async def test_returns_none_without_extractor(
        self, mock_storage, mock_embeddings, trace
    ):
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=None,
        )
        assert await mem.assess_situational(trace) is None

    async def test_returns_none_without_embedding(self, mem):
        traceless = MemoryTrace(id="no-emb", content="x", embedding=None)
        assert await mem.assess_situational(traceless) is None

    async def test_returns_none_when_no_related(
        self, mem, mock_storage, mock_extractor, trace
    ):
        _wire(mock_storage, mock_extractor, TokenAssessment(action="none"), related=[])
        assert await mem.assess_situational(trace) is None


class TestAssessSituationalNoApplyWrites:
    async def test_does_not_create_token(
        self, mem, mock_storage, mock_extractor, trace
    ):
        _wire(
            mock_storage,
            mock_extractor,
            TokenAssessment(
                action="create",
                linked_indices=[1],
                person_ref="Sarah",
                situation="x",
                implication="y",
            ),
        )
        await mem.assess_situational(trace)
        mock_storage.recall_tokens.create_token.assert_not_awaited()
        mock_storage.recall_tokens.stamp_traces.assert_not_awaited()

    async def test_does_not_update_token_on_extend(
        self, mem, mock_storage, mock_extractor, trace
    ):
        _wire(
            mock_storage,
            mock_extractor,
            TokenAssessment(
                action="extend",
                group_number=1,
                implication="new",
            ),
            groups=_GROUP,
        )
        await mem.assess_situational(trace)
        mock_storage.recall_tokens.update_token_label.assert_not_awaited()
        mock_storage.recall_tokens.stamp_traces.assert_not_awaited()

    async def test_does_not_update_token_on_revise(
        self, mem, mock_storage, mock_extractor, trace
    ):
        _wire(
            mock_storage,
            mock_extractor,
            TokenAssessment(
                action="revise",
                group_number=1,
                situation="updated",
                implication="resolved",
                significance=0.3,
            ),
            groups=_GROUP,
        )
        await mem.assess_situational(trace)
        mock_storage.recall_tokens.update_token.assert_not_awaited()


class TestAssessSituationalBypassesEnabledFlag:
    async def test_runs_when_recall_tokens_disabled(
        self,
        mock_storage,
        mock_embeddings,
        mock_extractor,
        trace,
    ):
        cfg = MemoryConfig()
        cfg._config["recall_tokens"]["enabled"] = False
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=mock_extractor,
            config=cfg,
        )
        _wire(mock_storage, mock_extractor, TokenAssessment(action="none"))
        result = await mem.assess_situational(trace)
        assert result is not None
        assert result.assessment.action == "none"
