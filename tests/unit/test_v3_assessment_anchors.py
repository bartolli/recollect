"""Anchored candidate collection for write-time token assessment.

Entity/temporal association neighbors and recent same-session traces enter
assessment inputs without passing the semantic threshold -- the write-time
rescue for semantically opaque chain tails (story-7 direction a).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from recollect.core import CognitiveMemory
from recollect.llm.types import TokenAssessment
from recollect.models import Association, MemoryTrace

_EMB_DIM = 768


def _emb(seed: float = 0.1) -> list[float]:
    return [seed + i * 0.001 for i in range(_EMB_DIM)]


def _trace(
    tid: str, *, user_id: str | None = "u1", session_id: str | None = None
) -> MemoryTrace:
    return MemoryTrace(
        id=tid,
        content=f"content of {tid}",
        embedding=_emb(0.3),
        user_id=user_id,
        session_id=session_id,
    )


def _entity_edge(source: str, target: str) -> Association:
    return Association(
        source_trace_id=source,
        target_trace_id=target,
        association_type="entity",
        weight=0.7,
        forward_strength=0.7,
        backward_strength=0.7,
    )


def _wire_assessor(mock_extractor) -> None:
    mock_extractor._provider = AsyncMock()
    mock_extractor._provider.complete_structured = AsyncMock(
        return_value=TokenAssessment(action="none")
    )


@pytest.fixture()
def mem(mock_storage, mock_embeddings, mock_extractor):
    _wire_assessor(mock_extractor)
    return CognitiveMemory(
        storage=mock_storage,
        embeddings=mock_embeddings,
        extractor=mock_extractor,
    )


class TestEntityAnchoredCandidates:
    async def test_entity_neighbor_enters_when_semantic_empty(self, mem, mock_storage):
        new = _trace("tail-1")
        neighbor = _trace("bridge-1")
        mock_storage.vectors.search_semantic.return_value = []
        mock_storage.associations.get_associations.return_value = [
            _entity_edge("bridge-1", "tail-1")
        ]
        mock_storage.traces.get_traces_bulk.return_value = [neighbor]

        result = await mem.assess_situational(new)

        assert result is not None
        assert result.related_trace_ids == ["bridge-1"]


class TestSessionRecentCandidates:
    async def test_session_neighbor_enters_without_edges(self, mem, mock_storage):
        new = _trace("tail-1", session_id="s1")
        neighbor = _trace("earlier-1", session_id="s1")
        mock_storage.vectors.search_semantic.return_value = []
        mock_storage.associations.get_associations.return_value = []
        mock_storage.traces.get_traces_by_session.return_value = [neighbor, new]

        result = await mem.assess_situational(new)

        assert result is not None
        assert result.related_trace_ids == ["earlier-1"]
        mock_storage.traces.get_traces_by_session.assert_awaited_once()


class TestUserIsolation:
    async def test_cross_user_anchors_never_enter(self, mem, mock_storage):
        new = _trace("tail-1", user_id="u1", session_id="s1")
        foreign_edge = _trace("foreign-1", user_id="u2")
        foreign_session = _trace("foreign-2", user_id="u2", session_id="s1")
        mock_storage.vectors.search_semantic.return_value = []
        mock_storage.associations.get_associations.return_value = [
            _entity_edge("foreign-1", "tail-1")
        ]
        mock_storage.traces.get_traces_bulk.return_value = [foreign_edge]
        mock_storage.traces.get_traces_by_session.return_value = [foreign_session, new]

        assert await mem.assess_situational(new) is None


class TestUnionShape:
    async def test_semantic_and_anchor_twin_appears_once(self, mem, mock_storage):
        new = _trace("tail-1")
        bridge = _trace("bridge-1")
        mock_storage.vectors.search_semantic.return_value = [(bridge, 0.9)]
        mock_storage.associations.get_associations.return_value = [
            _entity_edge("bridge-1", "tail-1")
        ]

        result = await mem.assess_situational(new)

        assert result is not None
        assert result.related_trace_ids == ["bridge-1"]
        mock_storage.traces.get_traces_bulk.assert_not_awaited()

    async def test_anchored_set_capped_at_anchor_k(self, mem, mock_storage):
        new = _trace("tail-1")
        neighbors = [_trace(f"n-{i}") for i in range(7)]
        mock_storage.vectors.search_semantic.return_value = []
        mock_storage.associations.get_associations.return_value = [
            _entity_edge(t.id, "tail-1") for t in neighbors
        ]
        mock_storage.traces.get_traces_bulk.return_value = neighbors

        result = await mem.assess_situational(new)

        assert result is not None
        assert len(result.related_trace_ids) == 5
