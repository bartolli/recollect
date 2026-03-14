"""Tests for CognitiveMemory: experience, think_about, consolidate."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.datetime_utils import now_utc
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


class TestExperience:
    async def test_stores_trace(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        trace = await mem.experience("Hello world")
        assert trace.content == "Hello world"
        assert trace.strength == 0.3
        mock_trace_store.store_trace.assert_awaited_once()

    async def test_adds_to_working_memory(self, mem: CognitiveMemory) -> None:
        trace = await mem.experience("Hello")
        active = mem.active_traces()
        assert len(active) == 1
        assert active[0].id == trace.id

    async def test_displacement_decays_evicted(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        for i in range(7):
            await mem.experience(f"Memory {i}")
        mock_trace_store.update_trace_strength.reset_mock()
        await mem.experience("Overflow")
        mock_trace_store.update_trace_strength.assert_awaited()

    async def test_temporal_association(
        self, mem: CognitiveMemory, mock_association_store: AsyncMock
    ) -> None:
        await mem.experience("First")
        await mem.experience("Second")
        mock_association_store.store_association.assert_awaited()
        assoc = mock_association_store.store_association.call_args[0][0]
        assert assoc.association_type == "temporal"

    async def test_extracts_patterns(
        self, mem: CognitiveMemory, mock_extractor: AsyncMock
    ) -> None:
        trace = await mem.experience("Important meeting")
        mock_extractor.extract.assert_awaited_once_with("Important meeting")
        assert "concepts" in trace.pattern

    async def test_no_extractor_uses_empty_pattern(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        m = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        trace = await m.experience("No extractor")
        assert trace.pattern == {
            "concepts": [],
            "relations": [],
            "entities": [],
            "emotional_valence": 0.0,
            "significance": 0.1,
            "fact_type": "episodic",
        }


class TestThinkAbout:
    async def test_returns_thoughts(
        self, mem: CognitiveMemory, mock_vector_index: AsyncMock
    ) -> None:
        stored = MemoryTrace(content="Python is great", embedding=_fake_embedding())
        mock_vector_index.search_semantic.return_value = [(stored, 0.85)]
        thoughts = await mem.think_about("Python")
        assert len(thoughts) >= 1
        assert isinstance(thoughts[0], Thought)

    async def test_respects_token_budget(
        self, mem: CognitiveMemory, mock_vector_index: AsyncMock
    ) -> None:
        traces = [
            MemoryTrace(
                content="x" * 8000,
                embedding=_fake_embedding(0.1 + i * 0.01),
            )
            for i in range(5)
        ]
        mock_vector_index.search_semantic.return_value = [
            (t, 0.9 - i * 0.1) for i, t in enumerate(traces)
        ]
        thoughts = await mem.think_about("test", token_budget=1000)
        total = sum(t.token_count for t in thoughts)
        assert total <= 1000

    async def test_applies_retrieval_boost(
        self,
        mem: CognitiveMemory,
        mock_vector_index: AsyncMock,
        mock_trace_store: AsyncMock,
    ) -> None:
        stored = MemoryTrace(
            content="Python", embedding=_fake_embedding(), strength=0.5
        )
        mock_vector_index.search_semantic.return_value = [(stored, 0.85)]
        await mem.think_about("Python")
        mock_trace_store.mark_retrieved.assert_awaited()


class TestConsolidate:
    async def test_consolidates_strong(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        strong = MemoryTrace(content="Strong", strength=0.8)
        mock_trace_store.get_unconsolidated_traces.return_value = [strong]
        result = await mem.consolidate()
        assert result.consolidated == 1
        mock_trace_store.mark_consolidated.assert_awaited_once()

    async def test_forgets_old_weak(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        old = MemoryTrace(
            content="Weak",
            strength=0.05,
            created_at=now_utc() - timedelta(hours=48),
        )
        mock_trace_store.get_unconsolidated_traces.return_value = [old]
        result = await mem.consolidate()
        assert result.forgotten == 1
        mock_trace_store.delete_trace.assert_awaited_once()

    async def test_keeps_young_weak_pending(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        young = MemoryTrace(content="Young", strength=0.2)
        mock_trace_store.get_unconsolidated_traces.return_value = [young]
        result = await mem.consolidate()
        assert result.still_pending == 1
        mock_trace_store.delete_trace.assert_not_awaited()
