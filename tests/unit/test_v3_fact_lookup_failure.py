"""Persona fact lookup failure: recall fails closed, write-time surfacing degrades."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.exceptions import StorageError
from recollect.models import MemoryTrace, PersonaFact

_D = 768


def _emb(s: float = 0.1) -> list[float]:
    return [s + i * 0.001 for i in range(_D)]


def _memory(storage: MagicMock, embeddings: AsyncMock) -> CognitiveMemory:
    return CognitiveMemory(
        storage=storage, embeddings=embeddings, config=MemoryConfig()
    )


class TestRecallFailsClosed:
    @pytest.mark.parametrize(
        "failure", [StorageError("down"), OSError("reset")], ids=["storage", "os"]
    )
    async def test_lookup_failure_raises_storage_error(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        failure: Exception,
    ) -> None:
        mock_storage.facts.search_facts_semantic = AsyncMock(side_effect=failure)
        memory = _memory(mock_storage, mock_embeddings)
        with pytest.raises(StorageError):
            await memory._find_relevant_persona_facts("query", _emb())

    async def test_think_about_raises_when_fact_lookup_fails(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        mock_storage.facts.search_facts_semantic = AsyncMock(
            side_effect=StorageError("down")
        )
        memory = _memory(mock_storage, mock_embeddings)
        with pytest.raises(StorageError):
            await memory.think_about("what is Mei allergic to")

    async def test_concept_lookup_failure_keeps_biencoder_ranking(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        fact = PersonaFact(
            subject="A",
            predicate="likes",
            object="X",
            category="general",
            content="t",
            status="promoted",
            embedding=_emb(),
        )
        mock_storage.facts.search_facts_semantic = AsyncMock(return_value=[(fact, 0.4)])
        mock_storage.facts.get_persona_facts = AsyncMock(return_value=[fact])
        mock_storage.concept_embeddings.get_max_sim_per_owner = AsyncMock(
            side_effect=StorageError("down")
        )
        memory = _memory(mock_storage, mock_embeddings)
        facts, scores = await memory._find_relevant_persona_facts("query", _emb())
        assert [f.id for f in facts] == [fact.id]
        assert scores[fact.id] == pytest.approx(0.4)


class TestWriteTimeSurfacingDegrades:
    async def test_non_safety_surfacing_is_empty_on_lookup_failure(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        # The trace is already stored: a failed lookup must not fail the write.
        mock_storage.facts.search_facts_semantic = AsyncMock(
            side_effect=StorageError("down")
        )
        memory = _memory(mock_storage, mock_embeddings)
        trace = MemoryTrace(content="x", pattern={"domains": ["finance"]})
        assert await memory.surface_relevant_facts(trace) == []
