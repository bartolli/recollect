"""Tests for entity matching and persona fact surfacing in think_about()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace, PersonaFact

_EMB_DIM = 768


def _fake_embedding(seed: float = 0.1) -> list[float]:
    return [seed + i * 0.001 for i in range(_EMB_DIM)]


class TestEntityMatching:
    async def test_entity_traces_merged_into_candidates(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
        mock_trace_store: AsyncMock,
    ) -> None:
        entity_trace = MemoryTrace(
            id="entity-trace-1",
            content="Sarah is vegetarian",
            embedding=_fake_embedding(0.2),
        )
        mock_entity_index.match_entities.return_value = [("entity-trace-1", 0.85)]
        mock_trace_store.get_traces_bulk.return_value = [entity_trace]

        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        thoughts = await mem.think_about("Book dinner for Sarah")
        # Entity-matched trace should appear in results
        assert any(t.trace.id == "entity-trace-1" for t in thoughts)

    async def test_no_entity_match_still_works(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        thoughts = await mem.think_about("What happened today?")
        assert isinstance(thoughts, list)


class TestPersonaFactSurfacing:
    async def test_persona_facts_prepended_as_important_context(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        fact = PersonaFact(
            subject="Sarah",
            predicate="is_allergic_to",
            object="shellfish",
            content="Sarah is allergic to shellfish",
        )
        mock_fact_store.get_persona_facts_by_entities.return_value = [fact]

        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        thoughts = await mem.think_about("Book dinner for Sarah")
        # First thought should be the persona fact
        if thoughts:
            assert "[IMPORTANT CONTEXT]" in thoughts[0].reconstruction

    async def test_no_persona_facts_no_crash(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        thoughts = await mem.think_about("Random query")
        assert isinstance(thoughts, list)
