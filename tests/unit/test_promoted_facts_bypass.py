"""Tests for promoted/pinned persona facts bypassing similarity threshold."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from recollect.core import CognitiveMemory
from recollect.models import PersonaFact

_EMB_DIM = 768


def _zero_embedding() -> list[float]:
    """Embedding orthogonal to the query embedding (all zeros -> cosine 0)."""
    return [0.0] * _EMB_DIM


def _promoted_allergy_fact() -> PersonaFact:
    return PersonaFact(
        id="fact-allergy",
        subject="Alex",
        predicate="is_allergic_to",
        object="peanuts",
        category="health",
        content="Alex has a severe peanut allergy",
        status="promoted",
        confidence=0.9,
        embedding=_zero_embedding(),
        context_tags=["allergy", "peanut", "health"],
    )


def _candidate_allergy_fact() -> PersonaFact:
    return PersonaFact(
        id="fact-candidate",
        subject="Alex",
        predicate="is_allergic_to",
        object="shellfish",
        category="health",
        content="Alex might be allergic to shellfish",
        status="candidate",
        confidence=0.4,
        embedding=_zero_embedding(),
        context_tags=["allergy", "shellfish"],
    )


class TestPromotedFactsBypass:
    """Promoted/pinned persona facts must surface regardless of similarity."""

    async def test_promoted_fact_surfaces_below_threshold(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
        mock_vector_index: AsyncMock,
        mock_entity_index: AsyncMock,
        mock_concept_embedding_store: AsyncMock,
        mock_recall_token_store: AsyncMock,
    ) -> None:
        """A promoted fact with sim < 0.3 must still appear in results."""
        promoted = _promoted_allergy_fact()

        # Semantic search returns the fact with similarity well below 0.3
        mock_fact_store.search_facts_semantic.return_value = [(promoted, 0.1)]

        # The fetch-all-promoted path also returns this fact
        mock_fact_store.get_persona_facts.return_value = [promoted]

        # Entity path returns nothing extra
        mock_fact_store.get_persona_facts_by_entities.return_value = []

        # Vector search returns nothing (no trace results)
        mock_vector_index.search_semantic.return_value = []

        # Entity matching returns nothing
        mock_entity_index.match_entities.return_value = []

        # Concept attention returns nothing
        mock_concept_embedding_store.get_max_sim_per_owner.return_value = {}

        # Recall tokens inactive
        mock_recall_token_store.get_activated_trace_ids.return_value = []
        mock_recall_token_store.get_tokens_for_traces.return_value = []

        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        thoughts = await mem.think_about("Friday dinner plans at Siam Kitchen")

        peanut_thoughts = [
            t for t in thoughts if "peanut" in t.reconstruction.lower()
        ]
        assert len(peanut_thoughts) >= 1, (
            "Promoted fact with sim=0.1 should bypass the 0.3 threshold"
        )

    async def test_candidate_fact_filtered_below_threshold(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
        mock_vector_index: AsyncMock,
        mock_entity_index: AsyncMock,
        mock_concept_embedding_store: AsyncMock,
        mock_recall_token_store: AsyncMock,
    ) -> None:
        """A candidate fact with sim < 0.3 must NOT appear in results."""
        promoted = _promoted_allergy_fact()
        candidate = _candidate_allergy_fact()

        # Semantic search returns both below threshold
        mock_fact_store.search_facts_semantic.return_value = [
            (promoted, 0.1),
            (candidate, 0.1),
        ]

        # Fetch-all returns both -- but candidate should be filtered
        mock_fact_store.get_persona_facts.return_value = [promoted, candidate]

        mock_fact_store.get_persona_facts_by_entities.return_value = []
        mock_vector_index.search_semantic.return_value = []
        mock_entity_index.match_entities.return_value = []
        mock_concept_embedding_store.get_max_sim_per_owner.return_value = {}
        mock_recall_token_store.get_activated_trace_ids.return_value = []
        mock_recall_token_store.get_tokens_for_traces.return_value = []

        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        thoughts = await mem.think_about("Friday dinner plans at Siam Kitchen")

        shellfish_thoughts = [
            t for t in thoughts if "shellfish" in t.reconstruction.lower()
        ]
        assert len(shellfish_thoughts) == 0, (
            "Candidate fact below threshold must not surface in results"
        )
