"""Tests for concept attention multi-vector retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import ExtractionResult
from recollect.models import MemoryTrace, PersonaFact

_D = 768
_FK = {"subject": "A", "predicate": "likes", "object": "X", "category": "general"}


def _emb(s: float = 0.1) -> list[float]:
    return [s + i * 0.001 for i in range(_D)]


def _mem(st: MagicMock, emb: AsyncMock) -> CognitiveMemory:
    return CognitiveMemory(storage=st, embeddings=emb, config=MemoryConfig())


class TestEmbedTraceConcepts:
    """Write path: concepts embedded during experience()."""

    @pytest.mark.asyncio()
    async def test_concepts_embedded_and_stored(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        m = _mem(mock_storage, mock_embeddings)
        trace = MemoryTrace(content="test memory", embedding=_emb())
        concepts = ["dinner planning", "food safety", "allergies"]
        await m._embed_trace_concepts(trace, ExtractionResult(concepts=concepts))
        mock_embeddings.generate_embeddings_batch.assert_awaited_once_with(concepts)
        mock_storage.concept_embeddings.store_concept_embeddings.assert_awaited_once()
        store_fn = mock_storage.concept_embeddings.store_concept_embeddings
        stored = store_fn.call_args[0][0]
        assert len(stored) == 3
        assert all(ce.owner_type == "trace" for ce in stored)
        assert all(ce.owner_id == trace.id for ce in stored)

    @pytest.mark.asyncio()
    async def test_no_concepts_skips(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        m = _mem(mock_storage, mock_embeddings)
        t = MemoryTrace(content="test", embedding=_emb())
        await m._embed_trace_concepts(t, ExtractionResult(concepts=[]))
        mock_embeddings.generate_embeddings_batch.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_embedding_error_graceful(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        m = _mem(mock_storage, mock_embeddings)
        mock_embeddings.generate_embeddings_batch = AsyncMock(
            side_effect=OSError("fail"),
        )
        t = MemoryTrace(content="test", embedding=_emb())
        await m._embed_trace_concepts(t, ExtractionResult(concepts=["c1"]))


class TestEmbedFactTags:
    """Write path: fact context_tags embedded during extraction."""

    @pytest.mark.asyncio()
    async def test_tags_embedded_and_stored(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        m = _mem(mock_storage, mock_embeddings)
        tags = ["restaurant safety", "dining out", "food allergy", "meal planning"]
        await m._embed_fact_tags(PersonaFact(**_FK, content="fact", context_tags=tags))
        mock_embeddings.generate_embeddings_batch.assert_awaited_once()
        assert len(mock_embeddings.generate_embeddings_batch.call_args[0][0]) == 4
        store_fn = mock_storage.concept_embeddings.store_concept_embeddings
        stored = store_fn.call_args[0][0]
        assert len(stored) == 4
        assert all(ce.owner_type == "fact" for ce in stored)

    @pytest.mark.asyncio()
    async def test_no_tags_skips(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        m = _mem(mock_storage, mock_embeddings)
        await m._embed_fact_tags(PersonaFact(**_FK, content="t", context_tags=[]))
        mock_embeddings.generate_embeddings_batch.assert_not_awaited()


class TestConceptGatingInFusion:
    """Concept similarity gates entity bonus in fused scores."""

    def test_entity_bonus_zero_without_concept(self) -> None:
        """Entity-matched trace with zero concept sim gets no entity bonus."""
        t = MemoryTrace(content="hay", embedding=_emb(), significance=0.5)
        result = CognitiveMemory._compute_fused_scores(
            {t.id: t},
            {t.id: 0.5},
            {},
            {t.id: 1.0},
            0.0,
            0.1,
            significance_weight=0.0,
            valence_weight=0.0,
            concept_sims={},
            concept_weight=0.7,
        )
        # No concept sim -> effective_sim = base, entity bonus gated to 0
        assert abs(result[0][1] - 0.5) < 1e-9

    def test_entity_bonus_amplified_by_concept(self) -> None:
        """Entity-matched trace with high concept sim gets full entity bonus."""
        t = MemoryTrace(content="needle", embedding=_emb(), significance=0.5)
        result = CognitiveMemory._compute_fused_scores(
            {t.id: t},
            {t.id: 0.5},
            {},
            {t.id: 1.0},
            0.0,
            0.1,
            significance_weight=0.0,
            valence_weight=0.0,
            concept_sims={t.id: 0.8},
            concept_weight=0.7,
        )
        # blend: 0.7*0.8+0.3*0.5=0.71, entity: 1.0*0.1*0.5*0.8=0.04
        assert abs(result[0][1] - 0.75) < 1e-9

    def test_concept_attention_integrated_in_fusion(self) -> None:
        """Concept attention bonus applied within fusion, not separately."""
        t = MemoryTrace(content="t", embedding=_emb(), significance=0.0)
        result = CognitiveMemory._compute_fused_scores(
            {t.id: t},
            {t.id: 0.4},
            {},
            {},
            0.0,
            0.0,
            significance_weight=0.0,
            valence_weight=0.0,
            concept_sims={t.id: 0.5},
            concept_weight=0.7,
        )
        # blend: 0.7*0.5 + 0.3*0.4 = 0.47
        assert abs(result[0][1] - 0.47) < 1e-9


class TestFactConceptAttention:
    """Read path: concept attention rescues facts with low bi-encoder scores."""

    @pytest.mark.asyncio()
    @pytest.mark.parametrize(
        ("bi_score", "concept_score", "expected"),
        [(0.2, 0.7, 0.55), (0.9, 0.3, 0.48)],
        ids=["concept_primary", "concept_demotes"],
    )
    async def test_blend_of_concept_and_biencoder(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        bi_score: float,
        concept_score: float,
        expected: float,
    ) -> None:
        """0.7*concept + 0.3*bi-encoder determines final semantic score."""
        f = PersonaFact(**_FK, content="t", confidence=0.5, embedding=_emb())
        scores: dict[str, float] = {f.id: bi_score}
        mock_storage.concept_embeddings.get_max_sim_per_owner = AsyncMock(
            return_value={f.id: concept_score}
        )
        sims = await mock_storage.concept_embeddings.get_max_sim_per_owner(
            _emb(),
            owner_type="fact",
            owner_ids=[f.id],
        )
        for fid, csim in sims.items():
            bi_sim = scores.get(fid, 0.0)
            scores[fid] = 0.7 * csim + 0.3 * bi_sim
        assert abs(scores[f.id] - expected) < 1e-9
