"""Monotonic concept blend: weak tags never penalize.

effective = max(base, blend) on the trace path (_compute_fused_scores) and
the persona-fact path (_find_relevant_persona_facts). Blend wins iff
concept > base; extraction noise cannot drop a match below its untagged twin.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace, PersonaFact

_D = 768
_FK = {"subject": "A", "predicate": "likes", "object": "X", "category": "general"}


def _emb(s: float = 0.1) -> list[float]:
    return [s + i * 0.001 for i in range(_D)]


def _trace_score(base: float, concept: float | None) -> float:
    t = MemoryTrace(content="t", significance=0.0)
    result = CognitiveMemory._compute_fused_scores(
        {t.id: t},
        {t.id: base},
        {},
        {},
        0.0,
        0.0,
        significance_weight=0.0,
        valence_weight=0.0,
        concept_sims={} if concept is None else {t.id: concept},
        concept_weight=0.7,
    )
    return result[0][1]


class TestTraceBlendMonotonic:
    """Tagged trace never scores below its untagged twin at equal base sim."""

    def test_weak_tag_never_penalizes(self) -> None:
        # pre-fix: 0.7*0.1 + 0.3*0.6 = 0.25 displaced base 0.6
        assert _trace_score(0.6, 0.1) == pytest.approx(_trace_score(0.6, None))

    def test_strong_tag_still_blends(self) -> None:
        # blend wins iff concept > base: 0.7*0.8 + 0.3*0.5 = 0.71
        assert _trace_score(0.5, 0.8) == pytest.approx(0.71)

    def test_concept_equal_base_is_identity(self) -> None:
        assert _trace_score(0.5, 0.5) == pytest.approx(0.5)


class TestFactBlendMonotonic:
    """Fact path floors the blend at the bi-encoder score."""

    @pytest.mark.asyncio()
    @pytest.mark.parametrize(
        ("bi_sim", "concept_sim", "expected"),
        [(0.2, 0.7, 0.55), (0.9, 0.3, 0.9)],
        ids=["concept_promotes", "weak_concept_never_demotes"],
    )
    async def test_blend_floors_at_biencoder(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        bi_sim: float,
        concept_sim: float,
        expected: float,
    ) -> None:
        f = PersonaFact(**_FK, content="t", status="promoted", embedding=_emb())
        mock_storage.facts.search_facts_semantic = AsyncMock(return_value=[(f, bi_sim)])
        mock_storage.facts.get_persona_facts = AsyncMock(return_value=[f])
        mock_storage.concept_embeddings.get_max_sim_per_owner = AsyncMock(
            return_value={f.id: concept_sim}
        )
        m = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            config=MemoryConfig(),
        )
        facts, scores = await m._find_relevant_persona_facts("query", _emb())
        assert f.id in {x.id for x in facts}
        assert scores[f.id] == pytest.approx(expected)
