"""Tests for fused score computation with significance and valence boosting."""

from __future__ import annotations

from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace


def _trace(
    trace_id: str,
    significance: float = 0.5,
    valence: float = 0.0,
) -> MemoryTrace:
    return MemoryTrace(
        content=f"trace-{trace_id}",
        significance=significance,
        emotional_valence=valence,
    )


class TestSignificanceBoost:
    """High-significance traces rank above low-significance at equal cosine."""

    def test_high_significance_scores_higher(self) -> None:
        high = _trace("high", significance=0.9)
        low = _trace("low", significance=0.1)
        traces = {high.id: high, low.id: low}
        scores = {high.id: 0.5, low.id: 0.5}  # Equal cosine similarity

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.1,
            0.1,
            significance_weight=0.15,
            valence_weight=0.05,
        )
        result_map = {t.id: s for t, s in result}
        assert result_map[high.id] > result_map[low.id]

    def test_significance_boost_magnitude(self) -> None:
        """significance=0.95 at weight=0.15 adds ~0.14."""
        t = _trace("allergy", significance=0.95)
        traces = {t.id: t}
        scores = {t.id: 0.5}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.0,
            0.0,
            significance_weight=0.15,
            valence_weight=0.0,
        )
        fused = result[0][1]
        expected = 0.5 + 0.95 * 0.15  # 0.6425
        assert abs(fused - expected) < 1e-9

    def test_zero_significance_no_boost(self) -> None:
        t = _trace("trivial", significance=0.0)
        traces = {t.id: t}
        scores = {t.id: 0.4}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.0,
            0.0,
            significance_weight=0.15,
            valence_weight=0.0,
        )
        assert abs(result[0][1] - 0.4) < 1e-9


class TestValenceBoost:
    """Emotionally intense memories get a small additional boost."""

    def test_strong_emotion_scores_higher(self) -> None:
        emotional = _trace("fear", valence=-0.85)
        neutral = _trace("neutral", valence=0.0)
        traces = {emotional.id: emotional, neutral.id: neutral}
        scores = {emotional.id: 0.5, neutral.id: 0.5}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.0,
            0.0,
            significance_weight=0.0,
            valence_weight=0.05,
        )
        result_map = {t.id: s for t, s in result}
        assert result_map[emotional.id] > result_map[neutral.id]

    def test_valence_boost_uses_absolute_value(self) -> None:
        """Negative and positive valence of same magnitude get equal boost."""
        pos = _trace("joy", valence=0.8)
        neg = _trace("anger", valence=-0.8)
        traces = {pos.id: pos, neg.id: neg}
        scores = {pos.id: 0.5, neg.id: 0.5}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.0,
            0.0,
            significance_weight=0.0,
            valence_weight=0.05,
        )
        result_map = {t.id: s for t, s in result}
        assert abs(result_map[pos.id] - result_map[neg.id]) < 1e-9


class TestCombinedBoosts:
    """Significance + valence + existing bonuses all contribute."""

    def test_allergy_outranks_lunch(self) -> None:
        """Allergy (sig=0.95, val=-0.3) outranks lunch (sig=0.15, val=0.1)."""
        allergy = _trace("allergy", significance=0.95, valence=-0.3)
        lunch = _trace("lunch", significance=0.15, valence=0.1)
        traces = {allergy.id: allergy, lunch.id: lunch}
        scores = {allergy.id: 0.5, lunch.id: 0.5}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.1,
            0.1,
            significance_weight=0.15,
            valence_weight=0.05,
        )
        result_map = {t.id: s for t, s in result}
        diff = result_map[allergy.id] - result_map[lunch.id]
        # Significance diff: (0.95-0.15)*0.15 = 0.12
        # Valence diff: (0.3-0.1)*0.05 = 0.01
        assert diff > 0.1  # Meaningful separation

    def test_fused_score_clamped_to_one(self) -> None:
        """Even with all boosts, score never exceeds 1.0."""
        t = _trace("max", significance=1.0, valence=-1.0)
        traces = {t.id: t}
        scores = {t.id: 0.95}
        activation = {t.id: 1.0}
        entities = {t.id: 1.0}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            activation,
            entities,
            0.1,
            0.1,
            significance_weight=0.15,
            valence_weight=0.05,
        )
        assert result[0][1] == 1.0

    def test_default_weights_used_when_not_specified(self) -> None:
        """Without explicit weights, defaults (0.15, 0.05) are used."""
        t = _trace("default", significance=1.0, valence=-1.0)
        traces = {t.id: t}
        scores = {t.id: 0.5}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.0,
            0.0,
        )
        expected = 0.5 + 1.0 * 0.15 + 1.0 * 0.05  # 0.70
        assert abs(result[0][1] - expected) < 1e-9


class TestConceptGatedEntityBonus:
    """Entity bonus is multiplied by concept similarity."""

    def test_entity_bonus_gated_by_concept_sim(self) -> None:
        """Entity bonus only fires when concept sim > 0."""
        t1 = _trace("relevant", significance=0.5)
        t2 = _trace("irrelevant", significance=0.5)
        traces = {t1.id: t1, t2.id: t2}
        scores = {t1.id: 0.5, t2.id: 0.5}
        entity_sims = {t1.id: 1.0, t2.id: 1.0}
        concept_sims = {t1.id: 0.8}  # t2 has no concept match

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            entity_sims,
            0.0,
            0.1,
            significance_weight=0.0,
            valence_weight=0.0,
            concept_sims=concept_sims,
            concept_weight=0.7,
        )
        result_map = {t.id: s for t, s in result}
        # t1: blend(0.7*0.8+0.3*0.5=0.71) + entity(1.0*0.1*0.5*0.8=0.04) = 0.75
        # t2: base(0.5) + entity(gated to 0) = 0.5
        assert abs(result_map[t1.id] - 0.75) < 1e-9
        assert abs(result_map[t2.id] - 0.50) < 1e-9

    def test_concept_attention_without_entity(self) -> None:
        """Concept attention adds to score even without entity match."""
        t = _trace("pure_concept", significance=0.5)
        traces = {t.id: t}
        scores = {t.id: 0.4}
        concept_sims = {t.id: 0.6}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.0,
            0.1,
            significance_weight=0.0,
            valence_weight=0.0,
            concept_sims=concept_sims,
            concept_weight=0.7,
        )
        # blend: 0.7*0.6 + 0.3*0.4 = 0.54
        assert abs(result[0][1] - 0.54) < 1e-9
