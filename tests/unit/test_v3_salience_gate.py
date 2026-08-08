"""Salience and confidence gated by relevance: amplify, never fabricate.

Auxiliary signals (trace significance/valence, fact confidence) multiply by
the relevance the candidate already earned -- the entity-bonus precedent
extended to the last two un-gated terms in the scoring path.
"""

from __future__ import annotations

from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace, PersonaFact


def _fact(confidence: float) -> PersonaFact:
    return PersonaFact(
        subject="s",
        predicate="prefers",
        object="o",
        content="s prefers o",
        confidence=confidence,
        user_id="u1",
    )


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


class TestSalienceGate:
    """Salience boosts are gated by effective similarity."""

    def test_zero_relevance_gains_zero_from_salience(self) -> None:
        """Maximum salience cannot fabricate rank for an irrelevant trace."""
        t = _trace("loud-irrelevant", significance=1.0, valence=-1.0)
        traces = {t.id: t}
        scores = {t.id: 0.0}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.0,
            0.0,
            significance_weight=0.15,
            valence_weight=0.05,
        )
        assert result[0][1] == 0.0

    def test_salient_distractor_no_longer_outranks_relevant_neutral(self) -> None:
        """The recorded t3q07 order flip, reversed by the gate.

        Components from the story-8 inflation diagnosis: distractor base
        0.530 (sig 0.50, valence 0.70) overtook the tail at base 0.579
        (sig 0.35, valence 0.00) under flat boosts, 0.640 vs 0.632. Gated,
        amplification is bounded at 20% relative, and the higher-similarity
        neutral candidate wins: 0.588 vs 0.609.
        """
        distractor = _trace("emotional-distractor", significance=0.50, valence=0.70)
        tail = _trace("logistics-tail", significance=0.35, valence=0.00)
        traces = {distractor.id: distractor, tail.id: tail}
        scores = {distractor.id: 0.530, tail.id: 0.579}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.0,
            0.0,
            significance_weight=0.15,
            valence_weight=0.05,
        )
        result_map = {t.id: s for t, s in result}
        assert result_map[tail.id] > result_map[distractor.id]

    def test_salience_still_amplifies_among_equally_relevant(self) -> None:
        """At equal relevance the cognitive model survives: salient wins."""
        allergy = _trace("allergy", significance=0.95, valence=-0.30)
        lunch = _trace("lunch", significance=0.15, valence=0.10)
        traces = {allergy.id: allergy, lunch.id: lunch}
        scores = {allergy.id: 0.5, lunch.id: 0.5}

        result = CognitiveMemory._compute_fused_scores(
            traces,
            scores,
            {},
            {},
            0.0,
            0.0,
            significance_weight=0.15,
            valence_weight=0.05,
        )
        result_map = {t.id: s for t, s in result}
        # Gated separation at relevance 0.5: 0.5*((0.95-0.15)*0.15
        # + (0.3-0.1)*0.05) = 0.065
        assert result_map[allergy.id] - result_map[lunch.id] > 0.06


class TestConfidenceGate:
    """Fact confidence multiplies by similarity: sim * (0.7 + 0.3*conf)."""

    def test_low_sim_high_conf_fact_drops_below_trace_band(self) -> None:
        """The recorded slot-pollution case: an off-topic fact at
        confidence 0.95 and similarity 0.507 scored 0.640 flat -- inside
        the relevant-trace band. Gated it scores 0.499 and falls out.
        """
        off_topic = CognitiveMemory._compute_fact_relevance(_fact(0.95), 0.507)
        assert abs(off_topic - 0.507 * (0.7 + 0.3 * 0.95)) < 1e-9
        assert off_topic < 0.55

    def test_confidence_cannot_leapfrog_similarity(self) -> None:
        """Higher-similarity moderate-confidence beats low-sim high-conf."""
        on_topic = CognitiveMemory._compute_fact_relevance(_fact(0.60), 0.75)
        off_topic = CognitiveMemory._compute_fact_relevance(_fact(0.95), 0.507)
        assert on_topic > off_topic

    def test_embedding_less_fact_scores_zero_and_ranks_last(self) -> None:
        """No similarity measure -> no rank. The old 0.3*conf fallback was
        a non-monotonic seam: an embedding-less fact at 0.285 leapfrogged
        a barely-matched fact at 0.010. Uniform gating closes it.
        """
        unmatched = CognitiveMemory._compute_fact_relevance(_fact(0.95), 0.0)
        barely_matched = CognitiveMemory._compute_fact_relevance(_fact(0.50), 0.01)
        assert unmatched == 0.0
        assert barely_matched > unmatched

    def test_negative_similarity_clamps_to_zero(self) -> None:
        """Anti-correlated similarity never yields a negative relevance."""
        assert CognitiveMemory._compute_fact_relevance(_fact(0.95), -0.2) == 0.0
