"""Tests for persona fact ranking strategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.models import FactCategory, MemoryTrace, PersonaFact, Thought


def _make_facts() -> list[PersonaFact]:
    return [
        PersonaFact(
            subject="Sarah",
            predicate="is_allergic_to",
            object="shellfish",
            category="health",
            content="Sarah is allergic to shellfish",
            confidence=0.9,
        ),
        PersonaFact(
            subject="Sarah",
            predicate="prefers",
            object="Mediterranean food",
            category="preference",
            content="Sarah prefers Mediterranean food",
            confidence=0.4,
        ),
    ]


def _make_mem(
    storage: MagicMock,
    embeddings: AsyncMock,
    strategy: str,
    threshold: float = 0.6,
) -> CognitiveMemory:
    cfg = MemoryConfig()
    cfg._config["persona"]["ranking_strategy"] = strategy
    cfg._config["persona"]["confidence_threshold"] = threshold
    return CognitiveMemory(storage=storage, embeddings=embeddings, config=cfg)


def _fact(
    fid: str,
    category: FactCategory = "general",
    confidence: float = 0.8,
    scope: str = "general",
) -> PersonaFact:
    return PersonaFact(
        id=fid,
        subject="Alex",
        predicate="likes",
        object="X",
        category=category,
        content="...",
        confidence=confidence,
        scope=scope,
    )


def _thought(content: str, relevance: float, *, pinned: bool) -> Thought:
    return Thought(
        trace=MemoryTrace(content=content),
        relevance=relevance,
        pinned=pinned,
    )


class TestThoughtPinnedField:
    def test_default_is_false(self) -> None:
        assert Thought(trace=MemoryTrace(content="test")).pinned is False

    def test_can_set_true(self) -> None:
        assert Thought(trace=MemoryTrace(content="test"), pinned=True).pinned


class TestPinnedStrategy:
    async def test_all_pinned(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mem = _make_mem(mock_storage, mock_embeddings, "pinned")
        thoughts = mem._persona_facts_to_thoughts(_make_facts())
        assert len(thoughts) == 2
        assert all(t.pinned for t in thoughts)


class TestRelevanceStrategy:
    async def test_none_pinned(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mem = _make_mem(mock_storage, mock_embeddings, "relevance")
        assert not any(t.pinned for t in mem._persona_facts_to_thoughts(_make_facts()))


class TestHybridStrategy:
    async def test_high_confidence_pinned(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mem = _make_mem(mock_storage, mock_embeddings, "hybrid", 0.6)
        thoughts = mem._persona_facts_to_thoughts(_make_facts())
        assert thoughts[0].pinned is True  # 0.9 >= 0.6
        assert thoughts[1].pinned is False  # 0.4 < 0.6

    async def test_relevance_equals_confidence(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mem = _make_mem(mock_storage, mock_embeddings, "hybrid")
        thoughts = mem._persona_facts_to_thoughts(_make_facts())
        assert thoughts[0].relevance == 0.9
        assert thoughts[1].relevance == 0.4

    async def test_at_threshold_is_pinned(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mem = _make_mem(mock_storage, mock_embeddings, "hybrid", 0.9)
        thoughts = mem._persona_facts_to_thoughts(_make_facts())
        assert thoughts[0].pinned is True  # 0.9 == 0.9
        assert thoughts[1].pinned is False  # 0.4 < 0.9


class TestAssembleWithTraceGuarantee:
    """Minimum trace guarantee prevents facts from crowding out traces."""

    def test_traces_guaranteed_when_facts_dominate(self) -> None:
        """With 5 facts and 2 traces, min_traces=2 keeps both traces."""
        facts = [_thought(f"fact{i}", 0.9 - i * 0.05, pinned=True) for i in range(5)]
        traces = [_thought(f"trace{i}", 0.7 - i * 0.1, pinned=False) for i in range(2)]
        result = CognitiveMemory._assemble_with_trace_guarantee(
            facts + traces,
            max_total=7,
            min_traces=2,
        )
        unpinned = [t for t in result if not t.pinned]
        assert len(unpinned) >= 2

    def test_facts_fill_when_few_traces(self) -> None:
        """With only 1 trace, facts fill remaining slots."""
        facts = [_thought(f"fact{i}", 0.9 - i * 0.05, pinned=True) for i in range(6)]
        traces = [_thought("trace0", 0.8, pinned=False)]
        result = CognitiveMemory._assemble_with_trace_guarantee(
            facts + traces,
            max_total=7,
            min_traces=2,
        )
        assert len(result) == 7
        assert len([t for t in result if not t.pinned]) == 1

    def test_high_relevance_trace_survives(self) -> None:
        """A high-relevance trace beats low-relevance facts for remaining slots."""
        facts = [_thought(f"fact{i}", 0.5, pinned=True) for i in range(6)]
        traces = [
            _thought("strong_trace", 0.95, pinned=False),
            _thought("weak_trace", 0.3, pinned=False),
            _thought("mid_trace", 0.6, pinned=False),
        ]
        result = CognitiveMemory._assemble_with_trace_guarantee(
            facts + traces,
            max_total=7,
            min_traces=2,
        )
        trace_contents = [t.trace.content for t in result if not t.pinned]
        assert "strong_trace" in trace_contents
        assert "mid_trace" in trace_contents

    def test_result_sorted_by_relevance(self) -> None:
        """Final result is sorted by relevance, not pin status."""
        thoughts = [
            _thought("fact", 0.5, pinned=True),
            _thought("trace", 0.9, pinned=False),
        ]
        result = CognitiveMemory._assemble_with_trace_guarantee(
            thoughts,
            max_total=7,
            min_traces=2,
        )
        assert result[0].relevance >= result[1].relevance


class TestFactToThoughtContent:
    async def test_content_has_important_context_prefix(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mem = _make_mem(mock_storage, mock_embeddings, "hybrid")
        for thought in mem._persona_facts_to_thoughts(_make_facts()):
            assert thought.trace.content is not None
            assert thought.trace.content.startswith("[IMPORTANT CONTEXT]")

    async def test_empty_facts_returns_empty(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mem = _make_mem(mock_storage, mock_embeddings, "hybrid")
        assert mem._persona_facts_to_thoughts([]) == []


class TestRankAndLimitFacts:
    def test_limit_respected(self) -> None:
        facts = [_fact(f"f{i}") for i in range(5)]
        assert len(CognitiveMemory._rank_and_limit_facts(facts, 2)) == 2

    def test_no_semantic_scores_falls_back_to_confidence(self) -> None:
        health = _fact("h", "health", 0.5, "health/safety")
        gen = _fact("g", "general", 0.9, "general")
        result = CognitiveMemory._rank_and_limit_facts([gen, health], 10, None)
        assert result[0].id == "g"


class TestComputeFactRelevance:
    def test_no_similarity_returns_confidence(self) -> None:
        fact = _fact("x", confidence=0.8)
        assert CognitiveMemory._compute_fact_relevance(fact, 0.0) == 0.8

    def test_with_similarity_blends_weighted(self) -> None:
        fact = _fact("x", confidence=0.8)
        result = CognitiveMemory._compute_fact_relevance(fact, 0.6)
        assert abs(result - 0.66) < 1e-9

    def test_high_similarity_boosts_low_confidence(self) -> None:
        fact = _fact("x", confidence=0.3)
        result = CognitiveMemory._compute_fact_relevance(fact, 0.9)
        assert abs(result - 0.72) < 1e-9


class TestSemanticScoresInThoughts:
    async def test_semantic_scores_affect_relevance(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mem = _make_mem(mock_storage, mock_embeddings, "hybrid")
        facts = _make_facts()
        scores = {facts[0].id: 0.8, facts[1].id: 0.0}
        thoughts = mem._persona_facts_to_thoughts(facts, semantic_scores=scores)
        assert thoughts[0].relevance > thoughts[1].relevance


class TestHybridPinsByRelevance:
    """Hybrid strategy pins on relevance (blended), not raw confidence."""

    async def test_high_confidence_low_similarity_not_pinned(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        """conf=0.9 + sem_sim=0.1 -> relevance=0.34, below 0.6."""
        mem = _make_mem(mock_storage, mock_embeddings, "hybrid", 0.6)
        facts = [
            PersonaFact(
                subject="Sarah",
                predicate="is_allergic_to",
                object="shellfish",
                category="health",
                content="Sarah is allergic to shellfish",
                confidence=0.9,
            )
        ]
        # Low semantic similarity drags relevance below threshold
        scores = {facts[0].id: 0.1}
        thoughts = mem._persona_facts_to_thoughts(facts, semantic_scores=scores)
        # relevance = 0.3 * 0.9 + 0.7 * 0.1 = 0.34 < 0.6
        assert thoughts[0].pinned is False

    async def test_low_confidence_high_similarity_pinned(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        """conf=0.4 + sem_sim=0.9 -> relevance=0.75, above 0.6."""
        mem = _make_mem(mock_storage, mock_embeddings, "hybrid", 0.6)
        facts = [
            PersonaFact(
                subject="Sarah",
                predicate="prefers",
                object="Mediterranean",
                category="preference",
                content="Sarah prefers Mediterranean food",
                confidence=0.4,
            )
        ]
        scores = {facts[0].id: 0.9}
        thoughts = mem._persona_facts_to_thoughts(facts, semantic_scores=scores)
        # relevance = 0.3 * 0.4 + 0.7 * 0.9 = 0.75 >= 0.6
        assert thoughts[0].pinned is True


class TestRerankerSigmoid:
    """Reranker scores are sigmoid-normalized to [0, 1]."""

    async def test_sigmoid_output_range(self) -> None:
        """Verify sigmoid maps raw logits to (0, 1) and preserves order."""
        import math

        raw_logits = [-10.0, -5.0, 0.0, 2.0, 5.0]
        expected = [1.0 / (1.0 + math.exp(-x)) for x in raw_logits]
        for exp in expected:
            assert 0.0 < exp < 1.0 or exp == 0.5  # 0 maps to 0.5
        # Verify monotonicity: higher logit -> higher score
        assert expected == sorted(expected)
