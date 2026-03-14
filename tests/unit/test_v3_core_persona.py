"""Tests for persona fact extraction and management."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from recollect.core import (
    _VALID_CATEGORIES,
    CognitiveMemory,
    _canonicalize_predicate,
    _find_contradicting_fact,
    _find_exact_duplicate,
    _should_fast_track,
)
from recollect.llm.types import Entity, ExtractionResult, Relation
from recollect.models import MemoryTrace, PersonaFact


def _fact(
    subj: str = "Sarah",
    pred: str = "works_at",
    obj: str = "Google",
    content: str = "c",
) -> PersonaFact:
    return PersonaFact(subject=subj, predicate=pred, object=obj, content=content)


class TestValidCategories:
    def test_known_categories_in_set(self) -> None:
        for cat in (
            "health",
            "dietary",
            "identity",
            "relationship",
            "preference",
            "schedule",
            "constraint",
            "general",
        ):
            assert cat in _VALID_CATEGORIES

    def test_unknown_category_not_in_set(self) -> None:
        assert "xyz_unknown" not in _VALID_CATEGORIES


class TestShouldFastTrack:
    def test_health_fast_tracks(self) -> None:
        assert _should_fast_track("health", 0.5) is True

    def test_dietary_fast_tracks(self) -> None:
        assert _should_fast_track("dietary", 0.5) is True

    def test_constraint_fast_tracks(self) -> None:
        assert _should_fast_track("constraint", 0.5) is True

    def test_preference_does_not_fast_track(self) -> None:
        assert _should_fast_track("preference", 0.99) is False

    def test_general_does_not_fast_track(self) -> None:
        assert _should_fast_track("general", 1.0) is False


class TestCanonicalizePredicate:
    def test_known_alias(self) -> None:
        assert _canonicalize_predicate("started_at") == "works_at"
        assert _canonicalize_predicate("employed_at") == "works_at"

    def test_unknown_passes_through(self) -> None:
        assert _canonicalize_predicate("custom_pred") == "custom_pred"


class TestFindContradictingFact:
    def test_finds_contradiction(self) -> None:
        result = _find_contradicting_fact([_fact(obj="Acme")], _fact(obj="Google"))
        assert result is not None
        assert result.object == "Acme"

    def test_no_contradiction_same_object(self) -> None:
        assert _find_contradicting_fact([_fact()], _fact()) is None

    def test_no_contradiction_different_predicate(self) -> None:
        assert (
            _find_contradicting_fact([_fact()], _fact(pred="lives_in", obj="NYC"))
            is None
        )

    def test_alias_matching(self) -> None:
        result = _find_contradicting_fact(
            [_fact(obj="Acme")], _fact(pred="started_at", obj="Google")
        )
        assert result is not None


class TestFindExactDuplicate:
    def test_finds_exact_duplicate(self) -> None:
        result = _find_exact_duplicate([_fact()], _fact())
        assert result is not None
        assert result.object == "Google"

    def test_no_duplicate_different_object(self) -> None:
        assert _find_exact_duplicate([_fact()], _fact(obj="Meta")) is None

    def test_no_duplicate_different_predicate(self) -> None:
        assert _find_exact_duplicate([_fact()], _fact(pred="lives_in")) is None

    def test_duplicate_alias_matching(self) -> None:
        # employed_at canonicalizes to works_at
        result = _find_exact_duplicate([_fact()], _fact(pred="employed_at"))
        assert result is not None


class TestPersonaFactExtraction:
    async def test_semantic_trace_creates_persona_fact(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        result = ExtractionResult(
            fact_type="semantic",
            entities=[Entity(name="Sarah")],
            relations=[
                Relation(source="Sarah", relation="is_allergic_to", target="shellfish")
            ],
            significance=0.9,
        )
        extractor = AsyncMock()
        extractor.extract = AsyncMock(return_value=result)
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=extractor,
        )
        await mem.experience("Sarah is allergic to shellfish")
        mock_fact_store.store_persona_fact.assert_awaited()

    async def test_duplicate_fact_not_reinserted(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        result = ExtractionResult(
            fact_type="semantic",
            entities=[Entity(name="Sarah")],
            relations=[Relation(source="Sarah", relation="works_at", target="Google")],
            significance=0.5,
        )
        extractor = AsyncMock()
        extractor.extract = AsyncMock(return_value=result)
        existing_fact = _fact(content="Sarah works at Google")
        mock_fact_store.get_persona_facts.return_value = [existing_fact]
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=extractor,
        )
        await mem.experience("Sarah works at Google")
        mock_fact_store.store_persona_fact.assert_not_awaited()

    async def test_episodic_trace_skips_persona_facts(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        result = ExtractionResult(
            fact_type="episodic",
            relations=[Relation(source="A", relation="met", target="B")],
        )
        extractor = AsyncMock()
        extractor.extract = AsyncMock(return_value=result)
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=extractor,
        )
        await mem.experience("Met someone today")
        mock_fact_store.store_persona_fact.assert_not_awaited()


class TestPinUnpinFacts:
    async def test_pin_creates_fact(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_trace_store: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        trace = MemoryTrace(content="Remember this", id="trace-1")
        mock_trace_store.get_trace.return_value = trace
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        fact = await mem.pin("trace-1")
        assert fact.content == "Remember this"
        assert fact.confidence == 1.0
        mock_fact_store.store_persona_fact.assert_awaited_once()

    async def test_unpin_demotes_to_promoted(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        result = await mem.unpin("fact-1")
        assert result is True
        mock_fact_store.update_fact_status.assert_awaited_once_with(
            "fact-1", "promoted"
        )

    async def test_facts_delegates_to_storage(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        await mem.facts("Sarah")
        mock_fact_store.get_persona_facts.assert_awaited_once_with("Sarah")
