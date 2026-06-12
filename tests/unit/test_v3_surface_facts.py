"""Write-time persona-fact surfacing: contract + domain-gated safety (1.1, 1.2b)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from recollect.core import CognitiveMemory
from recollect.llm.types import ExtractionResult
from recollect.models import FactCategory, FactStatus, MemoryTrace, PersonaFact
from recollect_mcp.server import _format_remember_result


def _safety_fact(
    category: FactCategory = "health",
    obj: str = "peanuts",
    status: FactStatus = "pinned",
) -> PersonaFact:
    return PersonaFact(
        subject="Alex",
        predicate="is_allergic_to",
        object=obj,
        content=f"Alex is allergic to {obj}",
        category=category,
        status=status,
    )


def _trace(domains: list[str]) -> MemoryTrace:
    return MemoryTrace(content="x", pattern={"domains": domains})


class TestSurfaceRelevantFacts:
    async def test_safety_fact_surfaces_on_food_domain(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        fact = _safety_fact("health")
        mock_fact_store.get_persona_facts.return_value = [fact]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        surfaced = await mem.surface_relevant_facts(_trace(["food"]))
        assert [f.id for f in surfaced] == [fact.id]

    async def test_safety_fact_silent_on_unrelated_domain(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # finance maps to no safety categories -- not wallpaper.
        mock_fact_store.get_persona_facts.return_value = [_safety_fact("health")]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        assert await mem.surface_relevant_facts(_trace(["finance"])) == []

    async def test_mistag_generosity_travel_surfaces_dietary(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # travel -> {health, dietary}: the generous map absorbs a domain mis-tag.
        fact = _safety_fact("dietary")
        mock_fact_store.get_persona_facts.return_value = [fact]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        surfaced = await mem.surface_relevant_facts(_trace(["travel"]))
        assert [f.id for f in surfaced] == [fact.id]

    async def test_all_safety_categories_surface_on_safety_domain(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # Recall-maximal contract: a safety-relevant domain surfaces every safety
        # category, not a subset (exercise was health-only; constraint had none).
        mock_fact_store.get_persona_facts.return_value = [
            _safety_fact("health"),
            _safety_fact("dietary", obj="gluten"),
            _safety_fact("constraint", obj="no night shifts"),
        ]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        surfaced = await mem.surface_relevant_facts(_trace(["exercise"]))
        assert {f.category for f in surfaced} == {"health", "dietary", "constraint"}

    async def test_non_safety_fact_not_surfaced(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # preference is non-safety; the floor path arrives in 1.3, not 1.2b.
        pref = _safety_fact(category="preference")
        mock_fact_store.get_persona_facts.return_value = [pref]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        assert await mem.surface_relevant_facts(_trace(["food"])) == []

    async def test_candidate_safety_fact_not_surfaced(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mock_fact_store.get_persona_facts.return_value = [
            _safety_fact("health", status="candidate")
        ]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        assert await mem.surface_relevant_facts(_trace(["food"])) == []

    async def test_safety_silent_without_domain(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # No safety domain -> the safety gate stays shut even though a safety
        # fact exists (the non-safety path still runs but excludes safety cats).
        mock_fact_store.get_persona_facts.return_value = [_safety_fact("health")]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        assert await mem.surface_relevant_facts(_trace([])) == []

    async def test_is_read_only(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mock_fact_store.get_persona_facts.return_value = [_safety_fact("health")]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        await mem.surface_relevant_facts(_trace(["food"]))
        mock_fact_store.increment_mention_count.assert_not_awaited()
        mock_fact_store.update_fact_status.assert_not_awaited()
        mock_fact_store.store_persona_fact.assert_not_awaited()

    async def test_food_heavy_session_repeats_allergy(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # Surfacing is stateless: consecutive food writes re-surface the allergy.
        # No cross-turn suppression -- repetition tracks continued relevance.
        fact = _safety_fact("health")
        mock_fact_store.get_persona_facts.return_value = [fact]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        first = await mem.surface_relevant_facts(_trace(["food"]))
        second = await mem.surface_relevant_facts(_trace(["food"]))
        assert [f.id for f in first] == [fact.id]
        assert [f.id for f in second] == [fact.id]

    async def test_surface_does_not_touch_token_store(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
        mock_recall_token_store: AsyncMock,
    ) -> None:
        # Coupling guard: surfacing must stay off the recall-token assessment
        # path. (Does not prove the both-run-on-a-write split; that is structural.)
        mock_fact_store.get_persona_facts.return_value = [_safety_fact("health")]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        await mem.surface_relevant_facts(_trace(["food"]))
        mock_recall_token_store.find_token_by_traces.assert_not_awaited()
        mock_recall_token_store.create_token.assert_not_awaited()
        mock_recall_token_store.stamp_traces.assert_not_awaited()


class TestExperienceDomainLinkage:
    async def test_experience_persists_domains_under_read_key(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_trace_store: AsyncMock,
    ) -> None:
        # Producer->consumer contract: experience() must store domains under the
        # exact pattern key surface_relevant_facts reads. Guards string-key drift
        # that construction-correctness and the mocked tests cannot catch.
        extractor = AsyncMock()
        extractor.extract = AsyncMock(
            return_value=ExtractionResult(domains=["food"])
        )
        mem = CognitiveMemory(
            storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
        )
        await mem.experience("dinner plans", user_id="u1")
        stored = mock_trace_store.store_trace.call_args.args[0]
        assert stored.pattern["domains"] == ["food"]


class TestFormatRememberResult:
    def test_appends_context_block(self) -> None:
        out = _format_remember_result(MemoryTrace(content="dinner"), [_safety_fact()])
        assert "Remembered" in out
        assert "IMPORTANT CONTEXT:" in out
        assert "peanuts" in out

    def test_no_block_when_empty(self) -> None:
        trace = MemoryTrace(content="dinner")
        assert "IMPORTANT CONTEXT" not in _format_remember_result(trace, [])
        assert "IMPORTANT CONTEXT" not in _format_remember_result(trace)
