"""Write-time surfacing: non-safety relevance floor + safety/non-safety union (1.3)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.models import FactCategory, FactStatus, MemoryTrace, PersonaFact

_EMB = [0.1] * 768


def _fact(
    category: FactCategory,
    predicate: str,
    obj: str,
    status: FactStatus = "promoted",
) -> PersonaFact:
    return PersonaFact(
        subject="Alex",
        predicate=predicate,
        object=obj,
        content=f"Alex {predicate} {obj}",
        category=category,
        status=status,
    )


def _pref(obj: str = "window seats") -> PersonaFact:
    return _fact("preference", "prefers", obj)


def _health() -> PersonaFact:
    return _fact("health", "is_allergic_to", "peanuts", status="pinned")


def _emb_trace(domains: list[str] | None = None) -> MemoryTrace:
    return MemoryTrace(
        content="booking a flight", embedding=_EMB, pattern={"domains": domains or []}
    )


class TestNonSafetySurfacing:
    async def test_surfaces_above_floor(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        pref = _pref()
        mock_fact_store.search_facts_semantic.return_value = [(pref, 0.9)]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        surfaced = await mem.surface_relevant_facts(_emb_trace())
        assert [f.id for f in surfaced] == [pref.id]

    async def test_silent_below_floor(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # Unmatched promoted fact (no embedding) scores 0.0 -- below the floor.
        mock_fact_store.get_persona_facts.return_value = [_pref()]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        assert await mem.surface_relevant_facts(_emb_trace()) == []

    async def test_excludes_safety_category(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # A relevant health fact must NOT leak via the non-safety path; with no
        # safety domain on the trace it surfaces nowhere.
        mock_fact_store.search_facts_semantic.return_value = [(_health(), 0.9)]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        assert await mem.surface_relevant_facts(_emb_trace(domains=[])) == []

    async def test_union_lists_safety_first(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        health, pref = _health(), _pref()
        mock_fact_store.get_persona_facts.return_value = [health]
        mock_fact_store.search_facts_semantic.return_value = [(pref, 0.9)]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        surfaced = await mem.surface_relevant_facts(_emb_trace(domains=["food"]))
        assert [f.id for f in surfaced] == [health.id, pref.id]

    async def test_respects_max_surfaced_cap(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        p1, p2 = _pref("aisle"), _pref("window")
        mock_fact_store.search_facts_semantic.return_value = [(p1, 0.9), (p2, 0.8)]
        cfg = MemoryConfig()
        cfg._set("persona.max_surfaced_facts", 1)
        mem = CognitiveMemory(
            storage=mock_storage, embeddings=mock_embeddings, config=cfg
        )
        surfaced = await mem.surface_relevant_facts(_emb_trace())
        assert [f.id for f in surfaced] == [p1.id]

    async def test_query_embedded_as_search_query(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # nomic asymmetric retrieval: the non-safety path embeds the trace as a
        # query, not reuse its search_document write-side embedding.
        mock_fact_store.search_facts_semantic.return_value = [(_pref(), 0.9)]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        await mem.surface_relevant_facts(_emb_trace())
        tasks = [
            c.kwargs.get("task")
            for c in mock_embeddings.generate_embedding.call_args_list
        ]
        assert "search_query" in tasks
