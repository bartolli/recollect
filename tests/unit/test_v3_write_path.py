"""Write-path integrity: experience() user_id gate.

persona_facts.user_id is NOT NULL (m002); a user-less write with an
extractor wired would silently drop every promoted relation. The gate
raises at the API boundary instead.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.exceptions import StorageError
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import PersonaFact


class TestUserIdGate:
    async def test_raises_without_user_id_when_auto_extract_on(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_extractor: AsyncMock,
    ) -> None:
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=mock_extractor,
        )
        with pytest.raises(ValueError, match="user_id"):
            await mem.experience("Sarah is allergic to shellfish")
        mock_storage.traces.store_trace.assert_not_awaited()

    async def test_stores_with_user_id(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_extractor: AsyncMock,
    ) -> None:
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=mock_extractor,
        )
        trace = await mem.experience("Sarah is allergic to shellfish", user_id="u1")
        assert trace.user_id == "u1"
        mock_storage.traces.store_trace.assert_awaited_once()

    async def test_auto_extract_off_allows_userless_write(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_extractor: AsyncMock,
    ) -> None:
        cfg = MemoryConfig()
        cfg._set("persona.auto_extract", False)
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=mock_extractor,
            config=cfg,
        )
        await mem.experience("Sarah is allergic to shellfish")
        mock_storage.traces.store_trace.assert_awaited_once()
        mock_storage.facts.store_persona_fact.assert_not_awaited()

    async def test_no_extractor_allows_userless_write(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        # No extractor => no persona facts can exist => nothing the gate
        # protects; non-LLM usage stays valid without a user_id.
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
        )
        await mem.experience("Sarah is allergic to shellfish")
        mock_storage.traces.store_trace.assert_awaited_once()


def _semantic_extractor() -> AsyncMock:
    """Extractor yielding one promotable tagged relation and no trace concepts.

    concepts=[] keeps _embed_trace_concepts silent, so any
    store_concept_embeddings call is attributable to fact tags.
    """
    extractor = AsyncMock()
    extractor.extract = AsyncMock(
        return_value=ExtractionResult(
            fact_type="semantic",
            concepts=[],
            relations=[
                Relation(
                    source="user",
                    relation="is_allergic_to",
                    target="shellfish",
                    confidence=0.9,
                    category="health",
                    context_tags=["shellfish allergy", "food safety"],
                )
            ],
            significance=0.9,
        )
    )
    return extractor


class TestFactTagEmbeddingGate:
    async def test_duplicate_mention_embeds_no_tags(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        existing = PersonaFact(
            subject="user",
            predicate="is_allergic_to",
            object="shellfish",
            content="user is_allergic_to shellfish",
        )
        mock_fact_store.get_persona_facts.return_value = [existing]
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=_semantic_extractor(),
        )
        await mem.experience("Sarah noted the shellfish allergy again", user_id="u1")
        mock_fact_store.increment_mention_count.assert_awaited_once()
        mock_storage.concept_embeddings.store_concept_embeddings.assert_not_awaited()

    async def test_new_fact_embeds_tags_for_stored_id(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=_semantic_extractor(),
        )
        await mem.experience("Sarah is allergic to shellfish", user_id="u1")
        mock_fact_store.store_persona_fact.assert_awaited_once()
        stored_fact = mock_fact_store.store_persona_fact.await_args.args[0]
        ce_store = mock_storage.concept_embeddings.store_concept_embeddings
        ce_store.assert_awaited_once()
        embedded = ce_store.await_args.args[0]
        assert {e.owner_id for e in embedded} == {stored_fact.id}
        assert {e.concept for e in embedded} == {"shellfish allergy", "food safety"}

    async def test_fact_store_failure_embeds_no_tags(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mock_fact_store.get_persona_facts.side_effect = StorageError("down")
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=_semantic_extractor(),
        )
        await mem.experience("Sarah is allergic to shellfish", user_id="u1")
        mock_storage.concept_embeddings.store_concept_embeddings.assert_not_awaited()

    async def test_superseding_fact_embeds_tags(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # lives_in is CURRENT (a new value supersedes); is_allergic_to is SET
        # (additive) -- adr-persona-fact-cardinality.
        contradicting = PersonaFact(
            subject="user",
            predicate="lives_in",
            object="Lisbon",
            content="user lives_in Lisbon",
        )
        mock_fact_store.get_persona_facts.return_value = [contradicting]
        extractor = AsyncMock()
        extractor.extract = AsyncMock(
            return_value=ExtractionResult(
                fact_type="semantic",
                concepts=[],
                relations=[
                    Relation(
                        source="user",
                        relation="lives_in",
                        target="Berlin",
                        confidence=0.9,
                        category="identity",
                        context_tags=["relocation", "berlin"],
                    )
                ],
                significance=0.9,
            )
        )
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=extractor,
        )
        await mem.experience("Correction: now lives in Berlin", user_id="u1")
        mock_fact_store.supersede_persona_fact.assert_awaited_once()
        superseding = mock_fact_store.supersede_persona_fact.await_args.args[1]
        ce_store = mock_storage.concept_embeddings.store_concept_embeddings
        ce_store.assert_awaited_once()
        assert {e.owner_id for e in ce_store.await_args.args[0]} == {superseding.id}
