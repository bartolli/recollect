"""Tests for entity/concept association creation in experience()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from recollect.core import CognitiveMemory
from recollect.llm.types import Entity, ExtractionResult


def _make_extractor(result: ExtractionResult) -> AsyncMock:
    x = AsyncMock()
    x.extract = AsyncMock(return_value=result)
    return x


class TestEntityAssociations:
    async def test_stores_trace_entities(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
    ) -> None:
        result = ExtractionResult(
            entities=[Entity(name="Google", entity_type="organization")],
            concepts=["search"],
        )
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=_make_extractor(result),
        )
        await mem.experience("Google is a search company")
        mock_entity_index.store_trace_entities.assert_awaited_once()
        entities_arg = mock_entity_index.store_trace_entities.call_args[0][1]
        assert len(entities_arg) == 1
        assert entities_arg[0].entity_name == "Google"

    async def test_stores_trace_concepts(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
    ) -> None:
        result = ExtractionResult(concepts=["AI", "robotics"])
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=_make_extractor(result),
        )
        await mem.experience("AI and robotics")
        mock_entity_index.store_trace_concepts.assert_awaited_once()
        concepts_arg = mock_entity_index.store_trace_concepts.call_args[0][1]
        assert len(concepts_arg) == 2

    async def test_creates_entity_associations_bidirectional(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
        mock_association_store: AsyncMock,
    ) -> None:
        mock_entity_index.get_traces_by_entity.return_value = ["existing-trace-1"]
        result = ExtractionResult(entities=[Entity(name="Sarah", entity_type="person")])
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=_make_extractor(result),
        )
        await mem.experience("Met Sarah today")

        # One association per pair (pair_key deduplicates direction)
        entity_calls = [
            call
            for call in mock_association_store.store_association.call_args_list
            if call[0][0].association_type == "entity"
        ]
        assert len(entity_calls) == 1

    async def test_no_self_association(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
        mock_association_store: AsyncMock,
    ) -> None:
        """Entity associations should not link a trace to itself."""
        result = ExtractionResult(entities=[Entity(name="Bob", entity_type="person")])
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=_make_extractor(result),
        )
        # Return empty so no existing traces share this entity
        mock_entity_index.get_traces_by_entity.return_value = []
        await mem.experience("Bob is here")
        entity_calls = [
            call
            for call in mock_association_store.store_association.call_args_list
            if call[0][0].association_type == "entity"
        ]
        assert len(entity_calls) == 0

    async def test_concept_associations_created(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
        mock_association_store: AsyncMock,
    ) -> None:
        mock_entity_index.get_traces_by_concept.return_value = ["other-trace"]
        result = ExtractionResult(concepts=["machine learning"])
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=_make_extractor(result),
        )
        await mem.experience("Working on machine learning")
        concept_calls = [
            call
            for call in mock_association_store.store_association.call_args_list
            if call[0][0].association_type == "concept"
        ]
        assert len(concept_calls) == 1  # one per pair

    async def test_storage_error_does_not_crash(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
    ) -> None:
        from recollect.exceptions import StorageError

        mock_entity_index.store_trace_entities.side_effect = StorageError("db down")
        result = ExtractionResult(entities=[Entity(name="X")])
        mem = CognitiveMemory(
            storage=mock_storage,
            embeddings=mock_embeddings,
            extractor=_make_extractor(result),
        )
        # Should not raise -- error is caught and logged
        trace = await mem.experience("Something about X")
        assert trace.content == "Something about X"
