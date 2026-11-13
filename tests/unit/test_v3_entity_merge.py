"""CognitiveMemory.merge_entities: argument and owner refusals, store outcome."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import EntityNotFoundError, MemorySDKError
from recollect.models import KnownEntity

SOURCE = KnownEntity(id="e-src", user_id="u1", names=["auto-injector"])
TARGET = KnownEntity(id="e-into", user_id="u1", names=["epinephrine auto-injector"])
MERGED = TARGET.model_copy(
    update={"names": ["auto-injector", "epinephrine auto-injector"], "trace_count": 2}
)


@pytest.fixture()
def mem(mock_storage: MagicMock, mock_embeddings: AsyncMock) -> CognitiveMemory:
    return CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)


def _registry(index: AsyncMock, *entities: KnownEntity | None) -> None:
    index.get_entity.side_effect = list(entities)


class TestMergeRefusals:
    @pytest.mark.parametrize(("source", "into"), [("", "e-into"), ("e-src", "")])
    async def test_rejects_empty_id(
        self, mem: CognitiveMemory, mock_entity_index: AsyncMock, source: str, into: str
    ) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await mem.merge_entities(source, into)
        mock_entity_index.merge_entities.assert_not_awaited()

    async def test_rejects_identical_ids(
        self, mem: CognitiveMemory, mock_entity_index: AsyncMock
    ) -> None:
        with pytest.raises(ValueError, match="itself"):
            await mem.merge_entities("e-src", "e-src")
        mock_entity_index.merge_entities.assert_not_awaited()

    @pytest.mark.parametrize(
        ("found", "missing"), [((None, TARGET), "e-src"), ((SOURCE, None), "e-into")]
    )
    async def test_missing_entity_raises_not_found(
        self,
        mem: CognitiveMemory,
        mock_entity_index: AsyncMock,
        found: tuple[KnownEntity | None, KnownEntity | None],
        missing: str,
    ) -> None:
        _registry(mock_entity_index, *found)
        with pytest.raises(EntityNotFoundError, match=missing):
            await mem.merge_entities("e-src", "e-into")
        mock_entity_index.merge_entities.assert_not_awaited()

    async def test_rejects_different_owners(
        self, mem: CognitiveMemory, mock_entity_index: AsyncMock
    ) -> None:
        other = SOURCE.model_copy(update={"user_id": None})
        _registry(mock_entity_index, other, TARGET)
        with pytest.raises(ValueError, match="owners"):
            await mem.merge_entities("e-src", "e-into")
        mock_entity_index.merge_entities.assert_not_awaited()


class TestMergeOutcome:
    async def test_returns_the_merged_target(
        self, mem: CognitiveMemory, mock_entity_index: AsyncMock
    ) -> None:
        _registry(mock_entity_index, SOURCE, TARGET, MERGED)
        assert await mem.merge_entities("e-src", "e-into") == MERGED
        mock_entity_index.merge_entities.assert_awaited_once_with("e-src", "e-into")

    async def test_store_refusal_raises_not_found(
        self, mem: CognitiveMemory, mock_entity_index: AsyncMock
    ) -> None:
        _registry(mock_entity_index, SOURCE, TARGET)
        mock_entity_index.merge_entities.return_value = False
        with pytest.raises(EntityNotFoundError):
            await mem.merge_entities("e-src", "e-into")

    async def test_target_gone_after_merge_raises_not_found(
        self, mem: CognitiveMemory, mock_entity_index: AsyncMock
    ) -> None:
        _registry(mock_entity_index, SOURCE, TARGET, None)
        with pytest.raises(EntityNotFoundError, match="e-into"):
            await mem.merge_entities("e-src", "e-into")

    def test_not_found_is_an_sdk_error(self) -> None:
        assert issubclass(EntityNotFoundError, MemorySDKError)
