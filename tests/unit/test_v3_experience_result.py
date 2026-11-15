"""experience_result: the trace plus what the write did beside it.

A derived persona fact whose store write fails is reported, not only
logged; the trace stays stored. experience() returns the trace alone.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import EmbeddingError, StorageError
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import MemoryTrace
from recollect_mcp.server import AppContext, remember


def _relation(obj: str) -> Relation:
    return Relation(
        source="user",
        relation="prefers",
        target=obj,
        confidence=0.9,
        category="preference",
        context=f"The user said they prefer {obj}.",
    )


@pytest.fixture()
def mem(mock_storage: MagicMock, mock_embeddings: AsyncMock) -> CognitiveMemory:
    extractor = AsyncMock()
    extractor.extract = AsyncMock(
        return_value=ExtractionResult(
            fact_type="semantic",
            relations=[_relation("green tea"), _relation("oat milk")],
            significance=0.5,
        )
    )
    return CognitiveMemory(
        storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
    )


class TestUnstoredFacts:
    async def test_failed_fact_write_is_listed_and_the_trace_stored(
        self, mem: CognitiveMemory, mock_storage: MagicMock
    ) -> None:
        mock_storage.facts.store_persona_fact.side_effect = [
            StorageError("down"),
            "fact-id",
        ]
        result = await mem.experience_result(
            "I prefer green tea with oat milk", user_id="u1"
        )
        mock_storage.traces.store_trace.assert_awaited_once_with(result.trace)
        assert [f.object for f in result.unstored_facts] == ["green tea"]
        # The loop continues past the failure: the next relation is stored.
        assert mock_storage.facts.store_persona_fact.await_count == 2

    async def test_a_failed_dedup_read_is_listed(
        self, mem: CognitiveMemory, mock_storage: MagicMock
    ) -> None:
        mock_storage.facts.get_persona_facts.side_effect = StorageError("down")
        result = await mem.experience_result("I prefer green tea", user_id="u1")
        assert [f.object for f in result.unstored_facts] == ["green tea", "oat milk"]

    async def test_clean_write_lists_nothing(self, mem: CognitiveMemory) -> None:
        result = await mem.experience_result("I prefer green tea", user_id="u1")
        assert result.unstored_facts == []
        assert result.superseded == []


class TestExperienceDelegates:
    async def test_experience_returns_the_stored_trace(
        self, mem: CognitiveMemory, mock_storage: MagicMock
    ) -> None:
        mock_storage.facts.store_persona_fact.side_effect = StorageError("down")
        trace = await mem.experience("I prefer green tea", user_id="u1")
        assert isinstance(trace, MemoryTrace)
        mock_storage.traces.store_trace.assert_awaited_once_with(trace)


class TestDisplacementDecayFailure:
    async def test_remember_confirms_a_full_buffer_write(
        self, mem: CognitiveMemory, mock_storage: MagicMock
    ) -> None:
        for n in range(9):  # past any working-memory capacity
            await mem.experience(f"note {n}", user_id="u1")
        mock_storage.traces.apply_strength_factor.side_effect = StorageError("down")
        derived = mock_storage.facts.store_persona_fact.await_count
        ctx = MagicMock()
        ctx.request_context.lifespan_context = AppContext(
            memory=mem, worker=MagicMock(), user_id="u1"
        )
        ctx.headers = None

        reply = await remember("one more note", ctx)

        # The trace is stored: the missed decay must not read as a failed write.
        assert reply.startswith("Remembered [")
        assert "Storage failed" not in reply
        mock_storage.traces.apply_strength_factor.assert_awaited()
        assert mock_storage.facts.store_persona_fact.await_count > derived


class TestPostStoreEmbeddingFailure:
    @pytest.fixture()
    def tagged(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> CognitiveMemory:
        relation = _relation("green tea").model_copy(
            update={"context_tags": ["morning"]}
        )
        extractor = AsyncMock()
        extractor.extract = AsyncMock(
            return_value=ExtractionResult(
                fact_type="semantic",
                concepts=["tea"],
                relations=[relation],
                significance=0.5,
            )
        )
        return CognitiveMemory(
            storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
        )

    async def test_a_failed_batch_embedding_leaves_the_write_standing(
        self,
        tagged: CognitiveMemory,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        mock_embeddings.generate_embeddings_batch.side_effect = EmbeddingError("onnx")
        result = await tagged.experience_result("I prefer green tea", user_id="u1")
        mock_storage.traces.store_trace.assert_awaited_once_with(result.trace)
        # The fact row is written; only its tag embeddings are missing.
        assert result.unstored_facts == []
        mock_storage.facts.store_persona_fact.assert_awaited_once()

    async def test_a_failed_fact_embedding_stores_the_fact_without_a_vector(
        self,
        tagged: CognitiveMemory,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        vector = mock_embeddings.generate_embedding.return_value

        async def embed(text: str, **_: object) -> list[float]:
            if text == "I prefer green tea":
                return list(vector)
            raise EmbeddingError("onnx")

        mock_embeddings.generate_embedding.side_effect = embed
        result = await tagged.experience_result("I prefer green tea", user_id="u1")
        assert result.unstored_facts == []
        (stored,) = mock_storage.facts.store_persona_fact.await_args_list
        assert stored.args[0].embedding is None
