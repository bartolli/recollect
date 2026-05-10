"""Tests for embedding generation.

Focus: contract verification, error wrapping.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from recollect.embeddings import FastEmbedProvider
from recollect.exceptions import EmbeddingError


class TestDimensions:
    def test_returns_configured_value(self) -> None:
        provider = FastEmbedProvider(dimensions=384)
        assert provider.dimensions == 384

    def test_default_is_768(self) -> None:
        provider = FastEmbedProvider()
        assert provider.dimensions == 768


class TestGenerateEmbedding:
    async def test_returns_float_list_of_correct_size(self) -> None:
        mock_model = MagicMock()
        fake_embedding = [float(i) for i in range(384)]
        mock_model.embed.return_value = iter([fake_embedding])

        provider = FastEmbedProvider(dimensions=384)
        provider._model = mock_model

        result = await provider.generate_embedding("test text")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    async def test_wraps_model_errors(self) -> None:
        mock_model = MagicMock()
        mock_model.embed.side_effect = RuntimeError("model crashed")

        provider = FastEmbedProvider(dimensions=384)
        provider._model = mock_model

        with pytest.raises(EmbeddingError, match="model crashed"):
            await provider.generate_embedding("test text")

    async def test_default_task_is_search_document(self) -> None:
        # Nomic requires a task prefix; default = search_document (ingest path).
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([[0.0] * 384])
        provider = FastEmbedProvider(dimensions=384)
        provider._model = mock_model

        await provider.generate_embedding("specific text")

        assert mock_model.embed.call_args[0][0] == ["search_document: specific text"]

    async def test_search_query_task_prefix(self) -> None:
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([[0.0] * 384])
        provider = FastEmbedProvider(dimensions=384)
        provider._model = mock_model

        await provider.generate_embedding("what is X?", task="search_query")

        assert mock_model.embed.call_args[0][0] == ["search_query: what is X?"]

    async def test_clustering_task_prefix(self) -> None:
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([[0.0] * 384])
        provider = FastEmbedProvider(dimensions=384)
        provider._model = mock_model

        await provider.generate_embedding("foo", task="clustering")

        assert mock_model.embed.call_args[0][0] == ["clustering: foo"]


class TestBatchEmbedding:
    async def test_returns_list_of_embeddings(self) -> None:
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([[0.0] * 384, [1.0] * 384])

        provider = FastEmbedProvider(dimensions=384)
        provider._model = mock_model

        results = await provider.generate_embeddings_batch(["text one", "text two"])

        assert len(results) == 2
        assert len(results[0]) == 384
        assert len(results[1]) == 384

    async def test_empty_input_returns_empty(self) -> None:
        provider = FastEmbedProvider(dimensions=384)
        results = await provider.generate_embeddings_batch([])
        assert results == []

    async def test_batch_applies_prefix_to_each_text(self) -> None:
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([[0.0] * 384, [1.0] * 384])
        provider = FastEmbedProvider(dimensions=384)
        provider._model = mock_model

        await provider.generate_embeddings_batch(["one", "two"], task="search_query")

        assert mock_model.embed.call_args[0][0] == [
            "search_query: one", "search_query: two",
        ]
