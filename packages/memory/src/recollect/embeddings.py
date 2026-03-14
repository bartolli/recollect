"""Embedding generation using FastEmbed.

Uses nomic-ai/nomic-embed-text-v1.5-Q (768 dimensions) for local embedding
generation. No API calls required -- runs entirely on-device.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from recollect.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class FastEmbedProvider:
    """FastEmbed-based embedding provider."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5-Q",
        dimensions: int = 768,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._model: Any = None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self._model_name)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self._model_name}': {exc}"
                ) from exc
        return self._model

    async def warm(self) -> None:
        """Pre-load the embedding model to avoid cold-start latency.

        Runs model initialization in a thread to avoid blocking the
        event loop. Safe to call multiple times (no-op if already loaded).
        """
        await asyncio.to_thread(self._get_model)

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        try:
            model = self._get_model()
            embeddings = await asyncio.to_thread(lambda: list(model.embed([text])))
            return [float(v) for v in embeddings[0]]
        except EmbeddingError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            raise EmbeddingError(str(exc)) from exc

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in one call."""
        if not texts:
            return []
        try:
            model = self._get_model()
            embeddings = await asyncio.to_thread(lambda: list(model.embed(texts)))
            return [[float(v) for v in emb] for emb in embeddings]
        except EmbeddingError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            raise EmbeddingError(str(exc)) from exc
