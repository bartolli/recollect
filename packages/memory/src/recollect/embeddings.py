"""Embedding generation using FastEmbed.

Uses nomic-ai/nomic-embed-text-v1.5 (768 dimensions) for local embedding
generation. No API calls required -- runs entirely on-device.
Nomic requires a task-instruction prefix on every input; see EmbeddingTask.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from recollect.exceptions import EmbeddingError

if TYPE_CHECKING:
    from recollect.config import MemoryConfig

logger = logging.getLogger(__name__)

EmbeddingTask = Literal[
    "search_document", "search_query", "clustering", "classification"
]


class FastEmbedProvider:
    """FastEmbed-based embedding provider."""

    TASK_PREFIX_VERSION = "v0.7"

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        dimensions: int = 768,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._cache_dir = self._resolve_cache_dir(cache_dir)
        self._model: Any = None

    @classmethod
    def from_config(cls, config: MemoryConfig) -> FastEmbedProvider:
        # Single construction path: CognitiveMemory and the m003 contract
        # stamp must build identical providers or fresh DBs refuse to start.
        return cls(
            model_name=str(
                config.get("embedding.model", "nomic-ai/nomic-embed-text-v1.5")
            ),
            dimensions=config.embedding_dimensions,
            cache_dir=str(config.get("embedding.cache_dir", "") or "") or None,
        )

    @staticmethod
    def _resolve_cache_dir(cache_dir: str | None) -> str:
        # Override fastembed's tempdir default; macOS purges /tmp, stranding ONNX blobs.
        path = (
            Path(cache_dir).expanduser()
            if cache_dir
            else Path.home() / ".cache" / "recollect" / "fastembed"
        )
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def contract(self) -> tuple[str, str]:
        """Return (model, task_prefix_version) — the stored-vector contract."""
        return (self._model_name, self.TASK_PREFIX_VERSION)

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(
                    model_name=self._model_name,
                    cache_dir=self._cache_dir,
                )
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

    async def generate_embedding(
        self, text: str, *, task: EmbeddingTask = "search_document"
    ) -> list[float]:
        """Generate embedding for a single text. Nomic requires a task prefix."""
        try:
            model = self._get_model()
            prefixed = f"{task}: {text}"
            embeddings = await asyncio.to_thread(lambda: list(model.embed([prefixed])))
            return [float(v) for v in embeddings[0]]
        except EmbeddingError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            raise EmbeddingError(str(exc)) from exc

    async def generate_embeddings_batch(
        self, texts: list[str], *, task: EmbeddingTask = "search_document"
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in one call. Single task per batch."""
        if not texts:
            return []
        try:
            model = self._get_model()
            prefixed = [f"{task}: {t}" for t in texts]
            embeddings = await asyncio.to_thread(lambda: list(model.embed(prefixed)))
            return [[float(v) for v in emb] for emb in embeddings]
        except EmbeddingError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            raise EmbeddingError(str(exc)) from exc
