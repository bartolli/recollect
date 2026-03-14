"""Cross-encoder reranker using FastEmbed.

Retained as an opt-in utility, not wired into the SDK pipeline.
Concept attention (ColBERT-style max-sim on LLM-extracted tags) replaced
cross-encoder re-ranking because MS MARCO models cannot perform causal
inference -- they score "peanut allergy" vs "restaurant dinner" at 0.0000.
The LLM reasons about relevance at extraction time; a lexical cross-encoder
at query time is structurally redundant.

Usage: instantiate FastEmbedReranker directly for experimentation.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from recollect.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class FastEmbedReranker:
    """Cross-encoder reranker using ms-marco-MiniLM-L-6-v2."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder.text_cross_encoder import (
                    TextCrossEncoder,
                )

                self._model = TextCrossEncoder(model_name=self._model_name)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise EmbeddingError(
                    f"Failed to load reranker model '{self._model_name}': {exc}"
                ) from exc
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """Score each document against the query.

        Returns a list of relevance scores (one per document),
        in the same order as the input documents.
        """
        if not documents:
            return []
        try:
            model = self._get_model()
            scores = await asyncio.to_thread(
                lambda: list(model.rerank(query, documents))
            )
            return [1.0 / (1.0 + math.exp(-float(s))) for s in scores]
        except EmbeddingError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            raise EmbeddingError(f"Reranking failed: {exc}") from exc
