"""Tests for relation context_tag embedding into trace concept space."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory, _collect_concept_phrases
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import MemoryTrace

_D = 768


def _emb(s: float = 0.1) -> list[float]:
    return [s + i * 0.001 for i in range(_D)]


def _rel(tags: list[str]) -> Relation:
    return Relation(
        source="x",
        relation="works_at",
        target="y",
        category="identity",
        context_tags=tags,
    )


def _mem(
    st: MagicMock, emb: AsyncMock, *, embed_tags: bool
) -> CognitiveMemory:
    cfg = MemoryConfig()
    cfg._set("extraction.embed_relation_tags", embed_tags)
    return CognitiveMemory(storage=st, embeddings=emb, config=cfg)


class TestCollectConceptPhrases:
    def test_concepts_only_when_flag_off(self) -> None:
        result = ExtractionResult(
            concepts=["a", "b"],
            relations=[_rel(["t1", "t2"])],
        )
        assert _collect_concept_phrases(result, include_relation_tags=False) == [
            "a",
            "b",
        ]

    def test_extends_with_relation_tags_when_flag_on(self) -> None:
        result = ExtractionResult(
            concepts=["a", "b"],
            relations=[_rel(["t1", "t2"]), _rel(["t3"])],
        )
        assert _collect_concept_phrases(result, include_relation_tags=True) == [
            "a",
            "b",
            "t1",
            "t2",
            "t3",
        ]

    def test_dedupe_case_insensitive_preserves_first_form(self) -> None:
        result = ExtractionResult(
            concepts=["Coffee", "Tea"],
            relations=[_rel(["coffee", "tea", "lemon"])],
        )
        out = _collect_concept_phrases(result, include_relation_tags=True)
        assert out == ["Coffee", "Tea", "lemon"]

    def test_strips_whitespace(self) -> None:
        result = ExtractionResult(
            concepts=["  a  ", "b"],
            relations=[_rel([" a", "  c"])],
        )
        out = _collect_concept_phrases(result, include_relation_tags=True)
        assert out == ["a", "b", "c"]

    def test_skips_blank_phrases(self) -> None:
        result = ExtractionResult(
            concepts=["a", "", "  "],
            relations=[_rel(["", "b"])],
        )
        out = _collect_concept_phrases(result, include_relation_tags=True)
        assert out == ["a", "b"]

    def test_empty_inputs_yield_empty(self) -> None:
        result = ExtractionResult(concepts=[], relations=[])
        assert _collect_concept_phrases(result, include_relation_tags=True) == []


class TestEmbedTraceConceptsFlag:
    @pytest.mark.asyncio()
    async def test_flag_off_excludes_relation_tags(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        m = _mem(mock_storage, mock_embeddings, embed_tags=False)
        trace = MemoryTrace(content="t", embedding=_emb())
        result = ExtractionResult(
            concepts=["c1", "c2"], relations=[_rel(["t1", "t2"])]
        )
        await m._embed_trace_concepts(trace, result)
        mock_embeddings.generate_embeddings_batch.assert_awaited_once_with(
            ["c1", "c2"], task="search_document"
        )

    @pytest.mark.asyncio()
    async def test_flag_on_includes_relation_tags(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        m = _mem(mock_storage, mock_embeddings, embed_tags=True)
        trace = MemoryTrace(content="t", embedding=_emb())
        result = ExtractionResult(
            concepts=["c1"], relations=[_rel(["t1", "t2", "c1"])]
        )
        await m._embed_trace_concepts(trace, result)
        # c1 deduped from t-tags
        mock_embeddings.generate_embeddings_batch.assert_awaited_once_with(
            ["c1", "t1", "t2"], task="search_document"
        )
        store_fn = mock_storage.concept_embeddings.store_concept_embeddings
        stored = store_fn.call_args[0][0]
        assert len(stored) == 3
        assert all(ce.owner_type == "trace" for ce in stored)
        assert all(ce.owner_id == trace.id for ce in stored)

    @pytest.mark.asyncio()
    async def test_flag_on_no_phrases_skips(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        m = _mem(mock_storage, mock_embeddings, embed_tags=True)
        trace = MemoryTrace(content="t", embedding=_emb())
        await m._embed_trace_concepts(
            trace, ExtractionResult(concepts=[], relations=[])
        )
        mock_embeddings.generate_embeddings_batch.assert_not_awaited()
