"""entity_relations.context keeps the extractor's relation context verbatim."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.llm.types import ExtractionResult, Relation

if TYPE_CHECKING:
    from recollect.models import EntityRelation

# Extractor output from the Haiku 5.5 guidance screen evidence, verbatim.
ATTRIBUTION = Relation(
    source="user's low-activity hive",
    relation="scheduled_for",
    target="colony merge october 2026",
    context=(
        "Halvard Lunde suggested merging the queenless low-activity hive with"
        " the other hive in October 2026, a planned logistics step that depends"
        " on the colony's condition."
    ),
    confidence=0.8,
)
UNCERTAINTY = Relation(
    source="user's low-activity hive",
    relation="requires",
    target="colony merge with user's other hive",
    context=(
        "The hive with almost no entrance activity is reported (not lab-confirmed)"
        " to have probably lost its queen, and merging it with the other hive is"
        " being considered as a remedy; confirm queen status before acting."
    ),
    confidence=0.6,
)
CONDITION = Relation(
    source="user",
    relation="holds_role",
    target="head of wine",
    context=(
        "The user's boss has promised the head of wine position contingent on"
        " passing the Level 3 sommelier certificate, so this is a prospective"
        " role not yet held."
    ),
    confidence=0.7,
)


async def _stored(
    mock_storage: MagicMock, mock_embeddings: AsyncMock, rel: Relation
) -> tuple[str, list[EntityRelation]]:
    extractor = AsyncMock()
    extractor.extract.return_value = ExtractionResult(relations=[rel])
    mem = CognitiveMemory(
        storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
    )
    trace = await mem.experience("hive notes", user_id="u1")
    store = mock_storage.entity_relations.store_relation
    return trace.id, [call.args[0] for call in store.await_args_list]


class TestExtractorContextSurvives:
    @pytest.mark.parametrize(
        ("rel", "phrase"),
        [
            (ATTRIBUTION, "Halvard Lunde suggested"),
            (UNCERTAINTY, "is being considered"),
            (CONDITION, "promised the head of wine position contingent on passing"),
        ],
        ids=["attribution", "uncertainty", "condition"],
    )
    async def test_context_is_the_extractor_text(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        rel: Relation,
        phrase: str,
    ) -> None:
        trace_id, [stored] = await _stored(mock_storage, mock_embeddings, rel)
        assert stored.context == rel.context
        assert phrase in stored.context
        assert (stored.source_entity, stored.relation, stored.target_entity) == (
            rel.source,
            rel.relation,
            rel.target,
        )
        assert (stored.weight, stored.source_trace_id) == (rel.confidence, trace_id)

    async def test_context_is_not_trimmed_or_rewritten(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        padded = f"  {ATTRIBUTION.context}\n"
        rel = ATTRIBUTION.model_copy(update={"context": padded})
        _, [stored] = await _stored(mock_storage, mock_embeddings, rel)
        assert stored.context == padded


class TestTripleFallback:
    @pytest.mark.parametrize("blank", ["", " \t\n "], ids=["empty", "whitespace"])
    async def test_blank_context_stores_the_literal_triple(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock, blank: str
    ) -> None:
        rel = CONDITION.model_copy(update={"context": blank})
        _, [stored] = await _stored(mock_storage, mock_embeddings, rel)
        assert stored.context == "user holds_role head of wine"
