"""Supersede keeps the forget guard: pinned and safety facts are retained.

A retained row restated by the replacement with the same triple adopts the
replacement's statement and keeps its status, with no mention credit.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import StorageError
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import MemoryTrace, PersonaFact

_A = MemoryTrace(id="trace-a", content="I am allergic to peanuts", user_id="u1")
_PINNED = PersonaFact(
    id="fact-pinned",
    subject="user",
    predicate="prefers",
    object="green tea",
    category="preference",
    content="The user prefers green tea.",
    source_trace_id=_A.id,
    user_id="u1",
    status="pinned",
)
_HEALTH = _PINNED.model_copy(
    update={
        "id": "fact-health",
        "predicate": "is_allergic_to",
        "object": "peanuts",
        "category": "health",
        "content": "The user is allergic to peanuts.",
        "status": "promoted",
    }
)


def _restating(row: PersonaFact) -> ExtractionResult:
    return ExtractionResult(
        fact_type="episodic",
        relations=[
            Relation(
                source=row.subject,
                relation=row.predicate,
                target=row.object,
                confidence=0.9,
                category=row.category,
                context="The user restated it at the clinic.",
                context_tags=["clinic visit", "restated"],
            )
        ],
        significance=0.5,
    )


@pytest.fixture()
def extractor() -> AsyncMock:
    x = AsyncMock()
    x.extract = AsyncMock(return_value=ExtractionResult(significance=0.5))
    return x


@pytest.fixture()
def mem(
    mock_storage: MagicMock, mock_embeddings: AsyncMock, extractor: AsyncMock
) -> CognitiveMemory:
    mock_storage.traces.get_trace.return_value = _A
    mock_storage.facts.get_facts_by_source_trace_id.return_value = [_PINNED, _HEALTH]
    return CognitiveMemory(
        storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
    )


async def test_pinned_and_safety_facts_are_retained_and_listed(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    result = await mem.experience_result(
        "Correction: no allergy", user_id="u1", supersedes=[_A.id]
    )
    (outcome,) = result.superseded
    assert [f.id for f in outcome.retained_facts] == [_PINNED.id, _HEALTH.id]
    assert outcome.archived_fact_ids == []
    mock_storage.facts.update_fact_status.assert_not_awaited()


@pytest.mark.parametrize("row", [_PINNED, _HEALTH], ids=["pinned", "safety"])
async def test_same_triple_retained_row_adopts_the_statement(
    row: PersonaFact,
    mem: CognitiveMemory,
    mock_storage: MagicMock,
    extractor: AsyncMock,
) -> None:
    extractor.extract.return_value = _restating(row)
    result = await mem.experience_result(
        "Restated at the clinic", user_id="u1", supersedes=[_A.id]
    )
    facts = mock_storage.facts
    facts.adopt_statement.assert_awaited_once()
    fact_id, statement = facts.adopt_statement.await_args.args
    assert fact_id == row.id
    assert statement.content == "The user restated it at the clinic."
    assert statement.source_trace_id == result.trace.id
    # Status kept, no mention credit, no second row.
    facts.update_fact_status.assert_not_awaited()
    facts.increment_mention_count.assert_not_awaited()
    facts.store_persona_fact.assert_not_awaited()
    concepts = mock_storage.concept_embeddings
    concepts.delete_by_owner.assert_awaited_once_with("fact", row.id)
    embedded = concepts.store_concept_embeddings.await_args.args[0]
    assert {e.owner_id for e in embedded} == {row.id}
    assert [e.concept for e in embedded] == ["clinic visit", "restated"]


@pytest.mark.parametrize(
    "failure",
    [{"side_effect": StorageError("down")}, {"return_value": False}],
    ids=["write-raises", "row-gone"],
)
async def test_failed_adoption_is_listed(
    failure: dict[str, object],
    mem: CognitiveMemory,
    mock_storage: MagicMock,
    extractor: AsyncMock,
) -> None:
    mock_storage.facts.adopt_statement.configure_mock(**failure)
    extractor.extract.return_value = _restating(_PINNED)
    result = await mem.experience_result(
        "Restated at the clinic", user_id="u1", supersedes=[_A.id]
    )
    assert [(f.predicate, f.object) for f in result.unstored_facts] == [
        ("prefers", "green tea")
    ]
    mock_storage.concept_embeddings.store_concept_embeddings.assert_not_awaited()


async def test_a_triple_repeated_in_the_replacement_keeps_its_first_context(
    mem: CognitiveMemory, mock_storage: MagicMock, extractor: AsyncMock
) -> None:
    restated = _restating(_PINNED)
    again = restated.relations[0].model_copy(update={"context": "Said twice."})
    extractor.extract.return_value = restated.model_copy(
        update={"relations": [*restated.relations, again]}
    )
    await mem.experience_result(
        "Restated at the clinic", user_id="u1", supersedes=[_A.id]
    )
    facts = mock_storage.facts
    facts.adopt_statement.assert_awaited_once()
    assert facts.adopt_statement.await_args.args[1].content == (
        "The user restated it at the clinic."
    )
    facts.increment_mention_count.assert_not_awaited()


async def test_outcome_lists_retained_rows_as_they_stand(
    mem: CognitiveMemory, mock_storage: MagicMock, extractor: AsyncMock
) -> None:
    extractor.extract.return_value = _restating(_PINNED)
    adopted = _PINNED.model_copy(
        update={"content": "The user restated it at the clinic."}
    )
    reads = iter([[_PINNED, _HEALTH]] * 2)

    async def by_source(trace_id: str) -> list[PersonaFact]:
        # Two reads up to forget's own. Afterwards the adopted row names the
        # replacement, and a derived fact has superseded the health row.
        return next(reads, [] if trace_id == _A.id else [adopted])

    mock_storage.facts.get_facts_by_source_trace_id.side_effect = by_source
    result = await mem.experience_result(
        "Restated at the clinic", user_id="u1", supersedes=[_A.id]
    )
    (outcome,) = result.superseded
    assert [(f.id, f.content) for f in outcome.retained_facts] == [
        (_PINNED.id, "The user restated it at the clinic.")
    ]
