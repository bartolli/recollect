"""A fact row adopts a restatement of its triple from another live trace.

A triple repeated inside one trace keeps its first context.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import StorageError
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import MemoryTrace, PersonaFact

_ROW = PersonaFact(
    id="fact-a",
    subject="user",
    predicate="prefers",
    object="green tea",
    category="preference",
    content="The user prefers green tea.",
    context_tags=["tea"],
    source_trace_id="trace-a",
    user_id="u1",
    status="candidate",
)


def _relation(context: str, tags: list[str]) -> Relation:
    return Relation(
        source="user",
        relation="prefers",
        target="green tea",
        confidence=0.9,
        category="preference",
        context=context,
        context_tags=tags,
    )


_C = _relation("The user prefers green tea, without sugar.", ["no sugar"])


def _extraction(*relations: Relation) -> ExtractionResult:
    return ExtractionResult(
        fact_type="semantic", relations=list(relations), significance=0.5
    )


def _trace(
    trace_id: str, *, status: Literal["active", "forgotten"] = "active"
) -> MemoryTrace:
    return MemoryTrace(
        id=trace_id,
        content="I prefer green tea, without sugar",
        user_id="u1",
        status=status,
        pattern=_extraction(_C).model_dump(),
    )


def _fact_concepts(store: AsyncMock) -> list[tuple[str, str]]:
    return [
        (e.owner_id, e.concept)
        for call in store.await_args_list
        for e in call.args[0]
        if e.owner_type == "fact"
    ]


@pytest.fixture()
def extractor() -> AsyncMock:
    x = AsyncMock()
    x.extract = AsyncMock(return_value=_extraction(_C))
    return x


@pytest.fixture()
def mem(
    mock_storage: MagicMock, mock_embeddings: AsyncMock, extractor: AsyncMock
) -> CognitiveMemory:
    mock_storage.facts.get_persona_facts.return_value = [_ROW]
    return CognitiveMemory(
        storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
    )


async def test_extraction_restatement_is_adopted_and_promoted(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    trace = await mem.experience("I prefer green tea, without sugar", user_id="u1")
    facts = mock_storage.facts
    facts.adopt_statement.assert_awaited_once()
    fact_id, statement = facts.adopt_statement.await_args.args
    assert (fact_id, statement.source_trace_id) == (_ROW.id, trace.id)
    assert statement.content == _C.context
    facts.increment_mention_count.assert_awaited_once_with(_ROW.id)
    facts.update_fact_status.assert_awaited_once_with(_ROW.id, "promoted")
    facts.store_persona_fact.assert_not_awaited()
    concepts = mock_storage.concept_embeddings
    concepts.delete_by_owner.assert_awaited_once_with("fact", _ROW.id)
    assert _fact_concepts(concepts.store_concept_embeddings) == [(_ROW.id, "no sugar")]


async def test_triple_repeated_inside_one_trace_keeps_first_context(
    mem: CognitiveMemory, mock_storage: MagicMock, extractor: AsyncMock
) -> None:
    rows: list[PersonaFact] = []

    async def store(fact: PersonaFact) -> str:
        rows.append(fact)
        return fact.id

    async def read(*_: object, **__: object) -> list[PersonaFact]:
        return list(rows)

    facts = mock_storage.facts
    facts.store_persona_fact.side_effect = store
    facts.get_persona_facts.side_effect = read
    extractor.extract.return_value = _extraction(
        _relation("The user prefers green tea.", ["tea"]), _C
    )
    await mem.experience("I prefer green tea, later without sugar", user_id="u1")
    (row,) = rows
    assert row.content == "The user prefers green tea."
    facts.adopt_statement.assert_not_awaited()
    facts.increment_mention_count.assert_awaited_once_with(row.id)


@pytest.mark.parametrize(
    "failure",
    [{"side_effect": StorageError("down")}, {"return_value": False}],
    ids=["write-raises", "row-gone"],
)
async def test_failed_adoption_is_listed_without_credit(
    failure: dict[str, object], mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    mock_storage.facts.adopt_statement.configure_mock(**failure)
    result = await mem.experience_result("I prefer green tea", user_id="u1")
    assert [(f.predicate, f.object) for f in result.unstored_facts] == [
        ("prefers", "green tea")
    ]
    mock_storage.facts.increment_mention_count.assert_not_awaited()


async def test_pin_restatement_is_adopted_and_pinned(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    mock_storage.traces.get_trace.return_value = _trace("trace-c")
    (pinned,) = await mem.pin("trace-c")
    facts = mock_storage.facts
    facts.adopt_statement.assert_awaited_once()
    fact_id, statement = facts.adopt_statement.await_args.args
    assert (fact_id, statement.content) == (_ROW.id, _C.context)
    facts.update_fact_status.assert_awaited_once_with(_ROW.id, "pinned")
    facts.store_persona_fact.assert_not_awaited()
    mock_storage.concept_embeddings.delete_by_owner.assert_awaited_once_with(
        "fact", _ROW.id
    )
    assert (pinned.id, pinned.status) == (_ROW.id, "pinned")
    assert (pinned.content, pinned.source_trace_id) == (_C.context, "trace-c")


@pytest.mark.parametrize(
    "trace",
    [_trace("trace-a"), _trace("trace-c", status="forgotten")],
    ids=["same-trace", "forgotten-trace"],
)
async def test_pin_flips_without_adopting(
    trace: MemoryTrace, mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    mock_storage.traces.get_trace.return_value = trace
    (pinned,) = await mem.pin(trace.id)
    mock_storage.facts.adopt_statement.assert_not_awaited()
    mock_storage.facts.update_fact_status.assert_awaited_once_with(_ROW.id, "pinned")
    assert (pinned.status, pinned.content, pinned.source_trace_id) == (
        "pinned",
        _ROW.content,
        _ROW.source_trace_id,
    )


async def test_failed_adoption_fails_the_pin(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    mock_storage.traces.get_trace.return_value = _trace("trace-c")
    mock_storage.facts.adopt_statement.side_effect = StorageError("down")
    with pytest.raises(StorageError):
        await mem.pin("trace-c")
    mock_storage.facts.update_fact_status.assert_not_awaited()
