"""Supersede: the replacement stands where the original stood.

The original is forgotten through forget() after the replacement is stored
and before its projections are derived, so the replacement never credits a
fact of the statement it replaces.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import FactStatus, ForgetResult, MemoryTrace, PersonaFact

_A = MemoryTrace(id="trace-a", content="I prefer green tea", user_id="u1")


def _fact(obj: str, status: FactStatus = "candidate") -> PersonaFact:
    return PersonaFact(
        id=f"fact-{obj}",
        subject="user",
        predicate="prefers",
        object=obj,
        category="preference",
        content=f"The user prefers {obj}.",
        source_trace_id=_A.id,
        user_id="u1",
        status=status,
    )


def _extraction(
    obj: str, fact_type: Literal["semantic", "episodic"] = "semantic"
) -> ExtractionResult:
    return ExtractionResult(
        fact_type=fact_type,
        relations=[
            Relation(
                source="user",
                relation="prefers",
                target=obj,
                confidence=0.9,
                category="preference",
                context=f"The user now says they prefer {obj}.",
            )
        ],
        significance=0.5,
    )


@pytest.fixture()
def extractor() -> AsyncMock:
    x = AsyncMock()
    x.extract = AsyncMock(return_value=_extraction("green tea"))
    return x


@pytest.fixture()
def mem(
    mock_storage: MagicMock, mock_embeddings: AsyncMock, extractor: AsyncMock
) -> CognitiveMemory:
    mock_storage.traces.get_trace.return_value = _A
    return CognitiveMemory(
        storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
    )


async def _supersede_a(mem: CognitiveMemory, content: str) -> list[PersonaFact]:
    result = await mem.experience_result(content, user_id="u1", supersedes=[_A.id])
    assert result.unstored_facts == []
    stored = mem.storage.facts.store_persona_fact.await_args_list
    return [c.args[0] for c in stored]


async def test_supersede_replaces_the_statement(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    original = _fact("green tea")
    mock_storage.facts.get_facts_by_source_trace_id.return_value = [original]
    # After forget() the original is archived: the dedup read sees it so.
    mock_storage.facts.get_persona_facts.return_value = [
        original.model_copy(update={"status": "archived"})
    ]
    result = await mem.experience_result(
        "Green tea, again", user_id="u1", supersedes=[_A.id]
    )
    assert result.superseded == [
        ForgetResult(trace_id=_A.id, archived_fact_ids=[original.id])
    ]
    mock_storage.facts.increment_mention_count.assert_not_awaited()
    (stored,) = [
        c.args[0] for c in mock_storage.facts.store_persona_fact.await_args_list
    ]
    assert stored.content == "The user now says they prefer green tea."
    assert stored.source_trace_id == result.trace.id
    assert stored.mention_count == 1


async def test_status_inherited_in_the_same_slot(
    mem: CognitiveMemory, mock_storage: MagicMock, extractor: AsyncMock
) -> None:
    mock_storage.facts.get_facts_by_source_trace_id.return_value = [
        _fact("green tea", status="promoted")
    ]
    extractor.extract.return_value = _extraction("black coffee")
    (stored,) = await _supersede_a(mem, "I prefer black coffee, not green tea")
    assert stored.object == "black coffee"
    assert stored.status == "promoted"


async def test_episodic_replacement_derives_when_the_original_carried_a_fact(
    mem: CognitiveMemory, mock_storage: MagicMock, extractor: AsyncMock
) -> None:
    mock_storage.facts.get_facts_by_source_trace_id.return_value = [_fact("green tea")]
    extractor.extract.return_value = _extraction("black coffee", "episodic")
    (stored,) = await _supersede_a(mem, "Today I switched to black coffee")
    assert stored.object == "black coffee"


@pytest.mark.parametrize(
    "carried",
    [[], [_fact("green tea", status="archived")]],
    ids=["no-fact", "only-archived-fact"],
)
async def test_episodic_replacement_without_a_live_fact_derives_none(
    carried: list[PersonaFact],
    mem: CognitiveMemory,
    mock_storage: MagicMock,
    extractor: AsyncMock,
) -> None:
    mock_storage.facts.get_facts_by_source_trace_id.return_value = carried
    extractor.extract.return_value = _extraction("black coffee", "episodic")
    assert await _supersede_a(mem, "Today I switched to black coffee") == []


async def test_a_concurrent_forget_is_a_no_op(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    mock_storage.traces.forget_trace.return_value = False
    result = await mem.experience_result(
        "Green tea, again", user_id="u1", supersedes=[_A.id]
    )
    assert result.superseded == [ForgetResult(trace_id=_A.id)]
    mock_storage.traces.store_trace.assert_awaited_once_with(result.trace)


async def test_the_replacement_is_not_a_temporal_neighbor_of_its_original(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    original = await mem.experience("I prefer green tea", user_id="u1")
    mock_storage.traces.get_trace.return_value = original
    await mem.experience_result(
        "I prefer black coffee", user_id="u1", supersedes=[original.id]
    )
    mock_storage.associations.store_association.assert_not_awaited()
