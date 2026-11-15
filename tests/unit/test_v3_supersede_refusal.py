"""Supersede validation and the one correction path.

Every id is validated under the caller's user before extraction: a refused
correction makes no model call and no write. Only `supersedes` forgets.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import TraceNotFoundError
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import MemoryTrace, PersonaFact

_A = MemoryTrace(id="trace-a", content="I live in Lisbon", user_id="u1")


@pytest.fixture()
def extractor() -> AsyncMock:
    x = AsyncMock()
    x.extract = AsyncMock(
        return_value=ExtractionResult(
            fact_type="semantic",
            relations=[
                Relation(
                    source="user",
                    relation="lives_in",
                    target="Porto",
                    confidence=0.9,
                    category="identity",
                )
            ],
            significance=0.5,
        )
    )
    return x


@pytest.fixture()
def mem(
    mock_storage: MagicMock, mock_embeddings: AsyncMock, extractor: AsyncMock
) -> CognitiveMemory:
    return CognitiveMemory(
        storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
    )


@pytest.mark.parametrize(
    "stored",
    [
        None,
        _A.model_copy(update={"user_id": "u2"}),
        _A.model_copy(update={"status": "forgotten"}),
    ],
    ids=["unknown", "another-user", "already-forgotten"],
)
async def test_refused_before_any_write(
    stored: MemoryTrace | None,
    mem: CognitiveMemory,
    mock_storage: MagicMock,
    extractor: AsyncMock,
) -> None:
    mock_storage.traces.get_trace.return_value = stored
    with pytest.raises(TraceNotFoundError, match="trace-a"):
        await mem.experience_result(
            "Correction: I live in Porto", user_id="u1", supersedes=["trace-a"]
        )
    extractor.extract.assert_not_awaited()
    mock_storage.traces.store_trace.assert_not_awaited()
    mock_storage.traces.forget_trace.assert_not_awaited()


async def test_a_later_bad_id_refuses_the_whole_call(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    mock_storage.traces.get_trace.side_effect = [_A, None]
    with pytest.raises(TraceNotFoundError, match="trace-b"):
        await mem.experience_result(
            "I live in Porto", user_id="u1", supersedes=["trace-a", "trace-b"]
        )
    mock_storage.traces.store_trace.assert_not_awaited()
    mock_storage.traces.forget_trace.assert_not_awaited()


async def test_empty_id_is_a_value_error(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    with pytest.raises(ValueError, match="Trace ID"):
        await mem.experience_result("I live in Porto", user_id="u1", supersedes=[""])
    mock_storage.traces.store_trace.assert_not_awaited()


async def test_an_archived_trace_may_be_superseded(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    mock_storage.traces.get_trace.return_value = _A.model_copy(
        update={"status": "archived"}
    )
    result = await mem.experience_result(
        "I live in Porto", user_id="u1", supersedes=["trace-a"]
    )
    mock_storage.traces.forget_trace.assert_awaited_once_with("trace-a")
    assert [o.trace_id for o in result.superseded] == ["trace-a"]


async def test_duplicate_ids_supersede_once(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    mock_storage.traces.get_trace.return_value = _A
    result = await mem.experience_result(
        "I live in Porto", user_id="u1", supersedes=["trace-a", "trace-a"]
    )
    mock_storage.traces.get_trace.assert_awaited_once_with("trace-a")
    assert [o.trace_id for o in result.superseded] == ["trace-a"]


async def test_experience_passes_supersedes(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    mock_storage.traces.get_trace.return_value = _A
    trace = await mem.experience(
        "I live in Porto", user_id="u1", supersedes=["trace-a"]
    )
    mock_storage.traces.forget_trace.assert_awaited_once_with("trace-a")
    assert trace.content == "I live in Porto"


async def test_nothing_else_corrects(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    mock_storage.facts.get_facts_by_source_trace_id.return_value = [
        PersonaFact(
            subject="user",
            predicate="lives_in",
            object="Lisbon",
            content="The user lives in Lisbon.",
            source_trace_id="trace-a",
            user_id="u1",
        )
    ]
    result = await mem.experience_result(
        "Correction: I live in Porto, not Lisbon", user_id="u1"
    )
    assert result.superseded == []
    mock_storage.traces.forget_trace.assert_not_awaited()
    mock_storage.facts.get_facts_by_source_trace_id.assert_not_awaited()
    mock_storage.facts.update_fact_status.assert_not_awaited()
