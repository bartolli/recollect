"""Supersede under a storage failure: the stored replacement is reported.

The replacement trace is written before any original is forgotten, so a
failed forget is listed in the result and never raised as a failed write.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import StorageError
from recollect.llm.types import ExtractionResult, Relation
from recollect.models import MemoryTrace, PersonaFact

_A1 = MemoryTrace(id="trace-a1", content="Dinner is on Friday", user_id="u1")
_A2 = MemoryTrace(id="trace-a2", content="Dinner is at seven", user_id="u1")
_TRACES = {t.id: t for t in (_A1, _A2)}
_PINNED = PersonaFact(
    id="fact-pinned",
    subject="user",
    predicate="prefers",
    object="green tea",
    content="The user prefers green tea.",
    source_trace_id=_A1.id,
    user_id="u1",
    status="pinned",
)
_ORDINARY = _PINNED.model_copy(
    update={"id": "fact-a2", "source_trace_id": _A2.id, "status": "candidate"}
)


@pytest.fixture()
def mem(mock_storage: MagicMock, mock_embeddings: AsyncMock) -> CognitiveMemory:
    extractor = AsyncMock()
    extractor.extract = AsyncMock(
        return_value=ExtractionResult(
            fact_type="semantic",
            relations=[
                Relation(
                    source="user",
                    relation="scheduled_for",
                    target="Saturday dinner",
                    confidence=0.9,
                    category="schedule",
                    context="Dinner moved to Saturday at eight.",
                )
            ],
            significance=0.5,
        )
    )
    mock_storage.traces.get_trace.side_effect = _TRACES.get
    return CognitiveMemory(
        storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
    )


async def _correct(mem: CognitiveMemory) -> tuple[list[str], list[str]]:
    result = await mem.experience_result(
        "Dinner moved to Saturday at eight",
        user_id="u1",
        supersedes=[_A1.id, _A2.id],
    )
    assert result.trace.content == "Dinner moved to Saturday at eight"
    return [o.trace_id for o in result.superseded], result.unsuperseded


async def test_a_failure_before_the_flip_is_listed_not_raised(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    async def flip(trace_id: str) -> bool:
        if trace_id == _A2.id:
            raise StorageError("down")
        return True

    mock_storage.traces.forget_trace.side_effect = flip
    assert await _correct(mem) == ([_A1.id], [_A2.id])
    # The replacement is stored and its projections are still derived.
    mock_storage.traces.store_trace.assert_awaited_once()
    mock_storage.facts.store_persona_fact.assert_awaited_once()


async def test_a_failure_after_the_flip_is_listed_not_raised(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    async def by_source(trace_id: str) -> list[PersonaFact]:
        return [_ORDINARY] if trace_id == _A2.id else []

    mock_storage.facts.get_facts_by_source_trace_id.side_effect = by_source
    mock_storage.facts.update_fact_status.side_effect = StorageError("down")
    assert await _correct(mem) == ([_A1.id], [_A2.id])
    assert mock_storage.traces.forget_trace.await_count == 2


async def test_a_failed_first_supersession_does_not_stop_the_second(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    async def flip(trace_id: str) -> bool:
        if trace_id == _A1.id:
            raise OSError("reset")
        return True

    mock_storage.traces.forget_trace.side_effect = flip
    assert await _correct(mem) == ([_A2.id], [_A1.id])


async def test_retained_rows_stay_listed_when_the_re_read_fails(
    mem: CognitiveMemory, mock_storage: MagicMock
) -> None:
    reads = iter([[_PINNED]] * 2)

    async def by_source(trace_id: str) -> list[PersonaFact]:
        if trace_id == _A1.id and (rows := next(reads, None)) is not None:
            return rows
        if trace_id == _A2.id:
            return []
        raise StorageError("down")

    mock_storage.facts.get_facts_by_source_trace_id.side_effect = by_source
    result = await mem.experience_result(
        "Dinner moved to Saturday at eight",
        user_id="u1",
        supersedes=[_A1.id, _A2.id],
    )
    assert result.unsuperseded == []
    assert [f.id for f in result.superseded[0].retained_facts] == [_PINNED.id]
