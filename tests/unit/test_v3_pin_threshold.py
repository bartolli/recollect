"""pin applies persona.confidence_threshold and falls back to the statement."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import Relation
from recollect.models import MemoryTrace, PersonaFact

_CONTENT = "I might take fencing lessons; my knee is fine"


def _relation(target: str, confidence: float) -> Relation:
    return Relation(
        source="user",
        relation="prefers",
        target=target,
        confidence=confidence,
        category="preference",
    )


def _trace(*relations: Relation) -> MemoryTrace:
    return MemoryTrace(
        content=_CONTENT,
        id="trace-1",
        user_id="u1",
        pattern={"relations": [r.model_dump() for r in relations]},
    )


def _memory(
    storage: MagicMock, embeddings: AsyncMock, *, threshold: float | None = None
) -> CognitiveMemory:
    cfg = MemoryConfig()
    if threshold is not None:
        cfg._set("persona.confidence_threshold", threshold)
    return CognitiveMemory(storage=storage, embeddings=embeddings, config=cfg)


async def test_pin_skips_a_relation_under_the_threshold(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_trace_store: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    mock_trace_store.get_trace.return_value = _trace(
        _relation("fencing", 0.3), _relation("tea", 0.9)
    )
    mem = _memory(mock_storage, mock_embeddings)

    facts = await mem.pin("trace-1")

    assert [(f.object, f.status) for f in facts] == [("tea", "pinned")]
    (stored,) = [c.args[0] for c in mock_fact_store.store_persona_fact.await_args_list]
    assert stored.object == "tea"


async def test_pin_falls_back_to_the_statement(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_trace_store: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    mock_trace_store.get_trace.return_value = _trace(
        _relation("fencing", 0.3), _relation("tea", 0.5)
    )
    mem = _memory(mock_storage, mock_embeddings)

    (fact,) = await mem.pin("trace-1")

    assert (fact.subject, fact.predicate, fact.status) == ("user", "noted", "pinned")
    assert (fact.object, fact.content) == (_CONTENT, _CONTENT)
    assert (fact.source_trace_id, fact.user_id) == ("trace-1", "u1")
    mock_fact_store.store_persona_fact.assert_awaited_once_with(fact)
    mock_fact_store.get_persona_facts.assert_awaited_once_with(
        subject="user", user_id="u1", limit=None
    )


def _noted(status: str) -> PersonaFact:
    return PersonaFact.model_validate(
        {
            "id": f"noted-{status}",
            "subject": "user",
            "predicate": "noted",
            "object": _CONTENT,
            "content": _CONTENT,
            "source_trace_id": "trace-1",
            "user_id": "u1",
            "status": status,
        }
    )


async def test_a_second_pin_reuses_the_live_statement_row(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_trace_store: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    mock_trace_store.get_trace.return_value = _trace(_relation("fencing", 0.3))
    mock_fact_store.get_persona_facts.return_value = [_noted("pinned")]
    mem = _memory(mock_storage, mock_embeddings)

    (fact,) = await mem.pin("trace-1")

    assert (fact.id, fact.status) == ("noted-pinned", "pinned")
    mock_fact_store.store_persona_fact.assert_not_awaited()
    mock_fact_store.adopt_statement.assert_not_awaited()


async def test_an_archived_statement_row_is_reasserted_not_revived(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_trace_store: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    mock_trace_store.get_trace.return_value = _trace(_relation("fencing", 0.3))
    mock_fact_store.get_persona_facts.return_value = [_noted("archived")]
    mem = _memory(mock_storage, mock_embeddings)

    (fact,) = await mem.pin("trace-1")

    assert fact.id != "noted-archived"
    mock_fact_store.store_persona_fact.assert_awaited_once_with(fact)
    mock_fact_store.update_fact_status.assert_not_awaited()


async def test_pin_keeps_a_relation_at_the_threshold(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_trace_store: AsyncMock,
) -> None:
    mock_trace_store.get_trace.return_value = _trace(_relation("tea", 0.7))
    mem = _memory(mock_storage, mock_embeddings, threshold=0.7)

    facts = await mem.pin("trace-1")

    assert [(f.predicate, f.object) for f in facts] == [("prefers", "tea")]


async def test_pin_threshold_comes_from_config(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_trace_store: AsyncMock,
) -> None:
    mock_trace_store.get_trace.return_value = _trace(_relation("tea", 0.9))
    mem = _memory(mock_storage, mock_embeddings, threshold=0.95)

    (fact,) = await mem.pin("trace-1")

    assert (fact.predicate, fact.object) == ("noted", _CONTENT)
