"""pin() reconciles with existing facts instead of duplicating.

adr-pin-upsert-semantics: pin flips a live (non-archived) subject+predicate+object
twin to pinned, inserts only an unmatched relation, and re-asserts over an
archived twin without resurrecting the retracted row.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from recollect.core import CognitiveMemory
from recollect.llm.types import Relation
from recollect.models import MemoryTrace, PersonaFact

_REL = Relation(source="Angel", relation="works_at", target="DTCC", category="identity")


def _trace() -> MemoryTrace:
    return MemoryTrace(
        content="ctx",
        id="trace-1",
        user_id="u1",
        pattern={"relations": [_REL.model_dump()]},
    )


class TestPinUpsert:
    async def test_pin_flips_live_twin_without_duplicating(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_trace_store: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mock_trace_store.get_trace.return_value = _trace()
        twin = PersonaFact(
            subject="Angel", predicate="works_at", object="DTCC", content="c"
        )
        mock_fact_store.get_persona_facts.return_value = [twin]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)

        facts = await mem.pin("trace-1")

        mock_fact_store.update_fact_status.assert_awaited_once_with(twin.id, "pinned")
        mock_fact_store.store_persona_fact.assert_not_awaited()
        assert [f.status for f in facts] == ["pinned"]

    async def test_pin_inserts_unmatched_relation(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_trace_store: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mock_trace_store.get_trace.return_value = _trace()
        mock_fact_store.get_persona_facts.return_value = []
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)

        facts = await mem.pin("trace-1")

        mock_fact_store.store_persona_fact.assert_awaited_once()
        mock_fact_store.update_fact_status.assert_not_awaited()
        assert facts[0].status == "pinned"

    async def test_pin_reasserts_over_archived_without_flipping_it(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_trace_store: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mock_trace_store.get_trace.return_value = _trace()
        archived = PersonaFact(
            subject="Angel",
            predicate="works_at",
            object="DTCC",
            content="c",
            status="archived",
        )
        mock_fact_store.get_persona_facts.return_value = [archived]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)

        facts = await mem.pin("trace-1")

        mock_fact_store.store_persona_fact.assert_awaited_once()
        mock_fact_store.update_fact_status.assert_not_awaited()
        assert facts[0].status == "pinned"
