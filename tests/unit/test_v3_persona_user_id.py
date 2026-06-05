"""Regression: fact-write paths propagate user_id (m002 NOT NULL).

Both fast-track promotion and pin previously constructed PersonaFact without
user_id, so against an m002 DB the insert violated the NOT NULL constraint --
fast-track swallowed the error (facts silently lost), pin crashed. The mocked
suite missed it because the storage mock does not enforce the constraint; these
assert user_id on the fact handed to store_persona_fact.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from recollect.core import CognitiveMemory
from recollect.llm.types import Entity, ExtractionResult, Relation
from recollect.models import MemoryTrace


class TestFactWriteUserId:
    async def test_fast_track_fact_carries_user_id(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        result = ExtractionResult(
            fact_type="semantic",
            entities=[Entity(name="user")],
            relations=[
                Relation(
                    source="user",
                    relation="is_allergic_to",
                    target="peanuts",
                    category="dietary",
                    confidence=0.95,
                )
            ],
            significance=0.95,
        )
        extractor = AsyncMock()
        extractor.extract = AsyncMock(return_value=result)
        mem = CognitiveMemory(
            storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
        )
        await mem.experience("I'm allergic to peanuts", user_id="u1")
        mock_fact_store.store_persona_fact.assert_awaited()
        fact = mock_fact_store.store_persona_fact.call_args.args[0]
        assert fact.user_id == "u1"
        assert fact.status == "promoted"

    async def test_pin_fact_carries_user_id(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_trace_store: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mock_trace_store.get_trace.return_value = MemoryTrace(
            content="Remember this", id="t1", user_id="u1"
        )
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        fact = await mem.pin("t1")
        assert fact.user_id == "u1"
        stored = mock_fact_store.store_persona_fact.call_args.args[0]
        assert stored.user_id == "u1"
