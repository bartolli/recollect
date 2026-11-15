"""Forget guard -- fact archival, hard/pinned refusals, dedup exclusion.

forget(trace_id, force=False) archives derived facts; hard categories
(health/dietary/constraint) and pinned facts are retained without force
and ride ForgetResult.retained_facts. _store_or_promote_fact excludes
archived facts so a re-stated retraction enters fresh, not swallowed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import TraceNotFoundError
from recollect.models import FactCategory, FactStatus, MemoryTrace, PersonaFact


@pytest.fixture()
def mem(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_extractor: AsyncMock,
) -> CognitiveMemory:
    return CognitiveMemory(
        storage=mock_storage,
        embeddings=mock_embeddings,
        extractor=mock_extractor,
    )


def _fact(
    category: FactCategory = "preference",
    status: FactStatus = "promoted",
    fact_id: str = "f1",
) -> PersonaFact:
    return PersonaFact(
        id=fact_id,
        subject="user",
        predicate="prefers",
        object="thing",
        category=category,
        content="user prefers thing",
        source_trace_id="t1",
        status=status,
    )


class TestForgetFactArchival:
    async def test_non_safety_fact_archived(
        self, mem: CognitiveMemory, mock_fact_store: AsyncMock
    ) -> None:
        mock_fact_store.get_facts_by_source_trace_id.return_value = [_fact()]
        result = await mem.forget("t1")
        mock_fact_store.update_fact_status.assert_awaited_once_with(
            "f1", "archived"
        )
        assert result.archived_fact_ids == ["f1"]
        assert result.retained_facts == []

    async def test_hard_fact_refused_without_force(
        self, mem: CognitiveMemory, mock_fact_store: AsyncMock
    ) -> None:
        fact = _fact(category="health")
        mock_fact_store.get_facts_by_source_trace_id.return_value = [fact]
        result = await mem.forget("t1")
        mock_fact_store.update_fact_status.assert_not_awaited()
        assert result.archived_fact_ids == []
        assert result.retained_facts == [fact]

    async def test_force_archives_hard_fact(
        self, mem: CognitiveMemory, mock_fact_store: AsyncMock
    ) -> None:
        mock_fact_store.get_facts_by_source_trace_id.return_value = [
            _fact(category="dietary")
        ]
        result = await mem.forget("t1", force=True)
        mock_fact_store.update_fact_status.assert_awaited_once_with(
            "f1", "archived"
        )
        assert result.archived_fact_ids == ["f1"]
        assert result.retained_facts == []

    async def test_pinned_fact_refused_without_force(
        self, mem: CognitiveMemory, mock_fact_store: AsyncMock
    ) -> None:
        # Explicit user state outranks the implicit trace cascade.
        fact = _fact(category="preference", status="pinned")
        mock_fact_store.get_facts_by_source_trace_id.return_value = [fact]
        result = await mem.forget("t1")
        mock_fact_store.update_fact_status.assert_not_awaited()
        assert result.retained_facts == [fact]

    async def test_already_archived_fact_skipped(
        self, mem: CognitiveMemory, mock_fact_store: AsyncMock
    ) -> None:
        mock_fact_store.get_facts_by_source_trace_id.return_value = [
            _fact(status="archived")
        ]
        result = await mem.forget("t1")
        mock_fact_store.update_fact_status.assert_not_awaited()
        assert result.archived_fact_ids == []
        assert result.retained_facts == []


class TestForgetOfAForgottenTrace:
    @pytest.fixture()
    def forgotten(self, mock_trace_store: AsyncMock) -> None:
        # No row flips: the trace was forgotten before.
        mock_trace_store.forget_trace.return_value = False
        mock_trace_store.get_trace.return_value = MemoryTrace(
            id="t1", content="x", status="forgotten"
        )

    @pytest.mark.usefixtures("forgotten")
    async def test_force_archives_what_the_first_forget_retained(
        self, mem: CognitiveMemory, mock_fact_store: AsyncMock
    ) -> None:
        mock_fact_store.get_facts_by_source_trace_id.return_value = [
            _fact(category="health"),
            _fact(status="pinned", fact_id="f2"),
        ]
        result = await mem.forget("t1", force=True)
        assert result.archived_fact_ids == ["f1", "f2"]
        assert result.retained_facts == []

    @pytest.mark.usefixtures("forgotten")
    async def test_a_plain_repeat_finishes_the_fact_pass_under_the_guard(
        self, mem: CognitiveMemory, mock_fact_store: AsyncMock
    ) -> None:
        # A forget that stopped after the trace flip left an ordinary fact live.
        health = _fact(category="health", fact_id="f2")
        mock_fact_store.get_facts_by_source_trace_id.return_value = [_fact(), health]
        result = await mem.forget("t1")
        assert result.archived_fact_ids == ["f1"]
        assert result.retained_facts == [health]

    @pytest.mark.usefixtures("forgotten")
    async def test_a_completed_repeat_changes_nothing(
        self, mem: CognitiveMemory, mock_fact_store: AsyncMock
    ) -> None:
        mock_fact_store.get_facts_by_source_trace_id.return_value = [
            _fact(status="archived")
        ]
        result = await mem.forget("t1")
        assert (result.archived_fact_ids, result.retained_facts) == ([], [])
        mock_fact_store.update_fact_status.assert_not_awaited()

    @pytest.mark.parametrize("force", [False, True])
    async def test_a_missing_id_stays_not_found(
        self,
        force: bool,
        mem: CognitiveMemory,
        mock_trace_store: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        mock_trace_store.forget_trace.return_value = False
        mock_trace_store.get_trace.return_value = None
        with pytest.raises(TraceNotFoundError):
            await mem.forget("t1", force=force)
        mock_fact_store.update_fact_status.assert_not_awaited()


class TestDedupExcludesArchived:
    async def test_restated_retraction_enters_fresh(
        self, mem: CognitiveMemory, mock_fact_store: AsyncMock
    ) -> None:
        # Swallowed-mention regression: increment-on-archived returned
        # None and the re-stated fact vanished.
        archived_dup = _fact(status="archived")
        mock_fact_store.get_persona_facts.return_value = [archived_dup]
        new_fact = _fact(status="candidate", fact_id="f2")
        stored = await mem._store_or_promote_fact(new_fact)
        assert stored is new_fact
        mock_fact_store.increment_mention_count.assert_not_awaited()
        mock_fact_store.store_persona_fact.assert_awaited_once()

    async def test_archived_fact_not_a_supersede_target(
        self, mem: CognitiveMemory, mock_fact_store: AsyncMock
    ) -> None:
        archived = _fact(status="archived")
        mock_fact_store.get_persona_facts.return_value = [archived]
        contradicting = PersonaFact(
            id="f3",
            subject="user",
            predicate="prefers",
            object="other thing",
            content="user prefers other thing",
            status="candidate",
        )
        stored = await mem._store_or_promote_fact(contradicting)
        assert stored is contradicting
        mock_fact_store.supersede_persona_fact.assert_not_awaited()
        mock_fact_store.store_persona_fact.assert_awaited_once()
