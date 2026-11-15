"""Unit: MCP remember when fact surfacing fails after the memory is stored."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.exceptions import StorageError
from recollect.models import (
    ExperienceResult,
    MemoryTrace,
    PersonaFact,
    Session,
    SurfacedFacts,
)
from recollect_mcp.server import AppContext, remember


def _ctx(memory: AsyncMock) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = AppContext(
        memory=memory, worker=MagicMock(), user_id="alex"
    )
    ctx.headers = None
    return ctx


@pytest.fixture()
def memory() -> AsyncMock:
    mem = AsyncMock()
    mem.start_session.return_value = Session(id="sess-1", user_id="alex")
    mem.experience_result.return_value = ExperienceResult(
        trace=MemoryTrace(content="dinner on Friday")
    )
    return mem


class TestRememberSurfacingFailure:
    async def test_reply_confirms_the_stored_memory(self, memory: AsyncMock) -> None:
        memory.surface_relevant_facts_result.side_effect = StorageError("down")
        reply = await remember("dinner on Friday", _ctx(memory))
        # "Storage failed" here would invite a second write of a stored memory.
        assert reply.startswith("Remembered [")
        assert "Storage failed" not in reply
        assert "do not store it again" in reply

    async def test_a_partial_lookup_prints_the_facts_and_says_what_is_missing(
        self, memory: AsyncMock
    ) -> None:
        allergy = PersonaFact(
            subject="user",
            predicate="is_allergic_to",
            object="peanuts",
            category="health",
            content="The user is allergic to peanuts.",
            status="promoted",
        )
        memory.surface_relevant_facts_result.return_value = SurfacedFacts(
            facts=[allergy], lookup_failed=True
        )
        reply = await remember("dinner on Friday", _ctx(memory))
        assert "IMPORTANT CONTEXT:" in reply
        assert "is_allergic_to peanuts" in reply
        assert reply.endswith(
            "Other related persona facts could not be loaded: storage error. "
            "The memory is stored; do not store it again."
        )

    async def test_storage_failure_of_the_write_still_reads_failed(
        self, memory: AsyncMock
    ) -> None:
        memory.experience_result.side_effect = StorageError("down")
        reply = await remember("dinner on Friday", _ctx(memory))
        assert reply.startswith("Storage failed")
        memory.surface_relevant_facts_result.assert_not_awaited()
