"""Unit: MCP remember with supersedes, and the echo of what the write did."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.exceptions import StorageError, TraceNotFoundError
from recollect.models import (
    ExperienceResult,
    ForgetResult,
    MemoryTrace,
    PersonaFact,
    Session,
    SurfacedFacts,
)
from recollect_mcp.server import AppContext, remember

_A = MemoryTrace(
    id="a1b2c3d4-0000-0000-0000-000000000000",
    content="Sarah's dentist appointment is on Monday at 9",
    user_id="alex",
)
_B = MemoryTrace(id="b1b2c3d4-0000-0000-0000-000000000000", content="moved")
_PINNED = PersonaFact(
    id="f1f2f3f4-0000-0000-0000-000000000000",
    subject="Sarah",
    predicate="scheduled_for",
    object="Monday dentist",
    content="Sarah's dentist appointment is on Monday.",
    status="pinned",
)


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
    mem.traces_by_id_prefix.side_effect = lambda prefix, user_id=None: (
        [_A] if _A.id.startswith(prefix) else []
    )
    mem.surface_relevant_facts_result.return_value = SurfacedFacts()
    mem.experience_result.return_value = ExperienceResult(
        trace=_B,
        superseded=[
            ForgetResult(
                trace_id=_A.id, archived_fact_ids=["x"], retained_facts=[_PINNED]
            )
        ],
    )
    return mem


async def test_echo_names_the_superseded_memory_and_retained_facts(
    memory: AsyncMock,
) -> None:
    reply = await remember("moved to Tuesday", _ctx(memory), supersedes=["a1b2c3d4"])
    assert memory.experience_result.await_args.kwargs["supersedes"] == [_A.id]
    lines = reply.splitlines()
    assert lines[0].startswith("Remembered [b1b2c3d4]")
    assert lines[1] == (
        'Superseded [a1b2c3d4] ("Sarah\'s dentist appointment is on Monday at 9"): '
        "1 derived facts archived."
    )
    assert lines[2] == "  [f1f2f3f4] [pinned] Sarah scheduled_for Monday dentist"
    assert "unpin" in lines[3]
    assert "force" not in reply


async def test_archived_count_is_printed_when_zero(memory: AsyncMock) -> None:
    memory.experience_result.return_value = ExperienceResult(
        trace=_B, superseded=[ForgetResult(trace_id=_A.id)]
    )
    reply = await remember("moved", _ctx(memory), supersedes=[_A.id])
    assert "0 derived facts archived." in reply
    assert "unpin" not in reply


async def test_echo_precedes_the_context_block(memory: AsyncMock) -> None:
    memory.surface_relevant_facts_result.return_value = SurfacedFacts(facts=[_PINNED])
    reply = await remember("moved", _ctx(memory), supersedes=["a1b2c3d4"])
    assert reply.index("Superseded [") < reply.index("IMPORTANT CONTEXT:")


async def test_echo_survives_a_surfacing_failure(memory: AsyncMock) -> None:
    memory.surface_relevant_facts_result.side_effect = StorageError("down")
    reply = await remember("moved", _ctx(memory), supersedes=["a1b2c3d4"])
    assert "Superseded [a1b2c3d4]" in reply
    assert "do not store it again" in reply


@pytest.mark.parametrize(
    ("raw", "refusal"),
    [
        ("0e0e0e0e", "No memory trace matches id 0e0e0e0e."),
        ("short", "'short' is not a memory id."),
    ],
)
async def test_unresolved_id_is_refused_before_any_write(
    raw: str, refusal: str, memory: AsyncMock
) -> None:
    reply = await remember("moved", _ctx(memory), supersedes=["a1b2c3d4", raw])
    assert reply.startswith(refusal)
    memory.experience_result.assert_not_awaited()
    memory.start_session.assert_not_awaited()


async def test_ambiguous_id_is_refused(memory: AsyncMock) -> None:
    twin = _A.model_copy(update={"id": "a1b2c3d4-ffff"})
    memory.traces_by_id_prefix.side_effect = lambda prefix, user_id=None: [_A, twin]
    reply = await remember("moved", _ctx(memory), supersedes=["a1b2c3d4"])
    assert reply.startswith("Id a1b2c3d4 matches more than one memory.")
    memory.experience_result.assert_not_awaited()


async def test_forgotten_id_is_refused_before_the_session_write(
    memory: AsyncMock,
) -> None:
    forgotten = _A.model_copy(update={"status": "forgotten"})
    memory.traces_by_id_prefix.side_effect = lambda prefix, user_id=None: [forgotten]
    reply = await remember("moved", _ctx(memory), supersedes=["a1b2c3d4"])
    assert reply == "Trace a1b2c3d4 not found."
    memory.start_session.assert_not_awaited()
    memory.experience_result.assert_not_awaited()


async def test_a_trace_forgotten_during_the_write_reads_not_found(
    memory: AsyncMock,
) -> None:
    memory.experience_result.side_effect = TraceNotFoundError(
        f"Trace {_A.id} not found"
    )
    reply = await remember("moved", _ctx(memory), supersedes=["a1b2c3d4"])
    assert reply == f"Trace {_A.id} not found."


async def test_a_failed_supersession_is_echoed_without_inviting_a_replay(
    memory: AsyncMock,
) -> None:
    other = _A.model_copy(
        update={"id": "c1c2c3c4-0000-0000-0000-000000000000", "content": "at nine"}
    )
    memory.traces_by_id_prefix.side_effect = lambda prefix, user_id=None: [
        t for t in (_A, other) if t.id.startswith(prefix)
    ]
    memory.experience_result.return_value = ExperienceResult(
        trace=_B,
        superseded=[ForgetResult(trace_id=_A.id)],
        unsuperseded=[other.id],
    )
    reply = await remember("moved", _ctx(memory), supersedes=["a1b2c3d4", "c1c2c3c4"])
    lines = reply.splitlines()
    assert lines[0].startswith("Remembered [b1b2c3d4]")
    assert lines[1].startswith("Superseded [a1b2c3d4]")
    assert lines[2] == 'Not superseded [c1c2c3c4] ("at nine"): storage error.'
    assert lines[3] == (
        "This memory is stored; do not store it again. "
        "Run forget on each memory listed as not superseded."
    )
    assert "Storage failed" not in reply


async def test_unstored_fact_is_echoed(memory: AsyncMock) -> None:
    memory.experience_result.return_value = ExperienceResult(
        trace=_B, unstored_facts=[_PINNED]
    )
    reply = await remember("moved", _ctx(memory))
    assert reply.splitlines()[1] == (
        "Persona fact Sarah scheduled_for Monday dentist was not stored: "
        "storage error. The memory itself is stored."
    )
    assert memory.experience_result.await_args.kwargs["supersedes"] == []
