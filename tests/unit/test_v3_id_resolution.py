"""Short-id resolution: unique or refuse, eight characters minimum, scoped
to the server's user. forget does not auto-revive, so it never guesses.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.exceptions import StorageError
from recollect.models import ForgetResult, MemoryTrace, PersonaFact
from recollect_mcp.server import AppContext, forget, pin, unpin

A = "3f2a9c1e-7b4d-4e6a-9c1f-2d8b5a7e4c10"
B = "3f2a9c1e-0000-4000-8000-000000000001"


def _ctx(memory: AsyncMock) -> MagicMock:
    fake = MagicMock()
    fake.request_context.lifespan_context = AppContext(
        memory=memory, worker=MagicMock(), user_id="alex"
    )
    fake.headers = None
    return fake


@pytest.fixture
def memory() -> AsyncMock:
    m = AsyncMock()
    m.traces_by_id_prefix.return_value = [
        MemoryTrace(id=A, content="The climbing session moved to Sunday")
    ]
    m.forget.return_value = ForgetResult(trace_id=A)
    return m


async def test_ambiguous_id_is_refused_with_candidates(memory: AsyncMock) -> None:
    memory.traces_by_id_prefix.return_value = [
        MemoryTrace(id=A, content="The climbing session moved to Sunday"),
        MemoryTrace(id=B, content="Dinner with Sara is on Friday"),
    ]
    result = await forget("3f2a9c1e", _ctx(memory))
    memory.forget.assert_not_awaited()
    assert A in result and B in result
    assert "climbing session" in result and "Dinner with Sara" in result


async def test_id_under_eight_characters_is_refused_unlooked(
    memory: AsyncMock,
) -> None:
    result = await forget("3f2a", _ctx(memory))
    memory.traces_by_id_prefix.assert_not_awaited()
    memory.forget.assert_not_awaited()
    assert "8-character" in result


async def test_lookup_is_scoped_to_the_server_user(memory: AsyncMock) -> None:
    await forget("3f2a9c1e", _ctx(memory))
    memory.traces_by_id_prefix.assert_awaited_once_with("3f2a9c1e", user_id="alex")


async def test_pasted_brackets_and_whitespace_are_tolerated(
    memory: AsyncMock,
) -> None:
    await forget(" [3f2a9c1e] ", _ctx(memory))
    memory.traces_by_id_prefix.assert_awaited_once_with("3f2a9c1e", user_id="alex")
    memory.forget.assert_awaited_once_with(A, force=False)


async def test_no_match_is_not_found(memory: AsyncMock) -> None:
    memory.traces_by_id_prefix.return_value = []
    result = await forget("0a0a0a0a", _ctx(memory))
    memory.forget.assert_not_awaited()
    assert "No memory trace matches id 0a0a0a0a" in result


async def test_lookup_failure_reports_and_forgets_nothing(
    memory: AsyncMock,
) -> None:
    memory.traces_by_id_prefix.side_effect = StorageError("db down")
    result = await forget("3f2a9c1e", _ctx(memory))
    memory.forget.assert_not_awaited()
    assert "db down" in result


async def test_pin_returns_the_refusal_for_an_unresolved_id(
    memory: AsyncMock,
) -> None:
    memory.traces_by_id_prefix.return_value = []
    result = await pin("0a0a0a0a", _ctx(memory))
    memory.pin.assert_not_awaited()
    assert "No memory trace matches id 0a0a0a0a" in result


@pytest.mark.parametrize("tool", [forget, pin, unpin])
async def test_ambiguous_candidates_reach_the_client_not_the_logs(
    tool: Callable[..., Awaitable[str]],
    memory: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Candidate lines quote memory text: reply only, never a log record."""
    memory.traces_by_id_prefix.return_value = [
        MemoryTrace(id=A, content="sentinel-alpha climbing moved to Sunday"),
        MemoryTrace(id=B, content="sentinel-beta dinner is on Friday"),
    ]
    memory.facts_by_id_prefix.return_value = [
        PersonaFact(
            id=A,
            subject="Nadia",
            predicate="prefers",
            object="sentinel-alpha",
            content="Nadia prefers sentinel-alpha",
        ),
        PersonaFact(
            id=B,
            subject="Sara",
            predicate="avoids",
            object="sentinel-beta",
            content="Sara avoids sentinel-beta",
        ),
    ]
    with caplog.at_level(logging.DEBUG):
        result = await tool("3f2a9c1e", _ctx(memory))
    assert "sentinel-alpha" in result and "sentinel-beta" in result
    assert "sentinel" not in caplog.text


async def test_ambiguous_fact_id_is_refused_with_candidates(
    memory: AsyncMock,
) -> None:
    memory.facts_by_id_prefix.return_value = [
        PersonaFact(
            id=A,
            subject="Nadia",
            predicate="prefers",
            object="tea",
            content="Nadia prefers tea",
        ),
        PersonaFact(
            id=B,
            subject="Sara",
            predicate="avoids",
            object="dairy",
            content="Sara avoids dairy",
        ),
    ]
    result = await unpin("3f2a9c1e", _ctx(memory))
    memory.unpin.assert_not_awaited()
    assert A in result and B in result
    assert "Nadia prefers tea" in result and "Sara avoids dairy" in result
