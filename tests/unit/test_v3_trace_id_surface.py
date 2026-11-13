"""Trace-id surface contract: the short ids recall and remember print
round-trip through forget and pin on tool output alone.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.models import ForgetResult, MemoryTrace, Session, Thought
from recollect_mcp.server import AppContext, forget, pin, recall, remember

FULL = "3f2a9c1e-7b4d-4e6a-9c1f-2d8b5a7e4c10"
_RECALL_LINE_ID = re.compile(r"^- \[([0-9a-f]{8})\] ", re.MULTILINE)
_REMEMBERED_ID = re.compile(r"Remembered \[([0-9a-f]{8})\]")


def _ctx(memory: AsyncMock) -> MagicMock:
    fake = MagicMock()
    fake.request_context.lifespan_context = AppContext(
        memory=memory, worker=MagicMock(), user_id="alex"
    )
    fake.headers = None
    return fake


@pytest.fixture
def trace() -> MemoryTrace:
    return MemoryTrace(id=FULL, content="The climbing session moved to Sunday")


@pytest.fixture
def memory(trace: MemoryTrace) -> AsyncMock:
    m = AsyncMock()
    m.traces_by_id_prefix.side_effect = lambda prefix, user_id=None: (
        [trace] if trace.id.startswith(prefix) else []
    )
    m.experience.return_value = trace
    m.surface_relevant_facts.return_value = []
    m.think_about.return_value = [
        Thought(trace=trace, relevance=0.8, token_count=9, reconstruction=trace.content)
    ]
    m.facts.return_value = []
    m.forget.return_value = ForgetResult(trace_id=FULL)
    m.pin.return_value = []
    m.start_session.return_value = Session(id="s1", user_id="alex")
    return m


async def _id_from_recall(ctx: MagicMock) -> str:
    printed = _RECALL_LINE_ID.search(await recall("when is climbing?", ctx))
    assert printed, "recall prints no id on the memory line"
    return printed.group(1)


async def _id_from_remember(ctx: MagicMock) -> str:
    printed = _REMEMBERED_ID.search(await remember("climbing moved", ctx))
    assert printed, "remember prints no short id"
    return printed.group(1)


_PRINTERS = [_id_from_recall, _id_from_remember]
Printer = Callable[[MagicMock], Awaitable[str]]


@pytest.mark.parametrize("printer", _PRINTERS)
async def test_printed_id_reaches_forget(printer: Printer, memory: AsyncMock) -> None:
    ctx = _ctx(memory)

    result = await forget(await printer(ctx), ctx)

    memory.forget.assert_awaited_once_with(FULL, force=False)
    assert FULL in result
    assert "climbing session" in result  # a wrong hit is visible in the reply


@pytest.mark.parametrize("printer", _PRINTERS)
async def test_printed_id_reaches_pin(printer: Printer, memory: AsyncMock) -> None:
    ctx = _ctx(memory)

    await pin(await printer(ctx), ctx)

    memory.pin.assert_awaited_once_with(FULL)


async def test_printers_never_emit_the_full_trace_id(memory: AsyncMock) -> None:
    ctx = _ctx(memory)
    assert FULL not in await remember("climbing moved", ctx)
    assert FULL not in await recall("when is climbing?", ctx)


async def test_full_id_still_resolves(memory: AsyncMock) -> None:
    await forget(FULL, _ctx(memory))
    memory.forget.assert_awaited_once_with(FULL, force=False)


async def test_persona_fact_line_carries_no_trace_id(memory: AsyncMock) -> None:
    """A fact thought rides a synthetic trace: its id names no stored row."""
    memory.think_about.return_value = [
        Thought(
            trace=MemoryTrace(
                content="[IMPORTANT CONTEXT] Nadia prefers Sunday climbing",
                pattern={"persona_fact": True, "category": "preference"},
            ),
            relevance=0.9,
            token_count=8,
            reconstruction="[IMPORTANT CONTEXT] Nadia prefers Sunday climbing",
        )
    ]
    result = await recall("climbing?", _ctx(memory))
    assert "- [score: 0.90" in result
    assert not _RECALL_LINE_ID.search(result)
