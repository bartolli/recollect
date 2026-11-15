"""Fact-id surface contract: the short ids reflect, pin, and the remember
context block print round-trip through unpin on tool output alone.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.models import (
    ExperienceResult,
    MemoryTrace,
    PersonaFact,
    Session,
    SurfacedFacts,
)
from recollect_mcp.server import (
    AppContext,
    _format_facts,
    _format_pin_result,
    reflect,
    remember,
    unpin,
)

FULL = "7c1d4e9a-2b6f-4a3c-8d5e-9f0a1b2c3d4e"
_HEADER_ID = re.compile(r"^\[([0-9a-f]{8})\] ", re.MULTILINE)


def _ctx(memory: AsyncMock) -> MagicMock:
    fake = MagicMock()
    fake.request_context.lifespan_context = AppContext(
        memory=memory, worker=MagicMock(), user_id="alex"
    )
    fake.headers = None
    return fake


@pytest.fixture
def fact() -> PersonaFact:
    return PersonaFact(
        id=FULL,
        subject="Nadia",
        predicate="scheduled_for",
        object="Sunday climbing",
        content="Nadia moved the climbing session to Sunday",
        category="schedule",
        status="pinned",
    )


@pytest.fixture
def memory(fact: PersonaFact) -> AsyncMock:
    m = AsyncMock()
    m.facts_by_id_prefix.side_effect = lambda prefix, user_id=None: (
        [fact] if fact.id.startswith(prefix) else []
    )
    m.facts.return_value = [fact]
    m.unpin.return_value = True
    m.experience_result.return_value = ExperienceResult(
        trace=MemoryTrace(content="climbing moved")
    )
    m.surface_relevant_facts_result.return_value = SurfacedFacts(facts=[fact])
    m.start_session.return_value = Session(id="s1", user_id="alex")
    return m


def _printed_id(text: str) -> str:
    match = _HEADER_ID.search(text)
    assert match, text
    return match.group(1)


async def _unpin_printed(memory: AsyncMock, text: str) -> str:
    assert FULL not in text
    return await unpin(_printed_id(text), _ctx(memory))


async def test_pin_result_id_reaches_unpin(
    memory: AsyncMock, fact: PersonaFact
) -> None:
    result = await _unpin_printed(memory, _format_pin_result([fact]))
    memory.unpin.assert_awaited_once_with(FULL)
    assert FULL in result
    assert "Nadia scheduled_for Sunday climbing" in result


async def test_facts_resource_listing_id_reaches_unpin(
    memory: AsyncMock, fact: PersonaFact
) -> None:
    """`memory://facts` is `_format_facts` over the surfacing read."""
    await _unpin_printed(memory, _format_facts([fact]))
    memory.unpin.assert_awaited_once_with(FULL)


async def test_reflect_listing_id_reaches_unpin(memory: AsyncMock) -> None:
    listing = await reflect(_ctx(memory))
    await _unpin_printed(memory, listing)
    memory.unpin.assert_awaited_once_with(FULL)


async def test_remember_context_block_id_reaches_unpin(memory: AsyncMock) -> None:
    block = await remember("climbing moved", _ctx(memory))
    await _unpin_printed(memory, block.split("IMPORTANT CONTEXT:")[1])
    memory.unpin.assert_awaited_once_with(FULL)


async def test_unknown_id_is_not_found(memory: AsyncMock) -> None:
    result = await unpin("0a0a0a0a", _ctx(memory))
    assert "not found" in result.lower()
    memory.unpin.assert_not_awaited()
