"""Fact-id surface contract: MCP-printed ids round-trip through unpin."""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

from recollect.core import CognitiveMemory
from recollect.models import PersonaFact
from recollect_mcp.server import _format_facts, _format_pin_result

_HEADER_ID = re.compile(r"\[([0-9a-f-]+)\]")


def _fact() -> PersonaFact:
    return PersonaFact(
        subject="Nadia",
        predicate="scheduled_for",
        object="Sunday climbing",
        content="Nadia moved the climbing session to Sunday",
        category="schedule",
        status="pinned",
    )


def _printed_id(formatted: str) -> str:
    match = _HEADER_ID.search(formatted)
    assert match, formatted
    return match.group(1)


class TestPinIdRoundTrip:
    async def test_pin_result_id_reaches_unpin_intact(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # The id an MCP client reads from the pin result must be the id the
        # store matches on -- update_fact_status is exact-match by contract.
        fact = _fact()
        printed = _printed_id(_format_pin_result([fact]))
        mock_fact_store.update_fact_status.return_value = True
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        assert await mem.unpin(printed) is True
        mock_fact_store.update_fact_status.assert_awaited_once_with(
            fact.id, "archived"
        )

    async def test_reflect_listing_id_reaches_unpin_intact(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # reflect and memory://facts render through _format_facts; per-surface
        # guard against a future formatter split re-truncating one path.
        fact = _fact()
        printed = _printed_id(_format_facts([fact]))
        mock_fact_store.update_fact_status.return_value = True
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        assert await mem.unpin(printed) is True
        mock_fact_store.update_fact_status.assert_awaited_once_with(
            fact.id, "archived"
        )

    async def test_unknown_id_is_not_found_verbatim(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        # No prefix or partial matching: the store sees the exact input and
        # a 0-row UPDATE stays not-found.
        mock_fact_store.update_fact_status.return_value = False
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        assert await mem.unpin("no-such-id") is False
        mock_fact_store.update_fact_status.assert_awaited_once_with(
            "no-such-id", "archived"
        )
