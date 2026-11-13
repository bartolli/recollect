"""Surfacing reads are status-filtered in storage and unbounded.

Wiring pins for sharers the integration suite cannot isolate: the server
helper behind reflect / primer / facts resource / safety net.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import StorageError
from recollect.models import SURFACING_STATUSES, PersonaFact
from recollect_mcp.server import (
    AppContext,
    _generate_primer,
    _safety_net_facts,
    _surfacing_facts,
    reflect,
)


async def test_server_surfacing_read_is_status_filtered_and_unbounded() -> None:
    memory = AsyncMock()
    memory.facts.return_value = []
    app = AppContext(memory=memory, worker=MagicMock(), user_id="alex")
    await _surfacing_facts(app)
    memory.facts.assert_awaited_once_with(
        user_id="alex", statuses=SURFACING_STATUSES, limit=None
    )


async def test_facts_passes_statuses_and_limit_to_storage(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    fact = PersonaFact(
        subject="Alex", predicate="prefers", object="tea", content="x", status="pinned"
    )
    mock_fact_store.get_persona_facts.return_value = [fact]
    mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
    assert await mem.facts(user_id="alex", statuses=SURFACING_STATUSES, limit=None) == [
        fact
    ]
    mock_fact_store.get_persona_facts.assert_awaited_once_with(
        None, user_id="alex", statuses=SURFACING_STATUSES, limit=None
    )


async def test_facts_default_is_a_browse_page(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
    await mem.facts()
    mock_fact_store.get_persona_facts.assert_awaited_once_with(
        None, user_id=None, statuses=None, limit=50
    )


def _ctx(memory: AsyncMock) -> MagicMock:
    fake = MagicMock()
    fake.request_context.lifespan_context = AppContext(
        memory=memory, worker=MagicMock(), user_id="alex"
    )
    fake.headers = None
    return fake


@pytest.mark.parametrize("sharer", ["reflect", "primer", "safety_net"])
async def test_every_server_sharer_reads_through_the_surfacing_read(
    sharer: str,
) -> None:
    # A sharer that goes back to a bare facts() call regains the recency
    # window; the gate-surfaced path would mask it in an end-to-end recall.
    memory = AsyncMock()
    memory.facts.return_value = []
    ctx = _ctx(memory)
    app = ctx.request_context.lifespan_context
    if sharer == "reflect":
        await reflect(ctx)
    elif sharer == "primer":
        await _generate_primer(app)
    else:
        await _safety_net_facts(app)
    for call in memory.facts.await_args_list:
        assert call.kwargs == {
            "user_id": "alex",
            "statuses": SURFACING_STATUSES,
            "limit": None,
        }
    assert memory.facts.await_count >= 1


async def test_health_report_degrades_when_storage_is_unreachable(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_fact_store: AsyncMock,
) -> None:
    # A health read that raises when unhealthy reports nothing.
    mock_fact_store.count_by_status.side_effect = StorageError("down")
    mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
    report = await mem.health_report(user_id="alex")
    assert report.status == "degraded"
    assert report.invariants == ["storage unreachable"]
