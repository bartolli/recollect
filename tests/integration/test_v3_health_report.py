"""Integration: the health report states lifecycle truth, not just liveness."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import ExtractionResult
from recollect.models import FactStatus, PersonaFact, RecallToken

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]


def _url(db_name: str) -> str:
    return urlunparse(urlparse(DB_URL)._replace(path=f"/{db_name}"))


@pytest.fixture()
def extractor() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
async def mem(extractor: AsyncMock) -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"health_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_url("postgres"))
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    m = CognitiveMemory(extractor=extractor, config=MemoryConfig())
    await m.connect(_url(db_name))
    try:
        yield m
    finally:
        await m.close()
        admin = await asyncpg.connect(_url("postgres"))
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


async def _fact(mem: CognitiveMemory, obj: str, status: FactStatus, user: str) -> None:
    await mem.storage.facts.store_persona_fact(
        PersonaFact(
            subject="Alex",
            predicate="is_interested_in",
            object=obj,
            content=f"Alex is interested in {obj}",
            status=status,
            user_id=user,
        )
    )


async def _token(mem: CognitiveMemory, strength: float) -> None:
    await mem.storage.recall_tokens.create_token(
        RecallToken(label="Alex | move | new city", strength=strength)
    )


async def test_health_report_counts_and_invariants(
    mem: CognitiveMemory, extractor: AsyncMock
) -> None:
    extractor.extract = AsyncMock(
        side_effect=[
            ExtractionResult(concepts=["relocation"]),
            ExtractionResult(),
            ExtractionResult(concepts=["gardening"]),
        ]
    )
    await mem.experience("Alex moved to Lisbon", user_id="u1")
    await mem.experience("extraction came back empty", user_id="u1")
    faded = await mem.experience("Alex planted tomatoes", user_id="u1")
    await mem.storage.traces.archive_trace(faded.id)
    await _fact(mem, "sailing", "pinned", "u1")
    await _fact(mem, "chess", "promoted", "u1")
    await _fact(mem, "opera", "candidate", "u1")
    await _token(mem, 1.0)
    await _token(mem, 0.01)
    await mem.start_session(user_id="u1")

    report = await mem.health_report(user_id="u1")

    assert report.status == "ok"
    assert report.traces == {"active": 2, "archived": 1}
    assert report.degraded_traces == 1
    assert report.facts == {"pinned": 1, "promoted": 1, "candidate": 1}
    assert (report.surfacing_total, report.surfacing_reachable) == (2, 2)
    assert (report.tokens_propagating, report.tokens_dormant) == (1, 1)
    assert report.sessions == {"active": 1}
    assert report.invariants == []


async def test_health_report_flags_an_inert_token_layer(
    mem: CognitiveMemory,
) -> None:
    await _token(mem, 0.01)
    await _token(mem, 0.01)

    report = await mem.health_report(user_id="u1")

    assert report.invariants == ["no propagating recall tokens: 2 dormant"]


async def test_health_report_is_user_scoped(mem: CognitiveMemory) -> None:
    await _fact(mem, "sailing", "pinned", "u1")
    await _fact(mem, "chess", "pinned", "u2")
    await mem.start_session(user_id="u2")

    report = await mem.health_report(user_id="u1")

    assert report.facts == {"pinned": 1}
    assert report.surfacing_total == 1
    assert report.sessions == {}
