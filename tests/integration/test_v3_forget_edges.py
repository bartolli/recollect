"""Integration: forget reaches archived traces and stays out of assessment."""

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
from recollect.llm.types import ExtractionResult, TokenAssessment

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)
USER = "u1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]


def _url(db_name: str) -> str:
    return urlunparse(urlparse(DB_URL)._replace(path=f"/{db_name}"))


@pytest.fixture()
async def mem() -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"forget_edges_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_url("postgres"))
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    extractor = AsyncMock()
    extractor.extract = AsyncMock(return_value=ExtractionResult())
    extractor._provider.complete_structured = AsyncMock(
        return_value=TokenAssessment(action="none")
    )
    cfg = MemoryConfig()
    # Seeding stays assessment-free; assess_situational bypasses the flag.
    cfg._set("recall_tokens.enabled", False)
    cfg._set("recall_tokens.write_time_threshold", 0.0)
    cfg._set("recall_tokens.write_time_anchor_k", 0)
    # The new trace matches itself, so top_k=2 leaves exactly one real slot.
    cfg._set("recall_tokens.write_time_top_k", 2)
    m = CognitiveMemory(extractor=extractor, config=cfg)
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


async def test_forget_accepts_an_archived_trace(mem: CognitiveMemory) -> None:
    content = "Maria's surgery is scheduled for Tuesday at CityMed"
    trace = await mem.experience(content, user_id=USER)
    assert await mem.storage.traces.archive_trace(trace.id)

    await mem.forget(trace.id)

    thoughts = await mem.think_about(content, user_id=USER)
    assert trace.id not in [t.trace.id for t in thoughts]
    stored = await mem.storage.traces.get_trace(trace.id)
    assert stored is not None
    assert stored.status == "forgotten"


async def test_forgotten_trace_is_not_an_assessment_candidate(
    mem: CognitiveMemory,
) -> None:
    closest = await mem.experience(
        "Maria's surgery is scheduled for Tuesday at CityMed", user_id=USER
    )
    runner_up = await mem.experience(
        "Maria needs a ride to the hospital next week", user_id=USER
    )
    await mem.forget(closest.id)
    new = await mem.experience(
        "Maria's surgery was moved to Thursday at CityMed", user_id=USER
    )

    assessment = await mem.assess_situational(new)

    # Excluded in SQL: the forgotten trace frees its slot for the runner-up.
    # A filter after the LIMIT would leave the candidate list empty.
    assert assessment is not None
    assert assessment.related_trace_ids == [runner_up.id]
