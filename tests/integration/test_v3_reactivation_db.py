"""Integration: reactivation matrix against live PG.

Semantic e2e revival; deep-decayed (<0.1) invisibility to the semantic
path; spread CTE carries status (archived rows must not masquerade as
active); idempotent revival; consolidated=FALSE reset; revival survives
the next consolidation pass via the recency-anchored grace.
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.core import CognitiveMemory
from recollect.datetime_utils import now_utc
from recollect.models import Association, MemoryTrace

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [pytest.mark.slow, pytest.mark.asyncio]


def _scratch_url(db_name: str) -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def _admin_url() -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path="/postgres"))


@pytest.fixture()
async def mem() -> AsyncGenerator[CognitiveMemory, None]:
    db_name = f"reactivation_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    m = CognitiveMemory()
    await m.connect(_scratch_url(db_name))
    try:
        yield m
    finally:
        await m.close()
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


async def _archived_trace(
    mem: CognitiveMemory,
    text: str,
    *,
    strength: float,
    significance: float = 0.1,
) -> MemoryTrace:
    vec = await mem._embeddings.generate_embedding(text)
    trace = MemoryTrace(
        content=text,
        embedding=vec,
        strength=strength,
        significance=significance,
        status="archived",
    )
    await mem.storage.traces.store_trace(trace)
    return trace


class TestReactivationMatrix:
    async def test_semantic_revival_end_to_end(self, mem: CognitiveMemory) -> None:
        trace = await _archived_trace(
            mem, "the tax filing deadline moved to May", strength=0.5
        )
        await mem.think_about("when is the tax filing deadline")
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.status == "active"
        assert stored.consolidated is False

    async def test_deep_decayed_invisible_to_semantic_stays_archived(
        self, mem: CognitiveMemory
    ) -> None:
        # strength 0.05 < selection_threshold 0.1: the bi-encoder cannot
        # surface it, and no other channel exists here -- accepted v1 limit.
        trace = await _archived_trace(
            mem, "the tax filing deadline moved to May", strength=0.05
        )
        await mem.think_about("when is the tax filing deadline")
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.status == "archived"

    async def test_spread_carries_archived_status(
        self, mem: CognitiveMemory
    ) -> None:
        # D2 regression: the CTE's explicit column list must SELECT
        # status, else row_to_trace defaults archived rows to active.
        seed = MemoryTrace(content="seed", strength=0.5)
        neighbor = MemoryTrace(
            content="neighbor", strength=0.05, status="archived"
        )
        await mem.storage.traces.store_trace(seed)
        await mem.storage.traces.store_trace(neighbor)
        await mem.storage.associations.store_association(
            Association(
                source_trace_id=seed.id,
                target_trace_id=neighbor.id,
                association_type="semantic",
                weight=0.9,
                forward_strength=0.9,
                backward_strength=0.9,
            )
        )
        spread = await mem.storage.vectors.spread_activation(seed.id)
        by_id = {t.id: t for t, _ in spread}
        assert neighbor.id in by_id
        assert by_id[neighbor.id].status == "archived"

    async def test_reactivate_refuses_forgotten(self, mem: CognitiveMemory) -> None:
        trace = MemoryTrace(content="retracted", strength=0.5)
        await mem.storage.traces.store_trace(trace)
        await mem.storage.traces.forget_trace(trace.id)
        assert (
            await mem.storage.traces.reactivate_trace(
                trace.id, min_strength=0.6, activated_at=now_utc()
            )
            is False
        )
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.status == "forgotten"

    async def test_reactivate_idempotent_and_resets(
        self, mem: CognitiveMemory
    ) -> None:
        trace = MemoryTrace(content="ratchet", strength=0.05, significance=0.6)
        await mem.storage.traces.store_trace(trace)
        await mem.storage.traces.mark_consolidated(trace.id)
        assert await mem.storage.traces.archive_trace(trace.id) is True

        first = await mem.storage.traces.reactivate_trace(
            trace.id, min_strength=0.6, activated_at=now_utc()
        )
        second = await mem.storage.traces.reactivate_trace(
            trace.id, min_strength=0.6, activated_at=now_utc()
        )
        assert first is True
        assert second is False
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.status == "active"
        assert stored.consolidated is False
        assert stored.strength == pytest.approx(0.6)

    async def test_revival_survives_next_consolidation(
        self, mem: CognitiveMemory
    ) -> None:
        # Cross-story contract: without the story-2 recency anchor, a
        # revived old weak trace is past-grace and re-archives next pass.
        trace = MemoryTrace(
            content="revived survivor",
            strength=0.05,
            significance=0.1,
            status="archived",
            created_at=now_utc() - timedelta(hours=48),
        )
        await mem.storage.traces.store_trace(trace)
        assert await mem.storage.traces.reactivate_trace(
            trace.id, min_strength=0.1, activated_at=now_utc()
        )
        result = await mem.consolidate()
        assert result.forgotten == 0
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.status == "active"
