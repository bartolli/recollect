"""Integration: reactivation matrix against live PG.

Semantic e2e revival; deep-decayed (<0.1) invisibility to the semantic
path; spread CTE carries status (archived rows must not masquerade as
active); idempotent revival; consolidated=FALSE reset; revival survives
the next consolidation pass via the recency-anchored grace.
"""

from __future__ import annotations

import copy
import os
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.config import config as sdk_config
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

    async def test_deep_decayed_revives_on_relevant_recall(
        self, mem: CognitiveMemory
    ) -> None:
        # No strength gate at the packaged default: a deep-decayed archived
        # trace is a semantic candidate and the reactivation floor decides.
        trace = await _archived_trace(
            mem, "the tax filing deadline moved to May", strength=0.05
        )
        await mem.think_about("when is the tax filing deadline")
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.status == "active"

    async def test_raised_strength_gate_hides_deep_decayed(
        self, mem: CognitiveMemory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # search_semantic reads the process config. Gate raised above the
        # trace's strength: the bi-encoder cannot surface it, and no other
        # channel exists here.
        monkeypatch.setattr(sdk_config, "_config", copy.deepcopy(sdk_config._config))
        sdk_config._set("retrieval.selection_threshold", 0.1)
        trace = await _archived_trace(
            mem, "the tax filing deadline moved to May", strength=0.05
        )
        await mem.think_about("when is the tax filing deadline")
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.status == "archived"

    async def test_archived_below_floor_is_delivered_unrevived(
        self, mem: CognitiveMemory
    ) -> None:
        # With no strength gate an archived trace is a semantic candidate
        # at any relevance; under the reactivation floor it surfaces and
        # the read leaves it archived.
        query = "when is the tax filing deadline"
        q = await mem._embeddings.generate_embedding(query, task="search_query")
        # Pairwise rotation: orthogonal to q with q's norm, so the mix
        # below has cosine 0.2 to the query.
        ortho = [0.0] * len(q)
        ortho[0::2] = [-x for x in q[1::2]]
        ortho[1::2] = q[0::2]
        mixed = [0.2 * a + (0.96**0.5) * b for a, b in zip(q, ortho, strict=True)]
        trace = MemoryTrace(
            content="an unrelated faded note",
            embedding=mixed,
            strength=0.05,
            status="archived",
        )
        await mem.storage.traces.store_trace(trace)
        thoughts = await mem.think_about(query)
        delivered = {t.trace.id: t.relevance for t in thoughts}
        assert trace.id in delivered
        assert delivered[trace.id] < 0.3
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
