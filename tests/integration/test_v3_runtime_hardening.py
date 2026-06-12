"""Integration: runtime hardening -- atomic strength factors, NULL-embedding guard.

Concurrent factors must compose multiplicatively (no last-writer-wins);
a NULL-embedding row must be excluded from search_semantic, not crash it.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace

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
    db_name = f"hardening_{uuid.uuid4().hex[:10]}"
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


class TestAtomicStrengthFactors:
    async def test_concurrent_factors_compose(self, mem: CognitiveMemory) -> None:
        """Factors commute: decay + boost land regardless of interleaving."""
        trace = MemoryTrace(content="contended", strength=0.5)
        await mem.storage.traces.store_trace(trace)
        await asyncio.gather(
            mem.storage.traces.apply_strength_factor(trace.id, 1.2),
            mem.storage.traces.apply_strength_factor(trace.id, 0.8),
        )
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.strength == pytest.approx(0.5 * 1.2 * 0.8)

    async def test_factor_clamps_in_sql(self, mem: CognitiveMemory) -> None:
        trace = MemoryTrace(content="strong", strength=0.9)
        await mem.storage.traces.store_trace(trace)
        await mem.storage.traces.apply_strength_factor(trace.id, 1.5)
        stored = await mem.storage.traces.get_trace(trace.id)
        assert stored is not None
        assert stored.strength == 1.0


class TestNullEmbeddingGuard:
    async def test_null_embedding_excluded_not_crashing(
        self, mem: CognitiveMemory
    ) -> None:
        vec = await mem._embeddings.generate_embedding("has vector")
        embedded = MemoryTrace(content="has vector", embedding=vec)
        bare = MemoryTrace(content="no vector", embedding=None)
        await mem.storage.traces.store_trace(embedded)
        await mem.storage.traces.store_trace(bare)
        query = await mem._embeddings.generate_embedding(
            "has vector", task="search_query"
        )
        results = await mem.storage.vectors.search_semantic(query, limit=10)
        ids = {t.id for t, _ in results}
        assert embedded.id in ids
        assert bare.id not in ids
