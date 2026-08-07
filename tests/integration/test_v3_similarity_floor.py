"""Integration: retrieval.trace_similarity_threshold floors storage search.

ORDER BY distance + LIMIT returns top-K regardless of absolute similarity;
the floor gates candidates when > 0 and is byte-identical to prior behavior
at the packaged default 0 -- sub-zero-similarity candidates included.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from recollect.config import config
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [pytest.mark.slow, pytest.mark.asyncio]

DIM = 768


def _unit(*components: tuple[int, float]) -> list[float]:
    vec = [0.0] * DIM
    for idx, val in components:
        vec[idx] = val
    return vec


def _scratch_url(db_name: str) -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def _admin_url() -> str:
    parsed = urlparse(DB_URL)
    return urlunparse(parsed._replace(path="/postgres"))


@pytest.fixture()
async def scratch_db() -> AsyncGenerator[str, None]:
    db_name = f"simfloor_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(_admin_url())
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    try:
        yield _scratch_url(db_name)
    finally:
        admin = await asyncpg.connect(_admin_url())
        try:
            await admin.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


@pytest.fixture()
async def mem(scratch_db: str) -> AsyncGenerator[CognitiveMemory, None]:
    m = CognitiveMemory()
    await m.connect(scratch_db)
    try:
        yield m
    finally:
        await m.close()


async def _seed(mem: CognitiveMemory) -> dict[str, str]:
    """Four traces at known cosine similarities to the idx-0 unit query."""
    vectors = {
        "aligned": _unit((0, 1.0)),  # sim 1.0
        "diagonal": _unit((0, 0.7071), (1, 0.7071)),  # sim ~0.707
        "orthogonal": _unit((1, 1.0)),  # sim 0.0
        "opposed": _unit((0, -1.0)),  # sim -1.0
    }
    ids: dict[str, str] = {}
    for name, vec in vectors.items():
        trace = MemoryTrace(content=name, embedding=vec, strength=0.8)
        await mem.storage.traces.store_trace(trace)
        ids[trace.id] = name
    return ids


class TestTraceSimilarityFloor:
    async def test_default_zero_preserves_topk(self, mem: CognitiveMemory) -> None:
        """Packaged default returns top-K unfloored, sub-zero sims included."""
        ids = await _seed(mem)
        results = await mem.storage.vectors.search_semantic(_unit((0, 1.0)), 10)
        names = {ids[trace.id] for trace, _ in results}
        assert names == {"aligned", "diagonal", "orthogonal", "opposed"}

    async def test_floor_excludes_below(self, mem: CognitiveMemory) -> None:
        """Floor 0.5 keeps sim {1.0, 0.707}, drops {0.0, -1.0}."""
        ids = await _seed(mem)
        config._set("retrieval.trace_similarity_threshold", 0.5)
        try:
            results = await mem.storage.vectors.search_semantic(_unit((0, 1.0)), 10)
        finally:
            config._set("retrieval.trace_similarity_threshold", 0.0)
        names = {ids[trace.id] for trace, _ in results}
        assert names == {"aligned", "diagonal"}
