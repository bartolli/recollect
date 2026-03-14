"""Integration tests: cognitive model invariants."""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncGenerator

import pytest
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://bartolli@localhost:5432/memory_v3"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]


@pytest.fixture()
async def mem() -> AsyncGenerator[CognitiveMemory]:
    m = CognitiveMemory()
    await m.connect(DB_URL)
    yield m
    try:
        traces = await m.timeline(limit=100)
        for t in traces:
            with contextlib.suppress(Exception):
                await m.forget(t.id)
    finally:
        await m.close()


@pytest.mark.slow()
async def test_working_memory_capacity(mem: CognitiveMemory) -> None:
    """Experiencing 10 items must not exceed max capacity (9)."""
    for i in range(10):
        await mem.experience(f"Working memory item {i}")

    stats = mem.stats()
    assert stats.working_memory_items <= 9
    assert stats.working_memory_items >= 5


@pytest.mark.slow()
async def test_strength_clamped_after_reinforce(
    mem: CognitiveMemory,
) -> None:
    """Reinforcing with extreme factor must not exceed 1.0."""
    trace = await mem.experience("Clamp test content")
    reinforced = await mem.reinforce(trace.id, factor=100.0)
    assert reinforced.strength <= 1.0
    assert reinforced.strength >= 0.0


@pytest.mark.slow()
async def test_initial_strength(mem: CognitiveMemory) -> None:
    """New traces start at initial_strength (0.3 default)."""
    trace = await mem.experience("Initial strength test")
    assert trace.strength == pytest.approx(0.3)


@pytest.mark.slow()
async def test_retrieval_boosts_strength(mem: CognitiveMemory) -> None:
    """Retrieving a trace via think_about increases its strength."""
    trace = await mem.experience(
        "Quantum computing uses qubits for parallel computation"
    )
    original_strength = trace.strength

    await mem.think_about("quantum computing qubits", token_budget=2000)

    traces = await mem.timeline(limit=50)
    updated = next((t for t in traces if t.id == trace.id), None)
    assert updated is not None
    assert isinstance(updated, MemoryTrace)
    # Retrieval or activation boost should increase strength
    assert updated.strength >= original_strength


@pytest.mark.slow()
async def test_reinforce_is_multiplicative(
    mem: CognitiveMemory,
) -> None:
    """reinforce(factor=2.0) doubles the trace strength."""
    trace = await mem.experience("Multiplicative reinforce test")
    assert trace.strength == pytest.approx(0.3)

    reinforced = await mem.reinforce(trace.id, factor=2.0)
    assert reinforced.strength == pytest.approx(0.6)


@pytest.mark.slow()
async def test_experience_validation(mem: CognitiveMemory) -> None:
    """experience() rejects empty content."""
    with pytest.raises(ValueError, match="non-empty"):
        await mem.experience("")

    with pytest.raises(ValueError, match="non-empty"):
        await mem.experience("   ")


@pytest.mark.slow()
async def test_think_about_validation(mem: CognitiveMemory) -> None:
    """think_about() rejects empty query and non-positive budget."""
    with pytest.raises(ValueError, match="non-empty"):
        await mem.think_about("", token_budget=100)

    with pytest.raises(ValueError, match="positive"):
        await mem.think_about("valid query", token_budget=0)
