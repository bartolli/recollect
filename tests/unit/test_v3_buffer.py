"""Tests for WorkingMemory buffer.

Focus: capacity enforcement, FIFO displacement, eviction.
"""

from recollect.buffer import WorkingMemory
from recollect.models import MemoryTrace


def _trace(content: str = "test") -> MemoryTrace:
    return MemoryTrace(content=content, pattern={"concepts": [content]})


class TestCapacity:
    def test_default_capacity_is_7(self) -> None:
        wm = WorkingMemory()
        assert wm.capacity == 7

    def test_enforces_minimum_5(self) -> None:
        wm = WorkingMemory(capacity=3)
        assert wm.capacity == 5

    def test_enforces_maximum_9(self) -> None:
        wm = WorkingMemory(capacity=15)
        assert wm.capacity == 9

    def test_accepts_valid_range(self) -> None:
        for cap in (5, 6, 7, 8, 9):
            wm = WorkingMemory(capacity=cap)
            assert wm.capacity == cap


class TestAddAndDisplacement:
    def test_add_returns_none_when_not_full(self) -> None:
        wm = WorkingMemory(capacity=5)
        displaced = wm.add(_trace())
        assert displaced is None

    def test_add_returns_displaced_when_full(self) -> None:
        wm = WorkingMemory(capacity=5)
        first = _trace("first")
        wm.add(first)
        for i in range(4):
            wm.add(_trace(f"item-{i}"))

        # This should displace 'first'
        displaced = wm.add(_trace("overflow"))
        assert displaced is not None
        assert displaced.content == "first"

    def test_displaced_not_in_buffer(self) -> None:
        wm = WorkingMemory(capacity=5)
        first = _trace("first")
        wm.add(first)
        for i in range(5):
            wm.add(_trace(f"item-{i}"))

        active = wm.get_active()
        contents = [t.content for t in active]
        assert "first" not in contents

    def test_buffer_size_never_exceeds_capacity(self) -> None:
        wm = WorkingMemory(capacity=5)
        for i in range(20):
            wm.add(_trace(f"item-{i}"))
        assert len(wm) <= 5

    def test_fifo_displaces_oldest_even_when_strongest(self) -> None:
        """Displacement is recency-based; strength plays no role."""
        wm = WorkingMemory(capacity=5)
        strongest = MemoryTrace(content="strongest", strength=1.0)
        wm.add(strongest)
        for i in range(4):
            wm.add(MemoryTrace(content=f"weak-{i}", strength=0.1))

        displaced = wm.add(MemoryTrace(content="overflow", strength=0.1))
        assert displaced is not None
        assert displaced.content == "strongest"


class TestStats:
    def test_tracks_total_seen(self) -> None:
        wm = WorkingMemory(capacity=5)
        for i in range(10):
            wm.add(_trace(f"item-{i}"))

        stats = wm.get_stats()
        assert stats["total_seen"] == 10

    def test_tracks_total_displaced(self) -> None:
        wm = WorkingMemory(capacity=5)
        for i in range(10):
            wm.add(_trace(f"item-{i}"))

        stats = wm.get_stats()
        assert stats["total_displaced"] == 5


class TestEvict:
    def test_removes_matching_trace(self) -> None:
        wm = WorkingMemory(capacity=5)
        a, b = _trace("a"), _trace("b")
        wm.add(a)
        wm.add(b)
        assert wm.evict(a.id) is True
        assert [t.id for t in wm.get_active()] == [b.id]

    def test_returns_false_when_absent(self) -> None:
        wm = WorkingMemory(capacity=5)
        assert wm.evict("nonexistent") is False
