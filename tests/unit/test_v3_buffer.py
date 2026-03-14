"""Tests for WorkingMemory buffer.

Focus: capacity enforcement, displacement tracking, rehearsal mechanics.
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


class TestRehearsal:
    def test_rehearse_existing_item(self) -> None:
        wm = WorkingMemory()
        t = _trace("rehearse-me")
        wm.add(t)
        original_strength = t.strength
        assert wm.rehearse(t) is True
        # rehearse() uses model_copy, so the updated trace is in the buffer
        active = wm.get_active()
        updated = active[-1]  # rehearsed item moved to end
        assert updated.strength > original_strength

    def test_rehearse_moves_to_end(self) -> None:
        wm = WorkingMemory()
        first = _trace("first")
        wm.add(first)
        wm.add(_trace("second"))
        wm.add(_trace("third"))
        wm.rehearse(first)

        # 'first' should now be last (most recent position)
        active = wm.get_active()
        assert active[-1].content == "first"

    def test_rehearse_nonexistent_returns_false(self) -> None:
        wm = WorkingMemory()
        t = _trace("not-in-buffer")
        assert wm.rehearse(t) is False


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
