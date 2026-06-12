"""Working memory buffer with 7 +/- 2 capacity limit.

Implements Miller's Law: human short-term memory holds approximately
7 items, with a range of 5 to 9. Displacement is FIFO: the oldest item
leaves when the buffer is full, regardless of strength -- the buffer
holds stale strength copies (mutations land in storage), so recency is
the only signal it can displace on honestly.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from recollect.models import MemoryTrace


class WorkingMemory:
    """Limited-capacity active memory buffer."""

    def __init__(self, capacity: int = 7) -> None:
        self.capacity = min(9, max(5, capacity))
        self._buffer: deque[MemoryTrace] = deque(maxlen=self.capacity)
        self.total_items_seen = 0
        self.total_displaced = 0

    def add(self, trace: MemoryTrace) -> MemoryTrace | None:
        """Add trace to working memory. Returns displaced trace if buffer was full."""
        displaced: MemoryTrace | None = None

        if len(self._buffer) >= self.capacity:
            displaced = self._buffer[0]
            self.total_displaced += 1

        self._buffer.append(trace)
        self.total_items_seen += 1

        return displaced

    def get_active(self) -> list[MemoryTrace]:
        """Get all items currently in working memory."""
        return list(self._buffer)

    def find(self, predicate: Any) -> MemoryTrace | None:
        """Find first item matching predicate."""
        for trace in self._buffer:
            if predicate(trace):
                return trace
        return None

    def clear(self) -> None:
        """Clear working memory."""
        self._buffer.clear()

    def evict(self, trace_id: str) -> bool:
        """Remove a trace by id; returns True iff it was in the buffer."""
        for trace in self._buffer:
            if trace.id == trace_id:
                self._buffer.remove(trace)
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get working memory statistics."""
        return {
            "current_items": len(self._buffer),
            "capacity": self.capacity,
            "utilization": len(self._buffer) / self.capacity,
            "total_seen": self.total_items_seen,
            "total_displaced": self.total_displaced,
            "displacement_rate": (self.total_displaced / max(1, self.total_items_seen)),
        }

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"WorkingMemory({len(self._buffer)}/{self.capacity} items)"
