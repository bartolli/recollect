"""Working memory buffer with 7 +/- 2 capacity limit.

Implements Miller's Law: human short-term memory holds approximately
7 items, with a range of 5 to 9.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from recollect.config import config
from recollect.models import _clamp_strength

if TYPE_CHECKING:
    from recollect.models import MemoryTrace


class WorkingMemory:
    """Limited-capacity active memory buffer."""

    def __init__(self, capacity: int = 7) -> None:
        self.capacity = min(9, max(5, capacity))
        self._buffer: deque[MemoryTrace] = deque(maxlen=self.capacity)
        self._rehearsal_counts: dict[str, int] = {}
        self.total_items_seen = 0
        self.total_displaced = 0

    def add(self, trace: MemoryTrace) -> MemoryTrace | None:
        """Add trace to working memory. Returns displaced trace if buffer was full."""
        displaced: MemoryTrace | None = None

        if len(self._buffer) >= self.capacity:
            displaced = self._buffer[0]
            self.total_displaced += 1
            self._rehearsal_counts.pop(displaced.id, None)

        self._buffer.append(trace)
        self._rehearsal_counts[trace.id] = 0
        self.total_items_seen += 1

        return displaced

    def rehearse(self, trace: MemoryTrace) -> bool:
        """Rehearse an item to keep it active. Returns False if not in buffer."""
        if trace not in self._buffer:
            return False

        self._rehearsal_counts[trace.id] = self._rehearsal_counts.get(trace.id, 0) + 1
        factor = float(config.get("strengthening.rehearsal_factor", 1.05))
        updated = trace.model_copy(
            update={"strength": _clamp_strength(trace.strength * factor)}
        )

        # Move to end (most recent position) with updated strength
        self._buffer.remove(trace)
        self._buffer.append(updated)
        return True

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
        self._rehearsal_counts.clear()

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
