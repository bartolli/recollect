"""Background consolidation worker.

Periodically calls CognitiveMemory.consolidate() to strengthen
or forget traces based on the cognitive model.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recollect.core import CognitiveMemory

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 1800.0  # 30 minutes


class ConsolidationWorker:
    """Background worker that runs consolidation on an interval."""

    def __init__(
        self,
        memory: CognitiveMemory,
        interval_seconds: float = _DEFAULT_INTERVAL,
    ) -> None:
        self._memory = memory
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the consolidation loop. Safe to call multiple times."""
        if self.running:
            return
        self._task = asyncio.get_event_loop().create_task(self._loop())
        logger.info("Consolidation worker started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        """Stop the consolidation loop. Safe to call multiple times."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _loop(self) -> None:
        """Run consolidation on interval. Never crash."""
        while True:
            await asyncio.sleep(self._interval)
            try:
                result = await self._memory.consolidate()
                logger.info(
                    "Consolidation: %d consolidated, %d forgotten, %d pending",
                    result.consolidated,
                    result.forgotten,
                    result.still_pending,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Consolidation failed")
