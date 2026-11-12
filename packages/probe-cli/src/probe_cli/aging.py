"""Aging: step a seeded probe store to its age with emulated passes."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from recollect.models import HealthReport

if TYPE_CHECKING:
    from recollect.core import CognitiveMemory

    from probe_cli.store_clock import StoreClock

_DAY = timedelta(days=1)
# One worker at the packaged 1800 s consolidation interval.
_DEFAULT_PASSES_PER_DAY = 48


class Aging(BaseModel):
    age_days: int = Field(ge=0)
    # 0: one closing pass after the full shift -- sufficient only once
    # lifecycle state is clock-invariant.
    passes_per_day: int = Field(default=_DEFAULT_PASSES_PER_DAY, ge=0)

    def arm_name(self, base: str) -> str:
        """Arm identity of the aged run: keys its evidence dir and user_id."""
        name = f"{base}-age{self.age_days}"
        if self.passes_per_day != _DEFAULT_PASSES_PER_DAY:
            name += f"-p{self.passes_per_day}"
        return name

    @property
    def span(self) -> timedelta:
        return _DAY * self.age_days

    def pass_offsets(self) -> list[timedelta]:
        """Emulated instant of each pass, measured from seeding."""
        if self.age_days and not self.passes_per_day:
            return [self.span]
        steps = self.age_days * self.passes_per_day
        return [self.span * step / steps for step in range(1, steps + 1)]


class SweepPoint(BaseModel):
    """One evaluation of a sweep; every point reads the same seeding."""

    aging: Aging
    # Control: the same groups and facts, read with the token hop disabled.
    tokens_off: bool = False

    def arm_name(self, base: str) -> str:
        name = self.aging.arm_name(base)
        return f"{name}-tokens-off" if self.tokens_off else name


class AgingReport(BaseModel):
    age_days: int
    passes_per_day: int
    passes_run: int
    seeding_user_id: str
    # Names one seeding; the user_id repeats across sweeps of an arm.
    seeding_session_id: str
    seeded_traces: int
    # Batch window the passes read through; sized to the seeding so every
    # pass reads every pending trace.
    consolidation_batch_size: int
    # Emulated time past the target age when evaluation starts: the final
    # shift, the settle, and reconnection are not absorbed by the clock.
    evaluation_skew_seconds: float
    recall_tokens_at_evaluation: bool
    # Lifecycle state at evaluation start, before any question writes.
    health: HealthReport


async def age_store(memory: CognitiveMemory, clock: StoreClock, aging: Aging) -> int:
    """Age the store; returns the number of emulated passes run."""
    offsets = aging.pass_offsets()
    for offset in offsets:
        # Current code decides trace fate at the first pass that sees a
        # trace and decays recall tokens per pass: each pass needs its
        # own emulated instant, not one shift followed by N passes.
        await clock.advance_to(offset)
        await memory.consolidate()
    await clock.settle()
    return len(offsets)
