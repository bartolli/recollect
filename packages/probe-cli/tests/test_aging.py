"""Aging: emulated passes step a seeded store to its age."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

from probe_cli.aging import Aging, age_store

Events = list[tuple[str, object]]


def _clock(events: Events) -> MagicMock:
    clock = MagicMock()
    clock.advance_to = AsyncMock(side_effect=lambda o: events.append(("advance", o)))
    clock.settle = AsyncMock(side_effect=lambda: events.append(("settle", None)))
    return clock


def _memory(events: Events) -> MagicMock:
    memory = MagicMock()
    memory.consolidate = AsyncMock(side_effect=lambda: events.append(("pass", None)))
    return memory


async def test_age_zero_shifts_nothing_and_runs_no_pass() -> None:
    events: Events = []

    passes = await age_store(_memory(events), _clock(events), Aging(age_days=0))

    assert passes == 0
    assert events == [("settle", None)]


async def test_each_pass_follows_its_own_advance() -> None:
    events: Events = []
    aging = Aging(age_days=1, passes_per_day=48)

    passes = await age_store(_memory(events), _clock(events), aging)

    assert passes == 48
    expected: Events = []
    for step in range(1, 49):
        expected += [("advance", timedelta(minutes=30) * step), ("pass", None)]
    assert events == [*expected, ("settle", None)]


async def test_zero_passes_per_day_runs_one_closing_pass() -> None:
    events: Events = []
    aging = Aging(age_days=30, passes_per_day=0)

    passes = await age_store(_memory(events), _clock(events), aging)

    assert passes == 1
    assert events == [
        ("advance", timedelta(days=30)),
        ("pass", None),
        ("settle", None),
    ]
