"""Aged task runs: no in-process state crosses the shift."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli.aging import Aging
from probe_cli.task_runner import TaskArmRunner
from recollect.config import MemoryConfig
from recollect.models import HealthReport

if TYPE_CHECKING:
    from pathlib import Path

Log = list[tuple[str, int]]
_CLOCK = -1


class _FakeMemory:
    """Stands in for the SDK; every call lands on one shared log."""

    def __init__(self, log: Log, serial: int) -> None:
        self._log = log
        self._serial = serial
        self.storage = MagicMock()
        self.storage.pool.get_pool = AsyncMock(return_value=MagicMock())
        self.storage.facts.get_persona_facts = AsyncMock(return_value=[])

    def _note(self, event: str) -> None:
        self._log.append((event, self._serial))

    async def connect(self, db_url: str | None = None) -> None:
        self._note("connect")

    async def close(self) -> None:
        self._note("close")

    async def experience(self, text: str, **_: object) -> MagicMock:
        self._note("experience")
        return MagicMock(id=f"id-{text}")

    async def consolidate(self) -> None:
        self._note("pass")

    async def think_about(self, query: str, **_: object) -> list[Any]:
        self._note("think")
        return []

    async def health_report(self, *, user_id: str | None = None) -> HealthReport:
        self._note("health")
        return HealthReport(status="ok", tokens_dormant=4)


@pytest.fixture()
def log(monkeypatch: pytest.MonkeyPatch) -> Log:
    events: Log = []
    serials = iter(range(100))
    clock = MagicMock()
    clock.advance_to = AsyncMock(
        side_effect=lambda _offset: events.append(("advance", _CLOCK))
    )
    clock.settle = AsyncMock(side_effect=lambda: events.append(("settle", _CLOCK)))
    clock.emulated_offset = MagicMock(return_value=timedelta(days=1, seconds=9))
    monkeypatch.setattr(
        "probe_cli.task_runner.CognitiveMemory",
        lambda **_: _FakeMemory(events, next(serials)),
    )
    monkeypatch.setattr("probe_cli.task_runner.StoreClock", lambda _pool: clock)
    return events


def _runner(
    tmp_path: Path, aging: Aging | None, config: MemoryConfig | None = None
) -> TaskArmRunner:
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(
        "\n".join(
            json.dumps({"id": f"s{i}", "text": f"trace {i}", "density_tier": 0})
            for i in range(2)
        ),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "when?",
                "answers": ["noon"],
                "answer_type": "time",
                "tier_label": "any",
            }
        ),
        encoding="utf-8",
    )
    answer = MagicMock(model_name="answer")
    answer.complete = AsyncMock(return_value="unknown")
    return TaskArmRunner(
        arm_name="test",
        provider=MagicMock(model_name="extract"),
        answer_provider=answer,
        config=config or MemoryConfig(),
        seed_traces_path=str(seeds),
        questions_path=str(questions),
        aging=aging,
    )


_SEED: Log = [("connect", 0), ("experience", 0), ("experience", 0), ("close", 0)]
_EVALUATE: Log = [("connect", 2), ("health", 2), ("think", 2), ("close", 2)]


@pytest.mark.parametrize(
    ("aging", "stepping"),
    [
        (Aging(age_days=0), []),
        (
            Aging(age_days=1, passes_per_day=2),
            [("advance", _CLOCK), ("pass", 1), ("advance", _CLOCK), ("pass", 1)],
        ),
    ],
)
async def test_evaluation_memory_is_constructed_after_aging(
    tmp_path: Path, log: Log, aging: Aging, stepping: Log
) -> None:
    await _runner(tmp_path, aging).run_all()

    # The seeding instance closes before stored time moves; the evaluation
    # instance opens only after the store has settled.
    aged: Log = [("connect", 1), *stepping, ("settle", _CLOCK), ("close", 1)]
    assert log == [*_SEED, *aged, *_EVALUATE]


async def test_aged_run_reports_age_passes_seeding_and_health(
    tmp_path: Path, log: Log
) -> None:
    runner = _runner(tmp_path, Aging(age_days=1, passes_per_day=2))

    await runner.run_all()

    report = runner.aging_report
    assert report is not None
    assert (report.age_days, report.passes_per_day, report.passes_run) == (1, 2, 2)
    assert (report.seeding_user_id, report.seeded_traces) == ("probe-test", 2)
    assert report.health.tokens_dormant == 4
    # Final shift, settle, and reconnection run past the target age.
    assert report.evaluation_skew_seconds == 9.0


async def test_unaged_run_keeps_one_instance(tmp_path: Path, log: Log) -> None:
    runner = _runner(tmp_path, None)

    await runner.run_all()

    assert runner.aging_report is None
    assert log == [*_SEED[:-1], ("think", 0), ("close", 0)]


async def test_stepping_reads_every_seeded_trace_per_pass(
    tmp_path: Path, log: Log
) -> None:
    # Seeding compresses arrival into minutes; a batch window smaller than
    # the store would leave late traces unread, a state prod never holds.
    config = MemoryConfig()
    config._set("consolidation.batch_size", 1)
    runner = _runner(tmp_path, Aging(age_days=1, passes_per_day=2), config)

    await runner.run_all()

    assert config.get("consolidation.batch_size") == 2
    assert runner.aging_report is not None
    assert runner.aging_report.consolidation_batch_size == 2
