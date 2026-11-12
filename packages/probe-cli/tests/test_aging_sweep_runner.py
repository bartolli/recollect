"""Aged sweeps: one seeding, restored before every point."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli.aging import Aging, SweepPoint
from probe_cli.task_runner import TaskArmRunner
from recollect.config import MemoryConfig
from recollect.config import config as sdk_config
from recollect.models import HealthReport

if TYPE_CHECKING:
    from pathlib import Path

Log = list[str]


class _FakeMemory:
    """Stands in for the SDK; every call lands on one shared log."""

    def __init__(self, log: Log, serial: int, config: MemoryConfig) -> None:
        self._log = log
        self._serial = serial
        self.tokens_enabled = config.get("recall_tokens.enabled")
        self.storage = MagicMock()
        self.storage.pool.get_pool = AsyncMock(return_value=MagicMock())
        self.storage.facts.get_persona_facts = AsyncMock(return_value=[])

    def _note(self, event: str) -> None:
        self._log.append(f"{event}{self._serial}")

    async def connect(self, db_url: str | None = None) -> None:
        self._note("connect")

    async def close(self) -> None:
        self._note("close")

    async def experience(self, text: str, **_: object) -> MagicMock:
        return MagicMock(id=f"id-{text}")

    async def consolidate(self) -> None:
        self._note("pass")

    async def think_about(self, query: str, **_: object) -> list[Any]:
        self._note("think")
        return []

    async def health_report(self, *, user_id: str | None = None) -> HealthReport:
        return HealthReport(status="ok")


class _Harness:
    def __init__(self) -> None:
        self.log: Log = []
        self.memories: list[_FakeMemory] = []
        self.snapshot_urls: list[str] = []
        self.snapshot = MagicMock()
        for operation in ("capture", "restore", "discard"):
            recorder = AsyncMock(side_effect=lambda op=operation: self.log.append(op))
            setattr(self.snapshot, operation, recorder)

    def memory(self, *, config: MemoryConfig, **_: object) -> _FakeMemory:
        fake = _FakeMemory(self.log, len(self.memories), config)
        self.memories.append(fake)
        return fake


@pytest.fixture()
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    harness = _Harness()
    clock = MagicMock()
    clock.advance_to = AsyncMock()
    clock.settle = AsyncMock(side_effect=lambda: harness.log.append("settle"))
    clock.emulated_offset = MagicMock(return_value=timedelta(days=1, seconds=9))
    monkeypatch.setattr("probe_cli.task_runner.CognitiveMemory", harness.memory)
    monkeypatch.setattr("probe_cli.task_runner.StoreClock", lambda _pool: clock)

    def snapshot(db_url: str) -> MagicMock:
        harness.snapshot_urls.append(db_url)
        return harness.snapshot

    monkeypatch.setattr("probe_cli.task_runner.StoreSnapshot", snapshot)
    return harness


def _runner(
    tmp_path: Path, config: MemoryConfig | None = None, db_url: str = ""
) -> TaskArmRunner:
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(
        json.dumps({"id": "s0", "text": "trace 0", "density_tier": 0}),
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
        db_url=db_url,
    )


async def test_every_point_restores_before_its_own_instances(
    tmp_path: Path, harness: _Harness
) -> None:
    points = [
        SweepPoint(aging=Aging(age_days=1, passes_per_day=1)),
        SweepPoint(aging=Aging(age_days=0)),
    ]

    results = [result async for result in _runner(tmp_path).run_sweep(points)]

    # The seeding instance closes before the capture; each point restores,
    # steps on one instance, and evaluates on another opened after settle.
    assert harness.log == [
        *["connect0", "close0", "capture"],
        *["restore", "connect1", "pass1", "settle", "close1"],
        *["connect2", "think2", "close2"],
        *["restore", "connect3", "settle", "close3"],
        *["connect4", "think4", "close4", "discard"],
    ]
    assert [r.arm_name for r in results] == ["test-age1-p1", "test-age0"]
    assert {r.aging.seeding_user_id for r in results} == {"probe-test-sweep"}
    # user_id repeats across sweeps of one arm; the session id names this one.
    assert len({r.aging.seeding_session_id for r in results}) == 1
    assert results[0].aging.seeding_session_id.startswith("probe-test-seed-")
    assert [r.reports[0].arm_name for r in results] == ["test-age1-p1", "test-age0"]


async def test_the_control_steps_with_tokens_on_and_reads_with_them_off(
    tmp_path: Path, harness: _Harness
) -> None:
    config = MemoryConfig()
    config._set("recall_tokens.enabled", True)
    control = SweepPoint(aging=Aging(age_days=1, passes_per_day=1), tokens_off=True)

    results = [r async for r in _runner(tmp_path, config).run_sweep([control])]

    # Seeding and stepping see the arm's config; only the evaluation
    # instance reads with the hop disabled.
    assert [m.tokens_enabled for m in harness.memories] == [True, True, False]
    assert config.get("recall_tokens.enabled") is True
    assert results[0].arm_name == "test-age1-p1-tokens-off"
    assert results[0].aging.recall_tokens_at_evaluation is False


async def test_a_refused_point_still_discards_the_snapshot(
    tmp_path: Path, harness: _Harness
) -> None:
    harness.snapshot.restore.side_effect = ValueError("foreign session")
    points = [SweepPoint(aging=Aging(age_days=0))]

    with pytest.raises(ValueError, match="foreign session"):
        _ = [result async for result in _runner(tmp_path).run_sweep(points)]

    assert harness.log == ["connect0", "close0", "capture", "discard"]


@pytest.mark.parametrize("db_url", ["postgresql://host/probe_arm", ""])
async def test_the_snapshot_targets_the_store_the_sdk_connects_to(
    tmp_path: Path, harness: _Harness, db_url: str
) -> None:
    runner = _runner(tmp_path, db_url=db_url)

    _ = [r async for r in runner.run_sweep([SweepPoint(aging=Aging(age_days=0))])]

    # An arm without db_url falls back as the SDK pool does.
    assert harness.snapshot_urls == [db_url or sdk_config.database_url]
