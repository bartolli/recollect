"""Aging CLI: --age-days is an answering-mode option of the task arm."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli import __main__ as cli
from probe_cli.aging import Aging, AgingReport
from probe_cli.task_runner import TaskRunReport, VerdictOracleReport
from recollect.models import HealthReport

if TYPE_CHECKING:
    from pathlib import Path

    from probe_cli.arm import Arm

_TASK = """
[task]
enabled = true
seed_traces_path = "seeds.jsonl"
questions_path = "questions.jsonl"
answer_model = "test"
"""


def _arm(tmp_path: Path, *, task: bool) -> Path:
    path = tmp_path / "t.toml"
    path.write_text(
        '[arm]\nname = "t"\nruns = 1\n'
        '[extraction]\npydantic_ai_model = "test"\n'
        '[corpus]\npath = "unused.jsonl"\n'
        f'[output]\ndir = "{tmp_path / "out"}"\n' + (_TASK if task else ""),
        encoding="utf-8",
    )
    return path


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm_path: Path, *flags: str
) -> int:
    argv = ["probe", "run", "--arm", str(arm_path)]
    argv += ["--env-file", str(tmp_path / "absent.env"), *flags]
    monkeypatch.setattr(sys, "argv", argv)
    return cli.main()


@pytest.mark.parametrize(
    ("task", "mode"),
    [(True, "--verify-reachability"), (True, "--audit-fact-channel"), (False, None)],
)
def test_age_days_rejects_verify_audit_and_non_task_arms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: bool, mode: str | None
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("built a provider before rejecting the flags")

    monkeypatch.setattr(cli, "PydanticAIProvider", refuse)
    flags = ["--age-days", "7", *([mode] if mode else [])]

    assert _run(monkeypatch, tmp_path, _arm(tmp_path, task=task), *flags) == 2


@pytest.mark.parametrize(
    ("flags", "aged_name", "passes_per_day"),
    [
        (["--age-days", "7"], "t-age7", 48),
        (["--age-days", "7", "--passes-per-day", "336"], "t-age7-p336", 336),
    ],
)
def test_aged_run_writes_its_own_evidence_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flags: list[str],
    aged_name: str,
    passes_per_day: int,
) -> None:
    def from_arm(arm: Arm, _provider: object, *, aging: Aging) -> MagicMock:
        runner = MagicMock(answer_model="test")
        run = TaskRunReport(
            arm_name=arm.name,
            run_index=0,
            model="test",
            answer_model="test",
            density_tier=0,
            seeded_traces=2,
        )
        oracle = VerdictOracleReport(arm_name=arm.name, runs=1)
        runner.run_all = AsyncMock(return_value=([run], oracle))
        runner.aging_report = AgingReport(
            age_days=aging.age_days,
            passes_per_day=aging.passes_per_day,
            passes_run=aging.age_days * aging.passes_per_day,
            seeding_user_id=f"probe-{arm.name}",
            seeded_traces=2,
            consolidation_batch_size=50,
            evaluation_skew_seconds=6.5,
            health=HealthReport(status="ok"),
        )
        return runner

    monkeypatch.setattr(cli.TaskArmRunner, "from_arm", from_arm)

    assert _run(monkeypatch, tmp_path, _arm(tmp_path, task=True), *flags) == 0

    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [aged_name]
    assert (out / aged_name / "task_summary.json").exists()
    recorded = json.loads((out / aged_name / "aging.json").read_text("utf-8"))
    assert recorded["age_days"] == 7
    assert recorded["passes_per_day"] == passes_per_day
    assert recorded["passes_run"] == 7 * passes_per_day
    assert recorded["seeding_user_id"] == f"probe-{aged_name}"
    assert recorded["evaluation_skew_seconds"] == 6.5
    assert recorded["health"]["status"] == "ok"


def test_passes_per_day_requires_age_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arm = _arm(tmp_path, task=True)

    assert _run(monkeypatch, tmp_path, arm, "--passes-per-day", "48") == 2


def test_negative_age_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _run(monkeypatch, tmp_path, _arm(tmp_path, task=True), "--age-days", "-1")

    assert exit_info.value.code == 2
