"""Aging CLI: a list of ages is one sweep over one seeding."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from probe_cli import __main__ as cli
from probe_cli.aging import AgingReport, SweepPoint
from probe_cli.task_runner import (
    SweepPointResult,
    TaskRunReport,
    VerdictOracleReport,
)
from recollect.models import HealthReport

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from probe_cli.arm import Arm


def _arm(tmp_path: Path) -> Path:
    path = tmp_path / "t.toml"
    path.write_text(
        '[arm]\nname = "t"\nruns = 1\n'
        '[extraction]\npydantic_ai_model = "test"\n'
        '[corpus]\npath = "unused.jsonl"\n'
        f'[output]\ndir = "{tmp_path / "out"}"\n'
        '[task]\nenabled = true\nseed_traces_path = "seeds.jsonl"\n'
        'questions_path = "questions.jsonl"\nanswer_model = "test"\n',
        encoding="utf-8",
    )
    return path


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *flags: str) -> int:
    argv = ["probe", "run", "--arm", str(_arm(tmp_path))]
    argv += ["--env-file", str(tmp_path / "absent.env"), *flags]
    monkeypatch.setattr(sys, "argv", argv)
    return cli.main()


def _result(arm: Arm, point: SweepPoint) -> SweepPointResult:
    name = point.arm_name(arm.name)
    run = TaskRunReport(
        arm_name=name,
        run_index=0,
        model="test",
        answer_model="test",
        density_tier=0,
        seeded_traces=2,
    )
    return SweepPointResult(
        arm_name=name,
        reports=[run],
        oracle=VerdictOracleReport(arm_name=name, runs=1),
        aging=AgingReport(
            age_days=point.aging.age_days,
            passes_per_day=point.aging.passes_per_day,
            passes_run=point.aging.age_days * point.aging.passes_per_day,
            seeding_user_id=f"probe-{arm.name}-sweep",
            seeding_session_id=f"probe-{arm.name}-seed-1",
            seeded_traces=2,
            consolidation_batch_size=50,
            evaluation_skew_seconds=6.5,
            recall_tokens_at_evaluation=not point.tokens_off,
            health=HealthReport(status="ok"),
        ),
    )


@pytest.fixture()
def swept(monkeypatch: pytest.MonkeyPatch) -> list[SweepPoint]:
    """Points the CLI handed to one runner's sweep."""
    points: list[SweepPoint] = []

    def from_arm(arm: Arm, _provider: object, *, aging: object) -> MagicMock:
        assert aging is None, "a sweep seeds once, under the unsuffixed arm"

        async def run_sweep(
            requested: list[SweepPoint],
        ) -> AsyncGenerator[SweepPointResult, None]:
            for point in requested:
                points.append(point)
                yield _result(arm, point)

        return MagicMock(answer_model="test", run_sweep=run_sweep)

    monkeypatch.setattr(cli.TaskArmRunner, "from_arm", from_arm)
    return points


@pytest.mark.parametrize(
    ("flags", "dirs"),
    [
        (["--age-days", "0", "7"], ["t-age0", "t-age7"]),
        (
            ["--age-days", "0", "7", "--tokens-off-control"],
            ["t-age0", "t-age0-tokens-off", "t-age7"],
        ),
        (["--age-days", "7", "--tokens-off-control"], ["t-age0-tokens-off", "t-age7"]),
        (
            ["--age-days", "0", "1", "--passes-per-day", "336"],
            ["t-age0-p336", "t-age1-p336"],
        ),
    ],
)
def test_each_sweep_point_writes_its_own_evidence_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    swept: list[SweepPoint],
    flags: list[str],
    dirs: list[str],
) -> None:
    assert _run(monkeypatch, tmp_path, *flags) == 0

    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == dirs
    assert sorted(point.arm_name("t") for point in swept) == dirs
    for point in swept:
        point_dir = out / point.arm_name("t")
        assert (point_dir / "task_summary.json").exists()
        recorded = json.loads((point_dir / "aging.json").read_text("utf-8"))
        assert recorded["age_days"] == point.aging.age_days
        assert recorded["seeding_user_id"] == "probe-t-sweep"
        assert recorded["recall_tokens_at_evaluation"] is not point.tokens_off


def test_the_control_is_age_zero_and_runs_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, swept: list[SweepPoint]
) -> None:
    assert _run(monkeypatch, tmp_path, "--age-days", "7", "--tokens-off-control") == 0

    assert [(p.aging.age_days, p.tokens_off) for p in swept] == [(7, False), (0, True)]


@pytest.mark.parametrize(
    "flags",
    [
        ["--age-days", "1", "7", "1"],
        ["--tokens-off-control"],
        ["--age-days", "0", "7", "--verify-reachability"],
    ],
)
def test_sweep_usage_errors_exit_before_connecting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[str]
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("built a provider before rejecting the flags")

    monkeypatch.setattr(cli, "PydanticAIProvider", refuse)

    assert _run(monkeypatch, tmp_path, *flags) == 2


@pytest.mark.parametrize(
    ("flags", "existing"),
    [
        (["--age-days", "7"], "t-age7"),
        (["--age-days", "0", "7"], "t-age7"),
        (["--age-days", "7", "--tokens-off-control"], "t-age0-tokens-off"),
    ],
)
def test_an_aged_run_refuses_existing_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    swept: list[SweepPoint],
    flags: list[str],
    existing: str,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("built a provider before refusing the run")

    monkeypatch.setattr(cli, "PydanticAIProvider", refuse)
    earlier = tmp_path / "out" / existing / "aging.json"
    earlier.parent.mkdir(parents=True)
    earlier.write_text("earlier seeding", encoding="utf-8")

    assert _run(monkeypatch, tmp_path, *flags) == 2

    assert swept == []
    assert earlier.read_text("utf-8") == "earlier seeding"
    assert [p.name for p in (tmp_path / "out").iterdir()] == [existing]
