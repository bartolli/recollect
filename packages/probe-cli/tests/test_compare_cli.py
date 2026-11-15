"""Compare CLI: only VALID seedings are read; the flag admits an absent validity."""

from __future__ import annotations

import argparse
import json
import logging
from typing import TYPE_CHECKING

import pytest
from probe_cli import compare_cli
from probe_cli.guard_compare import GuardComparison
from probe_cli.situational_metrics import SituationalAggregate
from probe_cli.surfacing_compare import SurfacingComparison
from probe_cli.surfacing_runner import QuerySurfacing, SurfacedFact, SurfacingRunReport
from recollect.config import config as sdk_config

if TYPE_CHECKING:
    from pathlib import Path


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    compare_cli.add_parsers(sub)
    return parser.parse_args(argv)


_VALID = json.dumps({"state": "VALID", "checks": []})
_INVALID = json.dumps({"state": "INVALID", "checks": []})


def _validity(directory: Path, text: str | None) -> None:
    if text is not None:
        (directory / "validity.json").write_text(text)


def _guard_dirs(tmp_path: Path, validity: list[str | None]) -> list[Path]:
    dirs = []
    for i, text in enumerate(validity):
        d = tmp_path / f"seed{i}"
        d.mkdir()
        _validity(d, text)
        agg = SituationalAggregate(
            arm_name=f"arm{i}",
            model="m",
            runs=3,
            eval_count=15,
            action_accuracy_mean=0.9,
            action_accuracy_std=0.0,
            extension_target_accuracy_mean=1.0,
            extension_target_accuracy_std=0.0,
            none_rate_actual_mean=0.4,
            none_rate_expected=0.4,
            expected_distribution={},
        )
        (d / "situational_summary.json").write_text(agg.model_dump_json())
        dirs.append(d)
    return dirs


def _guard_argv(dirs: list[Path], out: Path) -> list[str]:
    c1, c2, k1, k2 = (str(d) for d in dirs)
    argv = ["guard-compare", "--control", c1, "--control", c2]
    return [*argv, "--candidate", k1, "--candidate", k2, "--out", str(out)]


@pytest.mark.parametrize(
    ("validity", "allow", "state"),
    [
        (None, False, "absent"),
        (_INVALID, False, "INVALID"),
        (_INVALID, True, "INVALID"),
        ("{not json", True, "unreadable"),
        ('{"checks": []}', True, "unreadable"),
        ('{"state": "maybe"}', True, "maybe"),
    ],
)
async def test_a_seeding_not_known_valid_is_refused_naming_the_state_found(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    validity: str | None,
    allow: bool,
    state: str,
) -> None:
    dirs = _guard_dirs(tmp_path, [_VALID, _VALID, _VALID, validity])
    out = tmp_path / "out.json"
    flag = ["--allow-unvalidated"] if allow else []

    with caplog.at_level(logging.ERROR):
        code = await compare_cli.dispatch(_parse([*_guard_argv(dirs, out), *flag]))

    assert code == 2
    assert not out.exists()
    assert f"{dirs[3]} (validity: {state})" in caplog.text
    assert str(dirs[0]) not in caplog.text


async def test_allow_unvalidated_admits_and_records_only_absent_validity(
    tmp_path: Path,
) -> None:
    dirs = _guard_dirs(tmp_path, [_VALID, None, _VALID, None])
    out = tmp_path / "out.json"

    code = await compare_cli.dispatch(
        _parse([*_guard_argv(dirs, out), "--allow-unvalidated"])
    )

    assert code == 0
    result = GuardComparison.model_validate_json(out.read_text())
    assert [(u.path, u.state) for u in result.unvalidated_inputs] == [
        (str(dirs[1]), "absent"),
        (str(dirs[3]), "absent"),
    ]
    assert result.verdict == "no_regression_called"


async def test_a_wrong_seeding_count_or_half_a_fresh_pair_exits_2(
    tmp_path: Path,
) -> None:
    dirs = _guard_dirs(tmp_path, [_VALID] * 5)
    out = tmp_path / "out.json"
    three = [*_guard_argv(dirs[:4], out), "--control", str(dirs[4])]
    half = [*_guard_argv(dirs[:4], out), "--fresh-control", str(dirs[4])]

    assert await compare_cli.dispatch(_parse(three)) == 2
    assert await compare_cli.dispatch(_parse(half)) == 2
    assert not out.exists()


async def test_another_command_is_not_dispatched_here() -> None:
    assert await compare_cli.dispatch(argparse.Namespace(cmd="run")) is None


def test_main_reads_verbose_and_env_file_on_every_command() -> None:
    args = _parse(["guard-compare", "--control", "a", "--candidate", "b", "--out", "o"])

    assert (args.verbose, args.env_file) == (False, None)


async def test_surfacing_compare_reads_the_floor_from_the_sdk_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queries = tmp_path / "q.jsonl"
    queries.write_text(
        json.dumps({"id": "q1", "text": "t", "relevant_trace_ids": ["a"]})
    )
    reading = tmp_path / "reading.toml"
    reading.write_text(f'queries = "{queries}"\noverlap_pairs = []\n')
    fact = SurfacedFact(
        fact_id="f", source_eval_id="a", score=0.9, relevant=True, passes_gate=True
    )
    report = SurfacingRunReport(
        arm_name="arm",
        model="m",
        seeded_traces=1,
        promoted_facts=0,
        queries=[QuerySurfacing(query_id="q1", surfaced=[fact])],
    )
    dirs = []
    for i in range(4):
        d = tmp_path / f"s{i}"
        d.mkdir()
        _validity(d, _VALID)
        (d / "surfacing_run.json").write_text(report.model_dump_json())
        dirs.append(d)
    out = tmp_path / "out.json"
    argv = ["surfacing-compare", "--config", str(reading)]
    argv += [x for d in dirs[:2] for x in ("--control", str(d))]
    argv += [x for d in dirs[2:] for x in ("--candidate", str(d))]

    assert await compare_cli.dispatch(_parse([*argv, "--out", str(out)])) == 0

    result = SurfacingComparison.model_validate_json(out.read_text())
    assert result.floor == float(sdk_config.get("persona.recall_relevance_floor"))
    assert result.verdict == "no_regression_called"
    assert "verdict=no_regression_called" in capsys.readouterr().out
