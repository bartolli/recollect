"""surfacing-compare refuses a run gated at a floor other than the config floor."""

from __future__ import annotations

import argparse
import json
import logging
from typing import TYPE_CHECKING

from probe_cli import compare_cli
from probe_cli.corpus import QueryEntry
from probe_cli.surfacing_metrics import (
    compute_required_coverage,
    compute_surfacing_metrics,
)
from probe_cli.surfacing_runner import QuerySurfacing, SurfacedFact, SurfacingRunReport
from recollect.config import config as sdk_config

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_FLOOR = float(sdk_config.get("persona.recall_relevance_floor"))
_QUERY = QueryEntry(id="q1", text="t", relevant_trace_ids=["a"])
_REPORT = SurfacingRunReport(
    arm_name="arm",
    model="m",
    seeded_traces=1,
    promoted_facts=0,
    queries=[
        QuerySurfacing(
            query_id="q1",
            surfaced=[
                SurfacedFact(
                    fact_id="f",
                    source_eval_id="a",
                    score=0.9,
                    relevant=True,
                    passes_gate=True,
                )
            ],
        )
    ],
)


def _seeding(directory: Path, summary_floor: float | None, *, coverage: bool) -> str:
    """summary_floor None: no summary; coverage False: a summary without the block."""
    directory.mkdir()
    (directory / "validity.json").write_text(json.dumps({"state": "VALID"}))
    (directory / "surfacing_run.json").write_text(_REPORT.model_dump_json())
    if summary_floor is not None:
        metrics = compute_surfacing_metrics(_REPORT)
        if coverage:
            block = compute_required_coverage(_REPORT, [_QUERY], floor=summary_floor)
            metrics = metrics.model_copy(update={"required_coverage": block})
        (directory / "surfacing_summary.json").write_text(metrics.model_dump_json())
    return str(directory)


def _argv(tmp_path: Path, dirs: list[str]) -> list[str]:
    queries = tmp_path / "q.jsonl"
    queries.write_text(_QUERY.model_dump_json())
    reading = tmp_path / "reading.toml"
    reading.write_text(f'queries = "{queries}"\noverlap_pairs = []\n')
    argv = ["surfacing-compare", "--config", str(reading)]
    argv += ["--control", dirs[0], "--control", dirs[1]]
    argv += ["--candidate", dirs[2], "--candidate", dirs[3]]
    return [*argv, "--out", str(tmp_path / "out.json")]


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    compare_cli.add_parsers(sub)
    return parser.parse_args(argv)


async def test_a_run_recorded_at_another_floor_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dirs = [_seeding(tmp_path / f"s{i}", _FLOOR, coverage=True) for i in range(3)] + [
        _seeding(tmp_path / "s3", _FLOOR - 0.1, coverage=True)
    ]

    with caplog.at_level(logging.ERROR):
        code = await compare_cli.dispatch(_parse(_argv(tmp_path, dirs)))

    assert code == 2
    assert not (tmp_path / "out.json").exists()
    assert "recorded floor differs" in caplog.text
    assert dirs[3] in caplog.text
    assert dirs[0] not in caplog.text


async def test_runs_at_the_config_floor_or_without_a_recorded_floor_are_read(
    tmp_path: Path,
) -> None:
    dirs = [
        _seeding(tmp_path / "s0", _FLOOR, coverage=True),
        _seeding(tmp_path / "s1", _FLOOR, coverage=False),
        _seeding(tmp_path / "s2", None, coverage=False),
        _seeding(tmp_path / "s3", _FLOOR, coverage=True),
    ]

    assert await compare_cli.dispatch(_parse(_argv(tmp_path, dirs))) == 0
