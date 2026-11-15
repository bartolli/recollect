"""probe validity: one line, validity.json in the evidence dir, exit by state."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from probe_cli import __main__ as cli
from probe_cli.arm import load_arm
from probe_cli.provenance import stamp_prompt
from probe_cli.report import write_surfacing_run_report
from probe_cli.surfacing_runner import SurfacingRunReport
from probe_cli.validity import SeedingValidity
from recollect.extraction import PatternExtractor

if TYPE_CHECKING:
    from pathlib import Path


def _arm(tmp_path: Path, *, mode: str = "surfacing") -> Path:
    traces, queries = tmp_path / "t.jsonl", tmp_path / "q.jsonl"
    traces.write_text('{"id": "a", "text": "a"}\n', encoding="utf-8")
    queries.write_text("", encoding="utf-8")
    path = tmp_path / "arm.toml"
    path.write_text(
        '[arm]\nname = "surf"\n[extraction]\npydantic_ai_model = "test"\n'
        f'instructions = "Suffix."\n[corpus]\npath = "{traces}"\n'
        f'[output]\ndir = "{tmp_path / "out"}"\n'
        f'[{mode}]\nenabled = true\ntraces_corpus_path = "{traces}"\n'
        f'query_corpus_path = "{queries}"\n',
        encoding="utf-8",
    )
    return path


def _validity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm_path: Path, *flags: str
) -> int:
    argv = ["probe", "validity", "--arm", str(arm_path)]
    argv += ["--env-file", str(tmp_path / "absent.env"), *flags]
    monkeypatch.setattr(sys, "argv", argv)
    return cli.main()


def test_a_valid_seeding_prints_valid_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arm_path = _arm(tmp_path)
    config = load_arm(arm_path).to_memory_config()
    stamp = stamp_prompt(PatternExtractor(MagicMock(), config=config))
    report = SurfacingRunReport(
        arm_name="surf-s2", model="m", seeded_traces=1, promoted_facts=0, prompt=stamp
    )
    out = tmp_path / "out" / "surf-s2"
    write_surfacing_run_report(
        report.model_copy(update={"max_facts_per_query": 5}), out
    )

    assert _validity(monkeypatch, tmp_path, arm_path, "--seeding", "s2") == 0

    line = capsys.readouterr().out
    assert line.startswith("VALID surf-s2 missing_artifact=0 prompt_mismatch=0 ")
    assert line.count("\n") == 1
    verdict = SeedingValidity.model_validate_json(
        (out / "validity.json").read_text("utf-8")
    )
    assert (verdict.state, verdict.expected_prompt) == ("VALID", stamp)


def test_a_seeding_that_produced_nothing_is_invalid_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _validity(monkeypatch, tmp_path, _arm(tmp_path), "--seeding", "s3") == 1

    assert capsys.readouterr().out.startswith("INVALID surf-s3 missing_artifact=1 ")
    assert (tmp_path / "out" / "surf-s3" / "validity.json").exists()


def test_an_arm_without_a_seeding_mode_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _validity(monkeypatch, tmp_path, _arm(tmp_path, mode="other")) == 2

    assert not (tmp_path / "out").exists()


def test_a_bad_seeding_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _validity(monkeypatch, tmp_path, _arm(tmp_path), "--seeding", "s0")

    assert exit_info.value.code == 2
