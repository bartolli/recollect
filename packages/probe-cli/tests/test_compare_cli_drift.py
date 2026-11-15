"""Drift CLI: validity sits beside each representations.json; SDK threshold."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from probe_cli import compare_cli
from probe_cli.drift import DriftReport
from probe_cli.representations import (
    RelationRow,
    RepresentationsReport,
    StatementRepresentation,
)
from recollect.config import config as sdk_config
from recollect.llm.types import ExtractionResult

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    compare_cli.add_parsers(sub)
    return parser.parse_args(argv)


def _representations(directory: Path, weight: float | None, *, valid: bool) -> Path:
    directory.mkdir()
    if valid:
        (directory / "validity.json").write_text(json.dumps({"state": "VALID"}))
    relations = (
        []
        if weight is None
        else [RelationRow(triple="a b c", context="", weight=weight)]
    )
    statement = StatementRepresentation(
        id="will-1",
        trace_content="text",
        trace_status="active",
        extraction=ExtractionResult(),
        degraded=False,
        entity_relations=relations,
        persona_facts=[],
        pinned_fact_lines=[],
    )
    report = RepresentationsReport(
        arm_name=directory.name,
        template_db="db",
        user_id="u",
        traces_path="t.jsonl",
        traces_sha256="0",
        degraded_statements=[],
        statements=[statement],
    )
    path = directory / "representations.json"
    path.write_text(report.model_dump_json())
    return path


def _argv(tmp_path: Path, *, candidate_valid: bool) -> list[str]:
    key = tmp_path / "key.jsonl"
    key.write_text(json.dumps({"id": "will-1", "certainty": "committed", "checks": []}))
    control = _representations(tmp_path / "c", 0.9, valid=True)
    k1 = _representations(tmp_path / "k1", None, valid=candidate_valid)
    k2 = _representations(tmp_path / "k2", 0.3, valid=True)
    argv = ["drift", "--control", str(control), "--reader-key", str(key)]
    return [*argv, "--candidate", str(k1), "--candidate", str(k2)]


async def test_a_candidate_without_a_sibling_validity_is_refused(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out.json"

    code = await compare_cli.dispatch(
        _parse([*_argv(tmp_path, candidate_valid=False), "--out", str(out)])
    )

    assert code == 2
    assert not out.exists()


async def test_drift_reads_the_threshold_and_names_the_loss(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out.json"

    code = await compare_cli.dispatch(
        _parse([*_argv(tmp_path, candidate_valid=True), "--out", str(out)])
    )

    assert code == 0
    report = DriftReport.model_validate_json(out.read_text())
    assert report.threshold == float(sdk_config.get("persona.confidence_threshold"))
    assert [(loss.statement_id, loss.status) for loss in report.relation_losses] == [
        ("will-1", "named_loss")
    ]
    assert report.unvalidated_inputs == []
    assert "named_loss" in capsys.readouterr().out
