"""SituationalSurfacingArmRunner factory + config wiring (slice-1b)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from probe_cli.arm import load_arm
from probe_cli.surfacing_situational_runner import SituationalSurfacingArmRunner


def _arm_file(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "arm.toml"
    p.write_text(
        '[arm]\nname = "ss"\n[corpus]\npath = ""\n' + body, encoding="utf-8"
    )
    return p


def test_from_arm_requires_seed_groups_path(tmp_path: Path) -> None:
    body = (
        "[surfacing]\nenabled = true\n"
        'traces_corpus_path = "t.jsonl"\nquery_corpus_path = "q.jsonl"\n'
    )
    arm = load_arm(_arm_file(tmp_path, body))
    with pytest.raises(ValueError, match="seed_groups_path"):
        SituationalSurfacingArmRunner.from_arm(arm, MagicMock(model_name="m"))


def test_from_arm_accepts_full_config(tmp_path: Path) -> None:
    body = (
        "[surfacing]\nenabled = true\n"
        'traces_corpus_path = "t.jsonl"\nquery_corpus_path = "q.jsonl"\n'
        'seed_groups_path = "g.jsonl"\n'
    )
    arm = load_arm(_arm_file(tmp_path, body))
    runner = SituationalSurfacingArmRunner.from_arm(arm, MagicMock(model_name="m"))
    assert runner._groups_path == "g.jsonl"
    assert runner.model == "m"
