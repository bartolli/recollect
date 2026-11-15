"""Representations CLI: an arm without a task is refused before any store call."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from probe_cli import __main__ as cli
from probe_cli import representations

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _refuse(*_: object, **__: object) -> None:
    raise AssertionError("reached the store")


def test_cli_refuses_an_arm_without_a_task_before_any_db_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "read_representations", _refuse)
    monkeypatch.setattr(representations, "_database_names", _refuse)
    monkeypatch.setattr(representations, "_copy_store", _refuse)
    arm = tmp_path / "a.toml"
    arm.write_text('[arm]\nname = "a"\n[corpus]\npath = ""\n', encoding="utf-8")
    out = tmp_path / "out"
    argv = ["probe", "representations", "--arm", str(arm), "--template-db", "keep"]
    argv += ["--traces", str(tmp_path / "t.jsonl"), "--out-dir", str(out)]
    argv += ["--env-file", str(tmp_path / "absent.env")]
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 2
    assert not out.exists()
