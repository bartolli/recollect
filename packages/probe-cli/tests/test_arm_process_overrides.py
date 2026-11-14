"""Arm overrides reach the process config the stores read."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from probe_cli.arm import load_arm
from recollect.config import config as sdk_config

_KEY = "retrieval.selection_threshold"
_OVERRIDE = 0.25
_ARM = (
    '[arm]\nname = "x"\n[corpus]\npath = "x.jsonl"\n'
    "[recollect_overrides]\n"
    f'"{_KEY}" = {_OVERRIDE}\n'
)


@pytest.fixture(autouse=True)
def _restore_process_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sdk_config, "_config", copy.deepcopy(sdk_config._config))


def _arm_path(tmp_path: Path) -> Path:
    p = tmp_path / "arm.toml"
    p.write_text(_ARM, encoding="utf-8")
    return p


def test_loading_an_arm_leaves_the_process_config(tmp_path: Path) -> None:
    before = sdk_config.get(_KEY)
    assert before != _OVERRIDE
    load_arm(_arm_path(tmp_path)).to_memory_config()
    assert sdk_config.get(_KEY) == before


def test_overrides_reach_the_process_config(tmp_path: Path) -> None:
    assert sdk_config.get(_KEY) != _OVERRIDE
    load_arm(_arm_path(tmp_path)).apply_process_overrides()
    assert sdk_config.get(_KEY) == _OVERRIDE
