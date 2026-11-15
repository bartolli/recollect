"""Base arms: production's prompt composition, one p121/p130 pair per mode."""

from __future__ import annotations

import tomllib
from importlib.resources import files
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from probe_cli.arm import Arm, load_arm
from probe_cli.provenance import stamp_prompt
from recollect.config import MemoryConfig
from recollect.extraction import PatternExtractor

_FIX = Path(__file__).parent.parent / "fixtures"
_CANDIDATE = "packages/probe-cli/fixtures/qualification/extraction.1.3.0.md"
_BASE_ARMS = {
    "qualification": "qualification/arm-{}.toml",
    "surfacing": "surfacing-h55/{}.toml",
    "guard": "situational/guard-h55-{}.toml",
}
# The assembled system message per condition, one value across the three modes.
_P121_SHA = "8af2ffecbcbc681a7861beec3f18467edebd75f54c85af38c91967bee207330e"
_P130_SHA = "9c261ea4a058e0dfc7e0943f825bacd1ed747e932528acc714844c52b8f3505d"
_COMPOSITION = {
    "p121": ("1.2.1", _P121_SHA, 11060),
    "p130": ("1.3.0", _P130_SHA, 13462),
}
_PROMPT_ENV = ("MEMORY_EXTRACTION_INSTRUCTIONS", "MEMORY_EXTRACTION_TEMPLATE_PATH")


def _base_arm(monkeypatch: pytest.MonkeyPatch, mode: str, cond: str) -> Arm:
    monkeypatch.setenv("RECOLLECT_PROBE_DB_URL", "postgresql://probe/probe_eval")
    monkeypatch.setenv("RECOLLECT_PROBE_SITUATIONAL_DB_URL", "postgresql://probe/s")
    # Hermetic default config: these would override the packaged values.
    for name in ("MEMORY_CONFIG", *_PROMPT_ENV):
        monkeypatch.delenv(name, raising=False)
    return load_arm(_FIX / _BASE_ARMS[mode].format(cond))


def _prompt(config: MemoryConfig) -> str:
    return PatternExtractor(MagicMock(), config=config)._build_prompt()


@pytest.mark.parametrize("mode", _BASE_ARMS)
def test_each_p130_arm_differs_from_its_p121_twin_by_name_and_template_alone(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    p121, p130 = (_base_arm(monkeypatch, mode, c) for c in ("p121", "p130"))

    assert p130.name == p121.name.replace("p121", "p130") != p121.name
    assert p130.extraction.template_path == _CANDIDATE
    twin = p130.extraction.model_copy(update={"template_path": ""})
    assert p130.model_copy(update={"name": p121.name, "extraction": twin}) == p121


@pytest.mark.parametrize("cond", _COMPOSITION)
@pytest.mark.parametrize("mode", _BASE_ARMS)
def test_every_base_arm_sends_production_composition(
    monkeypatch: pytest.MonkeyPatch, mode: str, cond: str
) -> None:
    arm = _base_arm(monkeypatch, mode, cond)
    packaged = tomllib.loads((files("recollect") / "config.toml").read_text("utf-8"))

    assert arm.extraction.instructions == packaged["extraction"]["instructions"]
    assert arm.extraction.pydantic_ai_model == "anthropic:claude-haiku-5-5"
    stamp = stamp_prompt(PatternExtractor(MagicMock(), config=arm.to_memory_config()))
    assert (stamp.version, stamp.sha256, stamp.chars) == _COMPOSITION[cond]


@pytest.mark.parametrize("mode", _BASE_ARMS)
def test_every_p121_arm_assembles_the_default_config_prompt(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    arm = _base_arm(monkeypatch, mode, "p121")

    assert _prompt(arm.to_memory_config()) == _prompt(MemoryConfig())
