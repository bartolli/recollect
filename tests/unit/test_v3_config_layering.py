"""Config layering: packaged defaults survive user TOML.

Resolution: bootstrap floor -> packaged config.toml -> first user TOML
(custom > MEMORY_CONFIG > cwd) -> env. Drift tests pin the bootstrap
floor and every literal call-site fallback to the packaged values.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import Any

import pytest
import recollect
from recollect.config import MemoryConfig

_PKG = Path(recollect.__file__).parent
_PACKAGED = tomllib.loads((_PKG / "config.toml").read_text())


def _toml_lookup(dotted: str) -> tuple[Any, bool]:
    node: Any = _PACKAGED
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None, False
        node = node[part]
    return node, True


class TestLayering:
    def test_partial_override_keeps_packaged_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One-key user TOML must not drop the packaged cognitive params."""
        monkeypatch.delenv("MEMORY_CONFIG", raising=False)
        user = tmp_path / "memory.toml"
        user.write_text('[database]\nurl = "postgresql://user/db"\n')
        cfg = MemoryConfig(config_path=user)
        assert cfg.get("retrieval.max_retrievals") == 7
        assert cfg.get("retrieval.selection_threshold") == 0.1

    def test_resolution_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """defaults -> packaged TOML -> user TOML -> env."""
        monkeypatch.delenv("MEMORY_CONFIG", raising=False)
        user = tmp_path / "memory.toml"
        user.write_text(
            '[database]\nurl = "postgresql://user/db"\n'
            "[retrieval]\nmax_retrievals = 4\n"
        )
        monkeypatch.setenv("DATABASE_URL", "postgresql://env/db")
        cfg = MemoryConfig(config_path=user)
        assert cfg.database_url == "postgresql://env/db"  # env > user
        assert cfg.get("retrieval.max_retrievals") == 4  # user > packaged
        assert cfg.get("retrieval.selection_threshold") == 0.1  # packaged

    def test_user_toml_beats_packaged_without_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MEMORY_CONFIG", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        user = tmp_path / "memory.toml"
        user.write_text('[database]\nurl = "postgresql://user/db"\n')
        cfg = MemoryConfig(config_path=user)
        assert cfg.database_url == "postgresql://user/db"


class TestDrift:
    def test_bootstrap_floor_matches_packaged(self) -> None:
        """Every _load_defaults leaf equals its packaged config.toml twin."""
        mismatches: list[str] = []

        def walk(node: dict[str, Any], prefix: str) -> None:
            for key, value in node.items():
                dotted = f"{prefix}{key}"
                if isinstance(value, dict):
                    walk(value, f"{dotted}.")
                    continue
                packaged, found = _toml_lookup(dotted)
                if not found:
                    mismatches.append(f"{dotted} missing from packaged config.toml")
                elif packaged != value:
                    mismatches.append(
                        f"{dotted}: floor {value!r} != packaged {packaged!r}"
                    )

        walk(MemoryConfig()._load_defaults(), "")
        assert not mismatches, "\n".join(mismatches)

    def test_call_site_fallbacks_match_packaged(self) -> None:
        """Literal config.get fallbacks agree with packaged config.toml.

        Unreachable in practice -- the packaged layer always loads -- but
        pinned so a stale fallback never misleads a reader. Empty-string
        fallbacks are unset sentinels, exempt.
        """
        mismatches: list[str] = []
        for py in sorted(_PKG.rglob("*.py")):
            tree = ast.parse(py.read_text())
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) == 2
                ):
                    continue
                key_node, fb_node = node.args
                if not (
                    isinstance(key_node, ast.Constant)
                    and isinstance(key_node.value, str)
                    and "." in key_node.value
                    and isinstance(fb_node, ast.Constant)
                ):
                    continue
                key, fallback = key_node.value, fb_node.value
                packaged, found = _toml_lookup(key)
                if not found or fallback == "":
                    continue
                if isinstance(packaged, int | float) and isinstance(
                    fallback, int | float
                ):
                    if float(packaged) == float(fallback):
                        continue
                elif packaged == fallback:
                    continue
                mismatches.append(
                    f"{py.name}:{node.lineno} {key} "
                    f"fallback {fallback!r} != packaged {packaged!r}"
                )
        assert not mismatches, "\n".join(mismatches)
