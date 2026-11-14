"""Authoring fixtures: conversation validation, guidance variants, proposed wording."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from probe_cli.authoring import load_conversations, load_variants
from pydantic import ValidationError
from recollect_mcp.server import _INSTRUCTIONS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "authoring"
GUIDANCE = FIXTURES / "guidance"


def _conversation(cid: str = "c1", **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": cid,
        "case": "elliptical-referent",
        "today": "2026-06-02",
        "turns": ["one", "two", "three"],
        "targets": [{"turn": 2, "must_mention": ["x"], "reader_checks": ["y?"]}],
    }
    return base | overrides


def _write(tmp_path: Path, *rows: dict[str, object]) -> Path:
    path = tmp_path / "conversations.jsonl"
    lines = ["# header", *(json.dumps(r) for r in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_header_lines_skipped_and_rows_load(tmp_path: Path) -> None:
    loaded = load_conversations(_write(tmp_path, _conversation()))
    assert [c.id for c in loaded] == ["c1"]


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, _conversation(), _conversation())
    with pytest.raises(ValueError, match="duplicate conversation id: c1"):
        load_conversations(path)


@pytest.mark.parametrize("turns", [[], ["one", ""], ["one", "   "]])
def test_empty_turns_rejected(tmp_path: Path, turns: list[str]) -> None:
    row = _conversation(turns=turns, targets=[])
    with pytest.raises(ValidationError):
        load_conversations(_write(tmp_path, row))


def test_target_turn_out_of_range_rejected(tmp_path: Path) -> None:
    target = {"turn": 3, "must_mention": ["x"], "reader_checks": ["y?"]}
    row = _conversation(targets=[target])
    with pytest.raises(ValidationError, match="past the last turn"):
        load_conversations(_write(tmp_path, row))


def test_shipped_conversations_cover_five_cases() -> None:
    loaded = load_conversations(FIXTURES / "conversations.jsonl")
    assert len(loaded) == 5
    assert {c.case for c in loaded} == {
        "elliptical-referent",
        "conversation-dependency",
        "reported-uncertain",
        "implied-consequence",
        "correction",
    }
    assert all(3 <= len(c.turns) <= 4 for c in loaded)


def test_shipped_target_turns_end_with_a_save_request() -> None:
    for conversation in load_conversations(FIXTURES / "conversations.jsonl"):
        for target in conversation.targets:
            last = conversation.turns[target.turn].lower()
            assert "save" in last or "remember" in last, conversation.id


def test_shipped_variants_load() -> None:
    paths = [GUIDANCE / f"{n}.toml" for n in ("served", "client-v1", "proposed")]
    served, client, proposed = load_variants(paths)
    assert served.model_dump(exclude={"name"}) == {
        "remember_description": None,
        "instructions": None,
        "system_prompt": None,
    }
    assert client.system_prompt is not None
    assert "mcp__recollect__" not in client.system_prompt
    assert client.remember_description is None
    assert proposed.remember_description is not None


def test_duplicate_variant_names_rejected() -> None:
    served = GUIDANCE / "served.toml"
    with pytest.raises(ValueError, match="duplicate guidance variant name: served"):
        load_variants([served, served])


def _blocks(text: str) -> list[str]:
    return text.split("\n\n")


def test_proposed_instructions_differ_only_in_remember_block() -> None:
    (proposed,) = load_variants([GUIDANCE / "proposed.toml"])
    assert proposed.instructions is not None
    served, ours = _blocks(_INSTRUCTIONS), _blocks(proposed.instructions)
    assert len(served) == len(ours)
    changed = [i for i, (a, b) in enumerate(zip(served, ours, strict=True)) if a != b]
    assert len(changed) == 1
    assert served[changed[0]].startswith("REMEMBER -- ")
    assert ours[changed[0]].startswith("REMEMBER -- ")


def test_proposed_description_mirrors_served_docstring_shape() -> None:
    (proposed,) = load_variants([GUIDANCE / "proposed.toml"])
    description = proposed.remember_description
    assert description is not None
    assert description.startswith("Store an experience in memory.\n\n    ")
    assert "\n    Args:\n        content: " in description
    assert description.endswith("\n    ")
