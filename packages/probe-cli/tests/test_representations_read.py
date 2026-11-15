"""Representations read: the store-copy lifecycle around the statements."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
from probe_cli import representations
from probe_cli.arm import Arm, CorpusConfig, TaskConfig
from probe_cli.fusion_screen import _DROP_SCREEN, ScreenPreconditionError
from probe_cli.representations import StatementRepresentation, read
from recollect.llm.types import ExtractionResult

if TYPE_CHECKING:
    from pathlib import Path


def _arm() -> Arm:
    task = TaskConfig(enabled=True, db_url="postgresql://probe@localhost/probe_task")
    return Arm(name="reps", corpus=CorpusConfig(path=""), task=task)


def _refuse(*_: object, **__: object) -> None:
    raise AssertionError("reached the store")


def _statement(cid: str, *, degraded: bool) -> StatementRepresentation:
    return StatementRepresentation(
        id=cid,
        trace_content=f"statement {cid}",
        trace_status="active",
        extraction=ExtractionResult(concepts=[] if degraded else [cid]),
        entity_relations=[],
        persona_facts=[],
        pinned_fact_lines=[],
        degraded=degraded,
    )


def _fake_copy(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    statements: list[StatementRepresentation],
) -> None:
    async def _names(_: str) -> set[str]:
        return {"keep"}

    async def _record(_: str, *sql: str) -> None:
        events.extend(sql)

    async def _on_copy(*_: object, **__: object) -> list[StatementRepresentation]:
        events.append("read")
        return statements

    monkeypatch.setattr(representations, "_database_names", _names)
    monkeypatch.setattr(representations, "_copy_store", _record)
    monkeypatch.setattr(representations, "_maintenance", _record)
    monkeypatch.setattr(representations, "_represented", _on_copy)


async def test_read_refuses_an_empty_traces_file_before_any_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(representations, "_database_names", _refuse)
    monkeypatch.setattr(representations, "_copy_store", _refuse)
    path = tmp_path / "empty.jsonl"
    path.write_text("# no statements\n", encoding="utf-8")
    with pytest.raises(ScreenPreconditionError, match="no statements"):
        await read(_arm(), template="keep", user_id="u", traces_path=path)


async def test_read_hashes_the_traces_file_and_drops_the_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text('{"id": "a", "text": "statement a"}\n', encoding="utf-8")
    events: list[str] = []
    _fake_copy(monkeypatch, events, [])
    report = await read(_arm(), template="keep", user_id="u", traces_path=path)
    assert report.traces_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    head = (report.arm_name, report.template_db, report.user_id, report.traces_path)
    assert head == ("reps", "keep", "u", str(path))
    assert events == ["keep", "read", _DROP_SCREEN]


async def test_read_lists_degraded_statements_in_corpus_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text('{"id": "a", "text": "statement a"}\n', encoding="utf-8")
    flags = {"c": True, "a": False, "b": True}
    statements = [_statement(cid, degraded=d) for cid, d in flags.items()]
    _fake_copy(monkeypatch, [], statements)
    report = await read(_arm(), template="keep", user_id="u", traces_path=path)
    assert report.degraded_statements == ["c", "b"]
