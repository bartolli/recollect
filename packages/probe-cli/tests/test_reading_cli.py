"""Reading CLI: packet and reader commands, wired through add_parsers and dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from probe_cli import reading_cli
from probe_cli.reading_packet import ADDED_CLAIM_CHECK
from probe_cli.reading_reader import ReaderAnswer
from probe_cli.representations import (
    RelationRow,
    RepresentationsReport,
    StatementRepresentation,
)
from recollect.llm.types import CompletionParams, ExtractionResult

_ROOT = Path(__file__).resolve().parents[3]


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="probe")
    reading_cli.add_parsers(parser.add_subparsers(dest="cmd", required=True))
    return parser.parse_args(argv)


def _config(tmp_path: Path) -> Path:
    key = tmp_path / "key.jsonl"
    checks = ["Read alone, is it hedged?", ADDED_CLAIM_CHECK]
    key.write_text("# h\n" + json.dumps({"id": "st-a", "checks": checks}) + "\n")
    (tmp_path / "brief.md").write_text("Brief.\n", encoding="utf-8")
    config = tmp_path / "reading.toml"
    config.write_text(
        f'[packet]\nseed = 5\nreader_key = "{key}"\nbrief = "{tmp_path / "brief.md"}"\n'
        '[readers.a]\nmodel = "fake:model"\nmax_tokens = 64\n'
        "[readers.a.model_settings]\neffort = 1\n",
        encoding="utf-8",
    )
    return config


def _source(tmp_path: Path, validity: str | None = None) -> Path:
    rel = RelationRow(triple="user plans trip", context="May travel.", weight=0.9)
    statement = StatementRepresentation(
        id="st-a",
        trace_content="The user might travel in May.",
        trace_status="active",
        extraction=ExtractionResult(),
        degraded=False,
        entity_relations=[rel, rel.model_copy(update={"context": "Trip later."})],
        persona_facts=[],
        pinned_fact_lines=[],
    )
    report = RepresentationsReport(
        arm_name="arm",
        template_db="keep",
        user_id="u",
        traces_path="t.jsonl",
        traces_sha256="0" * 64,
        degraded_statements=[],
        statements=[statement],
    )
    path = tmp_path / "ctl" / "representations.json"
    path.parent.mkdir()
    path.write_text(report.model_dump_json(), encoding="utf-8")
    if validity is not None:
        (path.parent / "validity.json").write_text(json.dumps({"state": validity}))
    return path


async def _packet(
    tmp_path: Path, validity: str | None = None
) -> tuple[int | None, Path]:
    out = tmp_path / "packet"
    argv = ["reading-packet", "--config", str(_config(tmp_path))]
    argv += ["--source", f"ctl={_source(tmp_path, validity)}", "--out-dir", str(out)]
    return await reading_cli.dispatch(_args(argv)), out


class _Seat:
    def __init__(self, model: str, *, defaults: CompletionParams) -> None:
        self.model_name = model
        self.defaults = defaults
        self.calls = 0
        seats.append(self)

    async def complete_structured(self, *_: Any, **__: Any) -> ReaderAnswer:
        self.calls += 1
        return ReaderAnswer(answers=["yes", "yes"], added_claim_quotes=[])


seats: list[_Seat] = []


def test_tracked_config_names_both_seats_and_existing_files() -> None:
    config = reading_cli.load_reading_config(
        _ROOT / "packages" / "probe-cli" / "fixtures" / "qualification" / "reading.toml"
    )

    assert config.packet.seed == 20261114
    assert (_ROOT / config.packet.reader_key).is_file()
    assert (_ROOT / config.packet.brief).is_file()
    assert sorted(config.readers) == ["a", "b"]


async def test_reading_packet_writes_the_packet_with_its_meta(tmp_path: Path) -> None:
    code, out = await _packet(tmp_path)

    assert code == 0
    meta = json.loads((out / "packet_meta.json").read_text(encoding="utf-8"))
    key_digest = hashlib.sha256((tmp_path / "key.jsonl").read_bytes()).hexdigest()
    assert (meta["seed"], meta["reader_key_sha256"]) == (5, key_digest)
    assert meta["items_by_kind"] == {"context": 2, "line": 0}
    assert [s["label"] for s in meta["sources"]] == ["ctl"]
    assert len((out / "packet.jsonl").read_text().splitlines()) == 2


async def test_reading_packet_refuses_an_invalid_source(tmp_path: Path) -> None:
    code, out = await _packet(tmp_path, "INVALID")

    assert code == 2
    assert not out.exists()


async def test_read_packet_answers_every_item_and_records_its_seat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seats.clear()
    monkeypatch.setattr(reading_cli, "PydanticAIProvider", _Seat)
    _, out = await _packet(tmp_path)
    argv = ["read-packet", "--config", str(tmp_path / "reading.toml")]
    argv += ["--reader", "a", "--packet-dir", str(out)]

    assert await reading_cli.dispatch(_args(argv)) == 0
    assert [(s.model_name, s.calls) for s in seats] == [("fake:model", 2)]
    assert seats[0].defaults == CompletionParams(effort=1)
    meta = json.loads((out / "reader_a_meta.json").read_text(encoding="utf-8"))
    assert (meta["model"], meta["max_tokens"], meta["answered"]) == (
        "fake:model",
        64,
        2,
    )
    packet_digest = hashlib.sha256((out / "packet.jsonl").read_bytes()).hexdigest()
    assert meta["packet_sha256"] == packet_digest
    assert len((out / "reader_a.jsonl").read_text().splitlines()) == 2

    assert await reading_cli.dispatch(_args(argv)) == 2
    assert [s.calls for s in seats] == [2]


async def test_read_packet_refuses_a_seat_absent_from_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reading_cli, "PydanticAIProvider", _Seat)
    _, out = await _packet(tmp_path)
    argv = ["read-packet", "--config", str(tmp_path / "reading.toml")]

    code = await reading_cli.dispatch(
        _args([*argv, "--reader", "c", "--packet-dir", str(out)])
    )

    assert code == 2
    assert not (out / "reader_c.jsonl").exists()


async def test_dispatch_leaves_other_commands_alone() -> None:
    assert await reading_cli.dispatch(argparse.Namespace(cmd="run")) is None
