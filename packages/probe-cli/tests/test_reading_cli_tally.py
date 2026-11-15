"""Reading CLI: frozen-reading refusals and the tally command."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

import pytest
from probe_cli import reading_cli
from probe_cli.reading_packet import ADDED_CLAIM_CHECK, parse_packet_items
from probe_cli.reading_reader import ReaderAnswer
from probe_cli.representations import (
    RelationRow,
    RepresentationsReport,
    StatementRepresentation,
)
from recollect.llm.types import ExtractionResult

if TYPE_CHECKING:
    from pathlib import Path


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="probe")
    reading_cli.add_parsers(parser.add_subparsers(dest="cmd", required=True))
    return parser.parse_args(argv)


def _setup(tmp_path: Path) -> list[str]:
    key = tmp_path / "key.jsonl"
    checks = ["Read alone, is it hedged?", ADDED_CLAIM_CHECK]
    key.write_text(json.dumps({"id": "st-a", "checks": checks}) + "\n")
    (tmp_path / "brief.md").write_text("Brief.\n", encoding="utf-8")
    config = tmp_path / "reading.toml"
    seats = "".join(
        f'[readers.{n}]\nmodel = "fake:{n}"\nmax_tokens = 64\n' for n in ("a", "b")
    )
    config.write_text(
        f'[packet]\nseed = 5\nreader_key = "{key}"\nbrief = "{tmp_path / "brief.md"}"\n'
        + seats,
        encoding="utf-8",
    )
    rel = RelationRow(triple="user plans trip", context="May travel.", weight=0.9)
    statement = StatementRepresentation(
        id="st-a",
        trace_content="The user might travel in May.",
        trace_status="active",
        extraction=ExtractionResult(),
        degraded=False,
        entity_relations=[rel],
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
    source = tmp_path / "ctl" / "representations.json"
    source.parent.mkdir()
    source.write_text(report.model_dump_json(), encoding="utf-8")
    return ["--config", str(config), "--source", f"ctl={source}"]


class _Seat:
    calls = 0

    def __init__(self, model: str, **_: Any) -> None:
        self.model_name = model

    async def complete_structured(self, *_: Any, **__: Any) -> ReaderAnswer:
        _Seat.calls += 1
        return ReaderAnswer(answers=["yes", "no"], added_claim_quotes=[])


async def _packet(tmp_path: Path) -> Path:
    out = tmp_path / "packet"
    argv = ["reading-packet", *_setup(tmp_path), "--out-dir", str(out)]
    assert await reading_cli.dispatch(_args(argv)) == 0
    return out


def _read(tmp_path: Path, out: Path, *flags: str) -> argparse.Namespace:
    argv = ["read-packet", "--config", str(tmp_path / "reading.toml")]
    return _args([*argv, "--reader", "a", "--packet-dir", str(out), *flags])


async def test_reading_packet_refuses_a_directory_that_holds_a_packet(
    tmp_path: Path,
) -> None:
    out = await _packet(tmp_path)
    before = (out / "packet.jsonl").read_bytes()
    argv = ["reading-packet", "--config", str(tmp_path / "reading.toml")]
    argv += ["--source", f"ctl={tmp_path / 'ctl' / 'representations.json'}"]

    assert await reading_cli.dispatch(_args([*argv, "--out-dir", str(out)])) == 2
    assert (out / "packet.jsonl").read_bytes() == before


@pytest.mark.parametrize("changed", ["packet/packet.jsonl", "brief.md"])
async def test_resume_refuses_a_changed_packet_or_brief_naming_both_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    changed: str,
) -> None:
    monkeypatch.setattr(reading_cli, "PydanticAIProvider", _Seat)
    out = await _packet(tmp_path)
    assert await reading_cli.dispatch(_read(tmp_path, out)) == 0
    recorded = json.loads((out / "reader_a_meta.json").read_text(encoding="utf-8"))
    with (tmp_path / changed).open("a", encoding="utf-8") as f:
        f.write("\n")
    _Seat.calls = 0

    assert await reading_cli.dispatch(_read(tmp_path, out, "--resume")) == 2
    assert _Seat.calls == 0
    assert recorded["packet_sha256"] in caplog.text
    assert recorded["brief_sha256"] in caplog.text


async def test_resume_continues_an_unchanged_reading_and_needs_its_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reading_cli, "PydanticAIProvider", _Seat)
    out = await _packet(tmp_path)
    assert await reading_cli.dispatch(_read(tmp_path, out)) == 0
    _Seat.calls = 0

    assert await reading_cli.dispatch(_read(tmp_path, out, "--resume")) == 0
    assert _Seat.calls == 0
    (out / "reader_a_meta.json").unlink()
    assert await reading_cli.dispatch(_read(tmp_path, out, "--resume")) == 2


async def test_reading_tally_writes_tally_json_and_refuses_one_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = await _packet(tmp_path)
    items = parse_packet_items((out / "packet.jsonl").read_text(encoding="utf-8"))
    for name in ("a", "b"):
        rows = [
            {"item_id": i.item_id, "answers": ["yes", "no"], "success": True}
            for i in items
        ]
        text = "".join(json.dumps(r) + "\n" for r in rows)
        (out / f"reader_{name}.jsonl").write_text(text, encoding="utf-8")
    argv = ["reading-tally", "--config", str(tmp_path / "reading.toml")]
    argv += ["--packet-dir", str(out), "--control", "ctl"]

    assert await reading_cli.dispatch(_args(argv)) == 0
    report = json.loads((out / "tally.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "not_applicable"
    claims = report["readers"][0]["labels"][0]["context"]["added_claims"]
    assert [c["quote_missing"] for c in claims] == [True]
    assert "verdict=not_applicable" in capsys.readouterr().out

    one = [*argv, "--candidate", "ctl"]
    assert await reading_cli.dispatch(_args(one)) == 2
