"""Reading packet files: key rows, source validity, written artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
from probe_cli.reading_packet import (
    ADDED_CLAIM_CHECK,
    PacketError,
    build_packet,
    packet_meta,
    parse_packet_items,
    parse_reader_key,
    read_source,
    write_packet,
)
from probe_cli.representations import (
    PinnedLine,
    RelationRow,
    RepresentationsReport,
    StatementRepresentation,
)
from recollect.llm.types import ExtractionResult

if TYPE_CHECKING:
    from pathlib import Path

_KIND = "Read alone, does this keep the view hedged?"
_KEY = json.dumps({"id": "st-a", "checks": [_KIND, ADDED_CLAIM_CHECK]})


def _report() -> RepresentationsReport:
    rel = RelationRow(triple="user plans trip", context="May travel.", weight=0.9)
    line = PinnedLine(
        line="[pinned] user plans trip -- May travel.", source="st-a", fallback=False
    )
    statement = StatementRepresentation(
        id="st-a",
        trace_content="The user might travel in May.",
        trace_status="active",
        extraction=ExtractionResult(),
        degraded=False,
        entity_relations=[rel],
        persona_facts=[],
        pinned_fact_lines=[line],
    )
    return RepresentationsReport(
        arm_name="arm",
        template_db="keep",
        user_id="u",
        traces_path="t.jsonl",
        traces_sha256="0" * 64,
        degraded_statements=[],
        statements=[statement],
    )


def _written(tmp_path: Path, name: str, validity: str | None = None) -> Path:
    path = tmp_path / name / "representations.json"
    path.parent.mkdir()
    path.write_text(_report().model_dump_json(indent=2), encoding="utf-8")
    if validity is not None:
        state = json.dumps({"state": validity, "reasons": []})
        (path.parent / "validity.json").write_text(state, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "checks",
    [
        [ADDED_CLAIM_CHECK, _KIND],
        [_KIND, "Does it add no claim?"],
        [_KIND, ADDED_CLAIM_CHECK, ADDED_CLAIM_CHECK],
    ],
)
def test_reader_key_requires_the_added_claim_check_once_and_last(
    checks: list[str],
) -> None:
    text = json.dumps({"id": "st-a", "checks": checks})

    with pytest.raises(PacketError, match="st-a"):
        parse_reader_key(text)


def test_reader_key_keeps_every_kind_check_in_key_order() -> None:
    checks = [_KIND, "Does it keep the condition?", ADDED_CLAIM_CHECK]
    rows = parse_reader_key("# c\n\n" + json.dumps({"id": "st-a", "checks": checks}))

    assert [(r.id, r.checks) for r in rows] == [("st-a", checks)]


def test_source_marked_invalid_is_refused(tmp_path: Path) -> None:
    path = _written(tmp_path, "bad", "INVALID")

    with pytest.raises(PacketError, match="INVALID"):
        read_source("bad", path)


@pytest.mark.parametrize(
    ("validity", "recorded"), [("VALID", "VALID"), (None, "absent")]
)
def test_source_records_path_digest_and_validity(
    tmp_path: Path, validity: str | None, recorded: str
) -> None:
    path = _written(tmp_path, "ctl", validity)

    source, report = read_source("ctl", path)

    assert source.label == "ctl"
    assert source.path == str(path)
    assert source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source.validity == recorded
    assert report == _report()


def test_written_packet_round_trips_and_meta_counts_it(tmp_path: Path) -> None:
    source, report = read_source("ctl", _written(tmp_path, "ctl"))
    packet = build_packet([("ctl", report)], parse_reader_key(_KEY), seed=3)
    meta = packet_meta(packet, seed=3, sources=[source], reader_key_sha256="k" * 64)
    out = tmp_path / "packet"

    write_packet(packet, meta, out)

    written_items = (out / "packet.jsonl").read_text(encoding="utf-8")
    assert parse_packet_items(written_items) == packet.items
    key_rows = (out / "key.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(row)["item_id"] for row in key_rows] == [
        "item-0001",
        "item-0002",
    ]
    listings = json.loads((out / "listings.json").read_text(encoding="utf-8"))
    assert [entry["label"] for entry in listings] == ["ctl"]
    written = json.loads((out / "packet_meta.json").read_text(encoding="utf-8"))
    assert written["seed"] == 3
    assert written["reader_key_sha256"] == "k" * 64
    assert written["items_by_kind"] == {"context": 1, "line": 1}
    assert written["listing_counts"]["ctl"]["pin_findings"] == 0
    assert written["sources"][0]["validity"] == "absent"
