"""Reading packet: one shuffled item per context and per distinct pinned line."""

from __future__ import annotations

import json

import pytest
from probe_cli.reading_packet import (
    ADDED_CLAIM_CHECK,
    PacketError,
    PacketItem,
    build_packet,
    parse_reader_key,
)
from probe_cli.representations import (
    PinnedLine,
    RelationRow,
    RepresentationsReport,
    StatementRepresentation,
)
from recollect.llm.types import ExtractionResult

_KIND = "Read alone, does this keep the view hedged?"


def _rel(triple: str, context: str, weight: float = 0.7) -> RelationRow:
    return RelationRow(triple=triple, context=context, weight=weight)


def _pin(line: str, source: str, *, fallback: bool = False) -> PinnedLine:
    return PinnedLine(line=line, source=source, fallback=fallback)


def _stmt(
    sid: str,
    relations: list[RelationRow],
    pinned: list[PinnedLine] | None = None,
) -> StatementRepresentation:
    return StatementRepresentation(
        id=sid,
        trace_content=f"Text of statement {sid.upper()}.",
        trace_status="active",
        extraction=ExtractionResult(),
        degraded=False,
        entity_relations=relations,
        persona_facts=[],
        pinned_fact_lines=pinned or [],
    )


def _report(statements: list[StatementRepresentation]) -> RepresentationsReport:
    return RepresentationsReport(
        arm_name="arm-secret",
        template_db="keep",
        user_id="probe-arm-secret",
        traces_path="traces.jsonl",
        traces_sha256="0" * 64,
        degraded_statements=[],
        statements=statements,
    )


def _key(ids: list[str]) -> str:
    rows = [{"id": i, "checks": [_KIND, ADDED_CLAIM_CHECK]} for i in ids]
    return "# header\n" + "\n".join(json.dumps(r) for r in rows) + "\n"


def _line(rel: RelationRow) -> str:
    return f"[pinned] {rel.triple} -- {rel.context}"


_A1 = _rel("user plans trip", "The user may travel in May.", 0.9)
_A2 = _rel("user knows Ana", "Ana is a friend.", 0.4)
_B1 = _rel("Ana holds_role guide", "Ana guides tours.", 0.8)


def _three() -> RepresentationsReport:
    return _report(
        [
            _stmt("st-a", [_A1, _A2], [_pin(_line(_A1), "st-a")]),
            _stmt("st-b", [_B1], [_pin(_line(_B1), "st-b"), _pin(_line(_A1), "st-a")]),
            _stmt("st-c", [_rel("x y z", "Context C.")]),
        ]
    )


def _bytes(seed: int) -> str:
    rows = parse_reader_key(_key(["st-a", "st-b", "st-c"]))
    packet = build_packet([("ctl", _three()), ("cand", _three())], rows, seed=seed)
    return "".join(i.model_dump_json() for i in packet.items) + "".join(
        k.model_dump_json() for k in packet.key
    )


def test_same_seed_gives_same_bytes_and_another_seed_another_order() -> None:
    assert _bytes(7) == _bytes(7)
    assert _bytes(7) != _bytes(8)


def test_items_carry_no_label_statement_id_arm_or_weight() -> None:
    rows = parse_reader_key(_key(["st-a", "st-b", "st-c"]))
    packet = build_packet([("ctl", _three())], rows, seed=1)

    assert set(PacketItem.model_fields) == {
        "item_id",
        "statement",
        "checks",
        "kind",
        "representation",
    }
    for item in packet.items:
        text = item.model_dump_json()
        for secret in ("ctl", "st-a", "st-b", "st-c", "arm-secret", "0.9", "0.4"):
            assert secret not in text
        assert item.checks == [_KIND, ADDED_CLAIM_CHECK]
    assert [i.item_id for i in packet.items] == [
        f"item-{n:04d}" for n in range(1, len(packet.items) + 1)
    ]
    assert [k.item_id for k in packet.key] == [i.item_id for i in packet.items]


def test_twin_line_is_read_once_against_its_source_with_both_pinned_under() -> None:
    rows = parse_reader_key(_key(["st-a", "st-b", "st-c"]))
    packet = build_packet([("ctl", _three())], rows, seed=1)
    lines = [
        (i, k)
        for i, k in zip(packet.items, packet.key, strict=True)
        if k.kind == "line"
    ]

    assert len(lines) == 2
    twin = next((i, k) for i, k in lines if i.representation == _line(_A1))
    item, entry = twin
    assert entry.statement_id == "st-a"
    assert entry.pinned_under == ["st-a", "st-b"]
    assert entry.pin_finding
    assert entry.weight == 0.9
    assert item.statement == "Text of statement ST-A."
    listings = packet.listings[0]
    assert [(p.pinned_under, p.source) for p in listings.pin_findings] == [
        ("st-b", "st-a")
    ]


def test_contexts_are_items_with_the_relation_weight() -> None:
    rows = parse_reader_key(_key(["st-a", "st-b", "st-c"]))
    packet = build_packet([("ctl", _three())], rows, seed=1)
    contexts = {
        (k.statement_id, k.index): (i.representation, k.weight)
        for i, k in zip(packet.items, packet.key, strict=True)
        if k.kind == "context"
    }

    assert contexts == {
        ("st-a", 0): (_A1.context, 0.9),
        ("st-a", 1): (_A2.context, 0.4),
        ("st-b", 0): (_B1.context, 0.8),
        ("st-c", 0): ("Context C.", 0.7),
    }


def test_line_weight_is_none_when_no_source_relation_matches() -> None:
    report = _report([_stmt("st-a", [_A1], [_pin("[pinned] a b c", "st-a")])])
    packet = build_packet([("ctl", report)], parse_reader_key(_key(["st-a"])), seed=1)

    line = next(k for k in packet.key if k.kind == "line")
    assert line.weight is None
    assert not line.pin_finding


def test_unread_representations_are_listed_per_label() -> None:
    report = _report(
        [
            _stmt("st-a", [], [_pin("[pinned] user noted x", "st-a", fallback=True)]),
            _stmt("st-b", [_rel("p q r", ""), _B1], [_pin("[pinned] a b c", "")]),
        ]
    )
    rows = parse_reader_key(_key(["st-a", "st-b"]))
    packet = build_packet([("ctl", report)], rows, seed=1)
    listings = packet.listings[0]

    assert listings.label == "ctl"
    assert listings.no_relation == ["st-a"]
    assert [(r.statement_id, r.index) for r in listings.empty_context] == [("st-b", 0)]
    assert [f.line for f in listings.fallback] == ["[pinned] user noted x"]
    assert [f.pinned_under for f in listings.empty_source] == ["st-b"]
    assert listings.pin_findings == []
    assert [(k.kind, k.statement_id) for k in packet.key] == [("context", "st-b")]


def test_statement_without_a_key_row_is_refused() -> None:
    rows = parse_reader_key(_key(["st-a", "st-b"]))

    with pytest.raises(PacketError, match="st-c"):
        build_packet([("ctl", _three())], rows, seed=1)
