"""Reading packet on the 1.2.1 representations: every one is accounted for."""

from __future__ import annotations

from pathlib import Path

from probe_cli.reading_packet import build_packet, parse_reader_key, read_source

_ROOT = Path(__file__).resolve().parents[3]
_REPS = _ROOT / "out" / "qualification-haiku55" / "representations.json"
_KEY = (
    _ROOT / "packages" / "probe-cli" / "fixtures" / "qualification" / "reader_key.jsonl"
)


def test_every_context_and_pinned_line_is_an_item_a_twin_or_a_listing() -> None:
    _, report = read_source("p121", _REPS)
    rows = parse_reader_key(_KEY.read_text(encoding="utf-8"))
    packet = build_packet([("p121", report)], rows, seed=20261114)
    pairs = list(zip(packet.items, packet.key, strict=True))
    listings = packet.listings[0]

    contexts = {(k.statement_id, k.index): i for i, k in pairs if k.kind == "context"}
    empty = {(r.statement_id, r.index) for r in listings.empty_context}
    for s in report.statements:
        assert (s.id in listings.no_relation) == (not s.entity_relations)
        for index, rel in enumerate(s.entity_relations):
            if rel.context:
                assert contexts[(s.id, index)].representation == rel.context
                assert contexts[(s.id, index)].statement == s.trace_content
            else:
                assert (s.id, index) in empty

    read = {
        (under, i.representation): k.statement_id
        for i, k in pairs
        if k.kind == "line"
        for under in k.pinned_under
    }
    unread = {
        (f.pinned_under, f.line) for f in listings.fallback + listings.empty_source
    }
    occurrences = 0
    for s in report.statements:
        for p in s.pinned_fact_lines:
            occurrences += 1
            assert read.get((s.id, p.line)) == p.source or (s.id, p.line) in unread

    lines = [k for _, k in pairs if k.kind == "line"]
    twins = sum(len(k.pinned_under) - 1 for k in lines)
    assert occurrences == len(lines) + twins + len(unread)
    assert (len(contexts), len(lines), twins) == (33, 32, 1)
    assert [(f.pinned_under, f.source) for f in listings.pin_findings] == [
        ("screen-2", "screen-1")
    ]
    assert [k.pinned_under for k in lines if k.pin_finding] == [
        ["screen-1", "screen-2"]
    ]
    assert listings.counts() == {
        "fallback": 0,
        "empty_source": 0,
        "no_relation": 0,
        "empty_context": 0,
        "pin_findings": 1,
    }
    assert all(k.weight is not None for k in lines)
