"""Reading tally: one state per qualification check, added claims, line extras."""

from __future__ import annotations

from typing import Any

from probe_cli.reading_packet import (
    ADDED_CLAIM_CHECK,
    KeyEntry,
    LineListing,
    PacketItem,
    ReaderKeyRow,
    RelationListing,
    SourceListings,
)
from probe_cli.reading_reader import ReaderRecord
from probe_cli.reading_tally import TallyInputs, TallyReport, TallyRequest, tally

_ROWS = {
    "st-3": ["Q0?", "Q1?", ADDED_CLAIM_CHECK],
    "st-4": ["Q0?", "Q1?", "Q2?", ADDED_CLAIM_CHECK],
    "st-n": ["Q0?", "Q1?", ADDED_CLAIM_CHECK],
    "st-e": ["Q0?", "Q1?", "Q2?", ADDED_CLAIM_CHECK],
}
_LISTINGS = SourceListings(
    label="ctl",
    no_relation=["st-n"],
    empty_context=[RelationListing(statement_id="st-e", index=0, triple="t")],
    fallback=[LineListing(pinned_under="st-n", source="st-n", line="l")],
)


class _Build:
    def __init__(self) -> None:
        self.key: list[KeyEntry] = []
        self.items: list[PacketItem] = []
        self.answers: dict[str, list[ReaderRecord]] = {"a": [], "b": []}

    def add(self, sid: str, rep: str, a: list[str] | None, **kw: Any) -> str:
        item_id = f"item-{len(self.key) + 1:04d}"
        kind = kw.get("kind", "context")
        self.key.append(
            KeyEntry(
                item_id=item_id,
                label="ctl",
                statement_id=sid,
                kind=kind,
                index=len(self.key),
                weight=kw.get("weight", 0.7),
            )
        )
        self.items.append(
            PacketItem(
                item_id=item_id,
                statement=sid,
                checks=_ROWS[sid],
                kind=kind,
                representation=rep,
            )
        )
        for name in ("a", "b"):
            given = kw.get(name, a)
            record = {"item_id": item_id, "success": given is not None}
            record |= {"answers": given or [], "added_claim_quotes": kw.get("q", [])}
            self.answers[name].append(ReaderRecord.model_validate(record))
        return item_id

    def tally(self, listings: SourceListings = _LISTINGS) -> TallyReport:
        inputs = TallyInputs(
            key=self.key,
            items=self.items,
            listings=[listings],
            rows=[ReaderKeyRow(id=sid, checks=c) for sid, c in _ROWS.items()],
            answers=self.answers,
        )
        return tally(inputs, TallyRequest(control="ctl", confidence_threshold=0.6))


def _states(report: TallyReport, kind: str) -> dict[tuple[str, int], tuple[Any, ...]]:
    label = report.readers[0].labels[0]
    tallied = label.context if kind == "context" else label.line
    return {(s.statement_id, s.check): (s.state, s.cause) for s in tallied.states}


def test_each_state_and_cause_per_check_on_three_and_four_check_rows() -> None:
    b = _Build()
    b.add("st-4", "Ctx 4a.", ["yes", "no", "not_applicable", "yes"])
    b.add("st-4", "Ctx 4b.", ["yes", "yes", "not_applicable", "yes"])
    b.add("st-3", "Ctx 3.", None)
    b.add("st-3", "[pinned] t -- Ctx 3.", ["not_applicable", "yes", "yes"], kind="line")

    report = b.tally()
    context = _states(report, "context")
    line = _states(report, "line")

    assert context == {
        ("st-3", 0): ("unrepresented", "unanswered"),
        ("st-3", 1): ("unrepresented", "unanswered"),
        ("st-4", 0): ("pass", None),
        ("st-4", 1): ("fail", None),
        ("st-4", 2): ("unrepresented", "all_not_applicable"),
        ("st-n", 0): ("unrepresented", "no_relation"),
        ("st-n", 1): ("unrepresented", "no_relation"),
        ("st-e", 0): ("unrepresented", "empty_context"),
        ("st-e", 1): ("unrepresented", "empty_context"),
        ("st-e", 2): ("unrepresented", "empty_context"),
    }
    assert line[("st-3", 0)] == ("unrepresented", "all_not_applicable")
    assert line[("st-3", 1)] == ("pass", None)
    assert line[("st-4", 2)] == ("unrepresented", "no_line")
    assert line[("st-n", 0)] == ("unrepresented", "fallback_only")
    counts = report.readers[0].labels[0].context.counts
    assert counts == {"pass": 1, "fail": 1, "unrepresented": 8}


def test_added_claims_carry_quotes_or_quote_missing() -> None:
    b = _Build()
    quoted = b.add("st-3", "Ctx A.", ["yes", "yes", "no"], q=["so plan early"])
    bare = b.add("st-4", "Ctx B.", ["yes", "yes", "yes", "no"])
    b.add("st-3", "[pinned] t -- Ctx A.", ["yes", "yes", "yes"], kind="line")

    claims = b.tally().readers[0].labels[0].context.added_claims

    assert [(c.item_id, c.quotes, c.quote_missing) for c in claims] == [
        (quoted, ["so plan early"], False),
        (bare, [], True),
    ]


def test_triple_failures_and_the_threshold_split_on_lines() -> None:
    b = _Build()
    ctx = b.add("st-4", "Ctx A.", ["yes", "no", "not_applicable", "yes"])
    triple = b.add(
        "st-4", "[pinned] t -- Ctx A.", ["no", "no", "no", "yes"], kind="line"
    )
    low = b.add(
        "st-4", "[pinned] u -- Ctx A.", ["yes", "yes", "yes", "no"], kind="line"
    )
    b.add("st-4", "[pinned] v -- Ctx A.", ["yes"] * 4, kind="line", weight=None)
    b.key[2] = b.key[2].model_copy(update={"weight": 0.3})

    lines = b.tally().readers[0].labels[0].lines

    assert [
        (f.item_id, f.check, f.context_item_id) for f in lines.fails_only_through_triple
    ] == [
        (triple, 0, ctx),
        (triple, 2, ctx),
    ]
    assert lines.all_lines.model_dump() == {
        "passing": 1,
        "failing": 2,
        "failing_only_through_triple": 0,
    }
    assert lines.at_or_above_threshold.failing == 1
    assert lines.at_or_above_threshold.passing == 0
    assert lines.weight_none == ["item-0004"]
    assert low not in [f.item_id for f in lines.fails_only_through_triple]


def test_a_line_failing_only_where_its_context_passes_counts_as_triple_only() -> None:
    b = _Build()
    b.add("st-3", "Ctx A.", ["yes", "not_applicable", "yes"])
    b.add("st-3", "[pinned] t -- Ctx A.", ["no", "no", "yes"], kind="line")

    counts = b.tally().readers[0].labels[0].lines.all_lines

    assert (counts.failing, counts.failing_only_through_triple) == (1, 1)


def test_disagreements_and_incomplete_items_are_listed() -> None:
    b = _Build()
    differ = b.add("st-3", "Ctx A.", ["yes", "yes", "yes"], b=["yes", "no", "yes"])
    missing = b.add("st-3", "Ctx B.", ["yes", "yes", "yes"], b=None)

    report = b.tally()

    assert [(d.item_id, d.answers) for d in report.disagreements] == [
        (differ, {"a": ["yes", "yes", "yes"], "b": ["yes", "no", "yes"]})
    ]
    assert [r.incomplete for r in report.readers] == [[], [missing]]
