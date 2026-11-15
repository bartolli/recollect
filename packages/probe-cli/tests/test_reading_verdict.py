"""Context verdict: per reader against the control, then the joint rule."""

from __future__ import annotations

import pytest
from probe_cli.reading_packet import (
    ADDED_CLAIM_CHECK,
    KeyEntry,
    PacketItem,
    ReaderKeyRow,
    SourceListings,
)
from probe_cli.reading_reader import ReaderRecord
from probe_cli.reading_tally import (
    TallyError,
    TallyInputs,
    TallyReport,
    TallyRequest,
    tally,
)

_CHECKS = ["Q0?", "Q1?", "Q2?", ADDED_CLAIM_CHECK]
_YES = ["yes", "yes", "yes", "yes"]
_Pair = tuple[list[str] | None, list[str] | None]


def _inputs(per_label: dict[str, _Pair]) -> TallyInputs:
    """One st-4 context item per label; a pair is (reader a, reader b) answers."""
    key: list[KeyEntry] = []
    items: list[PacketItem] = []
    answers: dict[str, list[ReaderRecord]] = {"a": [], "b": []}
    for n, (label, pair) in enumerate(per_label.items(), start=1):
        item_id = f"item-{n:04d}"
        key.append(
            KeyEntry(
                item_id=item_id,
                label=label,
                statement_id="st-4",
                kind="context",
                index=0,
                weight=0.7,
            )
        )
        items.append(
            PacketItem(
                item_id=item_id,
                statement="S.",
                checks=_CHECKS,
                kind="context",
                representation=f"Ctx {label}.",
            )
        )
        for name, given in zip(("a", "b"), pair, strict=True):
            record = {"item_id": item_id, "success": given is not None}
            record |= {"answers": given or [], "added_claim_quotes": ["q"]}
            answers[name].append(ReaderRecord.model_validate(record))
    return TallyInputs(
        key=key,
        items=items,
        listings=[SourceListings(label=label) for label in per_label],
        rows=[ReaderKeyRow(id="st-4", checks=_CHECKS)],
        answers=answers,
    )


def _tally(
    per_label: dict[str, _Pair], candidates: tuple[str, ...] = ("c1", "c2")
) -> TallyReport:
    request = TallyRequest(
        control="ctl", candidates=list(candidates), confidence_threshold=0.6
    )
    return tally(_inputs(per_label), request)


def _verdicts(report: TallyReport) -> tuple[list[str], str]:
    return [r.verdict for r in report.readers], report.verdict


def test_all_checks_passing_everywhere_holds() -> None:
    report = _tally({"ctl": (_YES, _YES), "c1": (_YES, _YES), "c2": (_YES, _YES)})

    assert _verdicts(report) == (["holds", "holds"], "holds")


def test_a_regression_on_a_non_first_check_does_not_hold() -> None:
    fails_q2 = ["yes", "yes", "no", "yes"]
    report = _tally(
        {"ctl": (_YES, _YES), "c1": (fails_q2, fails_q2), "c2": (_YES, _YES)}
    )

    assert _verdicts(report) == (["does_not_hold", "does_not_hold"], "does_not_hold")
    regression = report.readers[0].regressions[0].model_dump()
    assert regression == {
        "statement_id": "st-4",
        "check": 2,
        "check_text": "Q2?",
        "control_state": "pass",
        "candidate_label": "c1",
        "candidate_state": "fail",
        "cause": None,
    }


def test_a_candidate_context_claim_does_not_hold_and_a_control_claim_alone_holds() -> (
    None
):
    claim = ["yes", "yes", "yes", "no"]

    at_candidate = _tally(
        {"ctl": (_YES, _YES), "c1": (_YES, _YES), "c2": (claim, _YES)}
    )
    at_control = _tally({"ctl": (claim, claim), "c1": (_YES, _YES), "c2": (_YES, _YES)})

    assert _verdicts(at_candidate) == (["does_not_hold", "holds"], "undetermined")
    assert _verdicts(at_control) == (["holds", "holds"], "holds")


def test_unrepresented_where_control_passed_is_undetermined_and_listed() -> None:
    na_q1 = ["yes", "not_applicable", "yes", "yes"]
    report = _tally({"ctl": (_YES, _YES), "c1": (_YES, _YES), "c2": (na_q1, na_q1)})

    assert _verdicts(report) == (["undetermined", "undetermined"], "undetermined")
    entry = report.readers[1].unrepresented_where_control_passed[0]
    assert (entry.check, entry.candidate_label, entry.candidate_state) == (
        1,
        "c2",
        "unrepresented",
    )
    assert entry.cause == "all_not_applicable"
    assert report.readers[1].regressions == []


def test_an_incomplete_reading_is_undetermined_unless_it_does_not_hold() -> None:
    fails_q0 = ["no", "yes", "yes", "yes"]

    incomplete = _tally({"ctl": (_YES, _YES), "c1": (_YES, None), "c2": (_YES, _YES)})
    failing = _tally({"ctl": (_YES, _YES), "c1": (fails_q0, _YES), "c2": (None, _YES)})

    assert _verdicts(incomplete) == (["holds", "undetermined"], "undetermined")
    assert _verdicts(failing) == (["does_not_hold", "holds"], "undetermined")


def test_zero_candidates_is_not_applicable() -> None:
    report = _tally({"ctl": (_YES, _YES)}, candidates=())

    assert _verdicts(report) == (["not_applicable", "not_applicable"], "not_applicable")
    assert report.readers[0].regressions == []


@pytest.mark.parametrize("candidates", [("c1",), ("c1", "c9")])
def test_one_candidate_or_an_unknown_label_is_refused(
    candidates: tuple[str, ...],
) -> None:
    with pytest.raises(TallyError):
        _tally({"ctl": (_YES, _YES), "c1": (_YES, _YES)}, candidates=candidates)
