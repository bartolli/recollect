"""Authoring scoring: turn-scoped must_mention hits, table rows, session user ids."""

from __future__ import annotations

from datetime import date

from probe_cli.authoring import (
    AuthoringPlan,
    AuthoringSummary,
    Conversation,
    GuidanceVariant,
    SessionRecord,
    Target,
    TargetScore,
    ToolCallRecord,
    score_session,
    summary_table,
)

CONVERSATION = Conversation(
    id="c1",
    case="correction",
    today=date(2026, 10, 5),
    turns=["a, save it", "b", "c, save it"],
    targets=[
        Target(turn=0, must_mention=["Quillmere"], reader_checks=["?"]),
        Target(turn=2, must_mention=["Quillmere", "Book Club"], reader_checks=["?"]),
    ],
)


def _call(turn: int, content: str, tool: str = "remember") -> ToolCallRecord:
    return ToolCallRecord(turn=turn, tool=tool, args={"content": content}, returned="")


def _record(*calls: ToolCallRecord) -> SessionRecord:
    return SessionRecord(
        variant="served",
        conversation_id="c1",
        run=0,
        model="m",
        user_id="u",
        tool_calls=list(calls),
    )


def test_hits_are_case_insensitive() -> None:
    record = _record(_call(2, "The QUILLMERE book club moved to Oct 27."))
    _, late = score_session(record, CONVERSATION).targets
    assert late == TargetScore(
        turn=2, remember_calls=1, mentions={"Quillmere": True, "Book Club": True}
    )


def test_hits_scoped_to_the_target_turn() -> None:
    record = _record(_call(0, "Quillmere book club on Oct 24."), _call(2, "Moved."))
    early, late = score_session(record, CONVERSATION).targets
    assert early.mentions == {"Quillmere": True}
    assert late.mentions == {"Quillmere": False, "Book Club": False}


def test_only_remember_calls_count() -> None:
    record = _record(
        _call(2, "Quillmere book club", tool="recall"),
        _call(2, "Quillmere"),
        _call(2, "book club on the 27th"),
    )
    _, late = score_session(record, CONVERSATION).targets
    assert late.remember_calls == 2
    assert late.mentions == {"Quillmere": True, "Book Club": True}


def test_session_without_writes_scores_zero_calls() -> None:
    early, late = score_session(_record(), CONVERSATION).targets
    assert (early.remember_calls, late.remember_calls) == (0, 0)
    assert not any(late.mentions.values())


def test_table_lists_every_target_with_hits_and_errors() -> None:
    record = _record(_call(2, "Quillmere"))
    errored = record.model_copy(update={"run": 1, "error": "ModelHTTPError: 503"})
    summary = AuthoringSummary(
        name="t",
        execution_id="x1",
        model="m",
        sessions=[score_session(r, CONVERSATION) for r in (record, errored)],
    )
    lines = summary_table(summary).splitlines()
    assert len(lines) == 5
    assert lines[2].split() == [
        "served",
        "c1",
        "0",
        "2",
        "1",
        "Quillmere=hit",
        "Book",
        "Club=miss",
    ]
    assert lines[4].endswith("error: ModelHTTPError: 503")


def _plan(**overrides: str) -> AuthoringPlan:
    return AuthoringPlan(
        name="t",
        model_id="m",
        runs=2,
        variants=[GuidanceVariant(name="served"), GuidanceVariant(name="proposed")],
        conversations=[CONVERSATION],
        **overrides,
    )


def test_plan_specs_carry_distinct_user_ids_under_the_execution_id() -> None:
    assert [s.user_id for s in _plan(execution_id="x1").specs()] == [
        "probe-authoring-t-x1-served-c1-r0",
        "probe-authoring-t-x1-served-c1-r1",
        "probe-authoring-t-x1-proposed-c1-r0",
        "probe-authoring-t-x1-proposed-c1-r1",
    ]


def test_two_executions_of_one_name_share_no_user_id() -> None:
    first, second = _plan(), _plan()
    assert first.execution_id != second.execution_id
    assert not {s.user_id for s in first.specs()} & {s.user_id for s in second.specs()}
