"""Verdict oracle: deterministic used-line attribution from answer scoring.

Alias presence in a shown context line, joined with a correct answer, is the
oracle's proxy for "the consumer used this line" -- zero consumer cooperation.
An incorrect or abstained answer attributes nothing: absent evidence is an
honest unknown, never an unused verdict.
"""

from __future__ import annotations

from probe_cli.corpus import AnswerType
from probe_cli.task_runner import (
    TaskQuestionResult,
    TaskRunReport,
    UsedLineAttribution,
    aggregate_verdict_oracle,
    compute_used_lines,
)

_ALIASES = ["6pm", "after 6pm"]


def test_correct_answer_attributes_alias_bearing_trace_line() -> None:
    used = compute_used_lines(
        fact_lines=[],
        trace_lines=[
            ("t-head", "Sofi needs supplies from the Oak Street store"),
            ("t-tail", "Weekday parking on Oak Street is free after 6pm"),
        ],
        aliases=_ALIASES,
        answered_correctly=True,
    )
    assert len(used) == 1
    assert used[0].channel == "trace"
    assert used[0].rank == 2
    assert used[0].trace_id == "t-tail"


def test_fact_line_attributes_with_fact_channel() -> None:
    used = compute_used_lines(
        fact_lines=["Oak Street is_associated_with free parking after 6pm"],
        trace_lines=[],
        aliases=_ALIASES,
        answered_correctly=True,
    )
    assert len(used) == 1
    assert used[0].channel == "fact"
    assert used[0].rank == 1
    assert used[0].trace_id == ""


def test_incorrect_answer_attributes_nothing() -> None:
    """Honest unknown: no verdict without a correct answer, even when a
    shown line carries the alias."""
    used = compute_used_lines(
        fact_lines=["free after 6pm"],
        trace_lines=[("t1", "free after 6pm")],
        aliases=_ALIASES,
        answered_correctly=False,
    )
    assert used == []


def test_correct_answer_without_carrying_line_is_unattributed() -> None:
    """Empty attributions + correct answer = unattributed correct, the
    parametric-knowledge observability case."""
    used = compute_used_lines(
        fact_lines=["Sofi has a science project"],
        trace_lines=[("t1", "the hardware store sells baking soda")],
        aliases=_ALIASES,
        answered_correctly=True,
    )
    assert used == []


def test_alias_matches_at_word_boundary_only() -> None:
    used = compute_used_lines(
        fact_lines=[],
        trace_lines=[("t1", "the meeting runs until 16pm sharp")],
        aliases=_ALIASES,
        answered_correctly=True,
    )
    assert used == []


def test_multiple_carrying_lines_all_attributed_with_ranks() -> None:
    used = compute_used_lines(
        fact_lines=["parking free after 6pm on Oak"],
        trace_lines=[
            ("t1", "no match here"),
            ("t2", "metered until 6pm weekdays"),
        ],
        aliases=_ALIASES,
        answered_correctly=True,
    )
    assert [(u.channel, u.rank) for u in used] == [("fact", 1), ("trace", 2)]


def _result(
    qid: str,
    *,
    correct: bool,
    used: list[UsedLineAttribution],
    answer_type: AnswerType = "time",
) -> TaskQuestionResult:
    return TaskQuestionResult(
        question_id=qid,
        tier_label="t3",
        answer_type=answer_type,
        correct_with=correct,
        context_thoughts=7,
        context_facts=2,
        used_lines=used,
    )


def test_aggregate_counts_surfaced_used_and_unattributed() -> None:
    trace_use = UsedLineAttribution(
        channel="trace", rank=3, line="free after 6pm", trace_id="t9"
    )
    report = TaskRunReport(
        arm_name="a",
        run_index=0,
        model="m",
        answer_model="m",
        density_tier=0,
        seeded_traces=9,
        results=[
            _result("q1", correct=True, used=[trace_use]),
            _result("q2", correct=True, used=[]),
            _result("q3", correct=False, used=[]),
        ],
    )
    oracle = aggregate_verdict_oracle("a", [report])
    assert oracle.thoughts_surfaced == 21
    assert oracle.facts_surfaced == 6
    assert oracle.thoughts_used == 1
    assert oracle.facts_used == 0
    assert oracle.correct_with_total == 2
    assert oracle.unattributed_correct == 1
    assert oracle.used_by_question == {"q1": ["trace:3"]}
