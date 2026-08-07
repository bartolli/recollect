"""Task-eval aggregation: per-label accuracy, memory delta, fabrication rate."""

from __future__ import annotations

import pytest
from probe_cli.task_metrics import aggregate_task_runs
from probe_cli.task_runner import TaskQuestionResult, TaskRunReport


def _res(
    qid: str, label: str, *, with_ok: bool, without_ok: bool
) -> TaskQuestionResult:
    return TaskQuestionResult(
        question_id=qid,
        tier_label=label,  # type: ignore[arg-type]
        answer_type="abstain" if label == "none" else "item",
        correct_with=with_ok,
        correct_without=without_ok,
    )


def _report(run_index: int, results: list[TaskQuestionResult]) -> TaskRunReport:
    return TaskRunReport(
        arm_name="task-test",
        run_index=run_index,
        model="m",
        answer_model="am",
        density_tier=0,
        seeded_traces=10,
        results=results,
    )


def test_aggregate_per_label_and_delta() -> None:
    runs = [
        _report(
            0,
            [
                _res("q1", "t3", with_ok=True, without_ok=False),
                _res("q2", "t3", with_ok=False, without_ok=False),
                _res("q3", "any", with_ok=True, without_ok=True),
            ],
        ),
        _report(
            1,
            [
                _res("q1", "t3", with_ok=True, without_ok=False),
                _res("q2", "t3", with_ok=True, without_ok=False),
                _res("q3", "any", with_ok=True, without_ok=False),
            ],
        ),
    ]
    agg = aggregate_task_runs("task-test", runs)
    assert agg.runs == 2
    t3 = agg.per_label["t3"]
    assert t3.with_memory_mean == pytest.approx(0.75)
    assert t3.without_memory_mean == pytest.approx(0.0)
    assert t3.delta_mean == pytest.approx(0.75)
    assert agg.overall.with_memory_mean == pytest.approx((2 / 3 + 1.0) / 2)


def test_fabrication_rate_from_abstain_questions() -> None:
    runs = [
        _report(
            0,
            [
                _res("a1", "none", with_ok=True, without_ok=True),
                _res("a2", "none", with_ok=False, without_ok=True),
                _res("q1", "any", with_ok=True, without_ok=False),
            ],
        ),
    ]
    agg = aggregate_task_runs("task-test", runs)
    assert agg.fabrication_rate_with == pytest.approx(0.5)
    assert agg.fabrication_rate_without == pytest.approx(0.0)


def test_empty_runs_rejected() -> None:
    with pytest.raises(ValueError, match="no run reports"):
        aggregate_task_runs("task-test", [])
