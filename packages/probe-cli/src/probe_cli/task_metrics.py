"""Task-eval aggregation: per-label accuracy, memory delta, fabrication rate."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from probe_cli.task_runner import TaskQuestionResult, TaskRunReport


class LabelAccuracy(BaseModel):
    with_memory_mean: float = 0.0
    without_memory_mean: float = 0.0
    delta_mean: float = 0.0
    questions_per_run: int = 0


class TaskAggregate(BaseModel):
    arm_name: str
    runs: int
    per_label: dict[str, LabelAccuracy] = Field(default_factory=dict)
    overall: LabelAccuracy = Field(default_factory=LabelAccuracy)
    # Fabrication = 1 - accuracy on abstain questions: a concrete value where
    # the corpus holds none. Memory context must not raise it.
    fabrication_rate_with: float = 0.0
    fabrication_rate_without: float = 0.0


def _accuracy(results: list[TaskQuestionResult], *, with_memory: bool) -> float:
    if not results:
        return 0.0
    hits = sum(
        1 for r in results if (r.correct_with if with_memory else r.correct_without)
    )
    return hits / len(results)


def _label_accuracy(
    reports: list[TaskRunReport],
    select: str | None,
) -> LabelAccuracy:
    per_run_with: list[float] = []
    per_run_without: list[float] = []
    n = 0
    for report in reports:
        subset = [r for r in report.results if select is None or r.tier_label == select]
        n = max(n, len(subset))
        per_run_with.append(_accuracy(subset, with_memory=True))
        per_run_without.append(_accuracy(subset, with_memory=False))
    with_mean = statistics.mean(per_run_with)
    without_mean = statistics.mean(per_run_without)
    return LabelAccuracy(
        with_memory_mean=with_mean,
        without_memory_mean=without_mean,
        delta_mean=with_mean - without_mean,
        questions_per_run=n,
    )


def aggregate_task_runs(arm_name: str, reports: list[TaskRunReport]) -> TaskAggregate:
    if not reports:
        raise ValueError("no run reports to aggregate")
    labels = sorted({r.tier_label for rep in reports for r in rep.results})
    abstain_rates_with: list[float] = []
    abstain_rates_without: list[float] = []
    for report in reports:
        abstain = [r for r in report.results if r.answer_type == "abstain"]
        if abstain:
            abstain_rates_with.append(1.0 - _accuracy(abstain, with_memory=True))
            abstain_rates_without.append(1.0 - _accuracy(abstain, with_memory=False))
    return TaskAggregate(
        arm_name=arm_name,
        runs=len(reports),
        per_label={label: _label_accuracy(reports, label) for label in labels},
        overall=_label_accuracy(reports, None),
        fabrication_rate_with=(
            statistics.mean(abstain_rates_with) if abstain_rates_with else 0.0
        ),
        fabrication_rate_without=(
            statistics.mean(abstain_rates_without) if abstain_rates_without else 0.0
        ),
    )
