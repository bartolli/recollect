"""P6 situational metrics: action accuracy, distribution, extension target accuracy."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from probe_cli.corpus import ExpectedAction
from probe_cli.situational_runner import EvalResult, SituationalRunReport

_ACTIONS: tuple[ExpectedAction, ...] = ("create", "extend", "revise", "none")


class SituationalMetrics(BaseModel):
    eval_count: int
    success_count: int
    action_accuracy: float = Field(ge=0.0, le=1.0)
    extension_target_accuracy: float = Field(ge=0.0, le=1.0)
    extension_eligible: int
    extension_correct_action_count: int
    expected_distribution: dict[str, int] = Field(default_factory=dict)
    actual_distribution: dict[str, int] = Field(default_factory=dict)
    none_rate_expected: float = Field(ge=0.0, le=1.0)
    none_rate_actual: float = Field(ge=0.0, le=1.0)


class SituationalAggregate(BaseModel):
    arm_name: str
    model: str
    runs: int
    eval_count: int
    action_accuracy_mean: float
    action_accuracy_std: float
    extension_target_accuracy_mean: float
    extension_target_accuracy_std: float
    none_rate_actual_mean: float
    none_rate_expected: float
    expected_distribution: dict[str, int]


def _empty_dist() -> dict[str, int]:
    return dict.fromkeys(_ACTIONS, 0)


def compute_situational_metrics(results: list[EvalResult]) -> SituationalMetrics:
    if not results:
        raise ValueError("results must be non-empty")

    expected_dist = _empty_dist()
    actual_dist = _empty_dist()
    correct = 0
    success = 0
    extension_eligible = 0
    extension_correct_action = 0
    extension_target_hits = 0

    for r in results:
        expected_dist[r.expected_action] += 1
        if not r.success:
            continue
        success += 1
        action = r.actual_action or "none"
        actual_dist[action] += 1
        if action == r.expected_action:
            correct += 1
            if r.expected_action in ("extend", "revise"):
                extension_correct_action += 1
                if r.actual_group_id == r.expected_group_id:
                    extension_target_hits += 1
        if r.expected_action in ("extend", "revise"):
            extension_eligible += 1

    n = len(results)
    target_acc = (
        extension_target_hits / extension_correct_action
        if extension_correct_action > 0
        else 0.0
    )
    return SituationalMetrics(
        eval_count=n,
        success_count=success,
        action_accuracy=correct / n,
        extension_target_accuracy=target_acc,
        extension_eligible=extension_eligible,
        extension_correct_action_count=extension_correct_action,
        expected_distribution=expected_dist,
        actual_distribution=actual_dist,
        none_rate_expected=expected_dist["none"] / n,
        none_rate_actual=actual_dist["none"] / n if success > 0 else 0.0,
    )


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def aggregate_situational_runs(
    arm_name: str, model: str, reports: list[SituationalRunReport]
) -> SituationalAggregate:
    if not reports:
        raise ValueError("reports must be non-empty")
    per_run = [compute_situational_metrics(r.eval_results) for r in reports]
    accs = [m.action_accuracy for m in per_run]
    targets = [m.extension_target_accuracy for m in per_run]
    nones = [m.none_rate_actual for m in per_run]
    return SituationalAggregate(
        arm_name=arm_name,
        model=model,
        runs=len(reports),
        eval_count=per_run[0].eval_count,
        action_accuracy_mean=sum(accs) / len(accs),
        action_accuracy_std=_sample_std(accs),
        extension_target_accuracy_mean=sum(targets) / len(targets),
        extension_target_accuracy_std=_sample_std(targets),
        none_rate_actual_mean=sum(nones) / len(nones),
        none_rate_expected=per_run[0].none_rate_expected,
        expected_distribution=per_run[0].expected_distribution,
    )
