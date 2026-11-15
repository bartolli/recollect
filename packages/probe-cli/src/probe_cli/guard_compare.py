"""Situational guard read of the prompt candidate against two control seedings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from probe_cli.comparison import (
    MeasureRead,
    Seedings,
    UnvalidatedInput,
    Verdict,
    read_measure,
    verdict_of,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from probe_cli.situational_metrics import SituationalAggregate

# The fixture expects `none` on 6 of 15 entries: an assessor that forms no group
# scores 0.40 on action accuracy, so a control at or below this cannot show a loss.
UNINFORMATIVE_AT_OR_BELOW = 0.47

GuardVerdict = Literal["uninformative"] | Verdict


class GuardSeeding(BaseModel):
    label: str
    arm_name: str
    model: str
    runs: int = Field(ge=0)
    action_accuracy: float = Field(ge=0.0, le=1.0)
    # 0.0 also when no extend or revise action was correct.
    extension_target_accuracy: float = Field(ge=0.0, le=1.0)
    # Shown beside the accuracies; calls nothing.
    none_rate: float = Field(ge=0.0, le=1.0)
    none_rate_expected: float = Field(ge=0.0, le=1.0)


class GuardComparison(BaseModel):
    uninformative_at_or_below: float
    seedings: Seedings[GuardSeeding]
    measures: list[MeasureRead]
    verdict: GuardVerdict
    unvalidated_inputs: list[UnvalidatedInput] = Field(default_factory=list)


def guard_seeding(label: str, aggregate: SituationalAggregate) -> GuardSeeding:
    return GuardSeeding(
        label=label,
        arm_name=aggregate.arm_name,
        model=aggregate.model,
        runs=aggregate.runs,
        action_accuracy=aggregate.action_accuracy_mean,
        extension_target_accuracy=aggregate.extension_target_accuracy_mean,
        none_rate=aggregate.none_rate_actual_mean,
        none_rate_expected=aggregate.none_rate_expected,
    )


_MEASURES: dict[str, Callable[[GuardSeeding], float]] = {
    "action_accuracy": lambda s: s.action_accuracy,
    "extension_target_accuracy": lambda s: s.extension_target_accuracy,
}


def _measure(
    name: str, value: Callable[[GuardSeeding], float], seedings: Seedings[GuardSeeding]
) -> MeasureRead:
    fresh = None
    if seedings.fresh_control is not None and seedings.fresh_candidate is not None:
        fresh = (value(seedings.fresh_control), value(seedings.fresh_candidate))
    return read_measure(
        name,
        [value(s) for s in seedings.control],
        [value(s) for s in seedings.candidate],
        fresh=fresh,
    )


def compare_guard(seedings: Seedings[GuardSeeding]) -> GuardComparison:
    uninformative = any(
        s.action_accuracy <= UNINFORMATIVE_AT_OR_BELOW for s in seedings.control
    )
    measures = (
        []
        if uninformative
        else [_measure(name, value, seedings) for name, value in _MEASURES.items()]
    )
    verdict: GuardVerdict = (
        "uninformative"
        if uninformative
        else verdict_of(measures, [], fresh=seedings.has_fresh)
    )
    return GuardComparison(
        uninformative_at_or_below=UNINFORMATIVE_AT_OR_BELOW,
        seedings=seedings,
        measures=measures,
        verdict=verdict,
    )
