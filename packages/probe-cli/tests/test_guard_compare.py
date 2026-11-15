"""Situational guard: informativeness floor, calls on two accuracies, verdicts."""

from __future__ import annotations

from probe_cli.comparison import Seedings
from probe_cli.guard_compare import (
    GuardComparison,
    GuardSeeding,
    compare_guard,
    guard_seeding,
)
from probe_cli.situational_metrics import SituationalAggregate


def _agg(
    action: float, target: float = 1.0, none_rate: float = 0.4
) -> SituationalAggregate:
    return SituationalAggregate(
        arm_name="arm",
        model="m",
        runs=3,
        eval_count=15,
        action_accuracy_mean=action,
        action_accuracy_std=0.01,
        extension_target_accuracy_mean=target,
        extension_target_accuracy_std=0.0,
        none_rate_actual_mean=none_rate,
        none_rate_expected=0.4,
        expected_distribution={"create": 1, "extend": 5, "revise": 3, "none": 6},
    )


def _compare(
    control: list[SituationalAggregate],
    candidate: list[SituationalAggregate],
    fresh: tuple[SituationalAggregate, SituationalAggregate] | None = None,
) -> GuardComparison:
    def seed(agg: SituationalAggregate | None, label: str) -> GuardSeeding | None:
        return None if agg is None else guard_seeding(label, agg)

    return compare_guard(
        Seedings[GuardSeeding].model_validate(
            {
                "control": [seed(a, f"c{i}") for i, a in enumerate(control)],
                "candidate": [seed(a, f"k{i}") for i, a in enumerate(candidate)],
                "fresh_control": seed(fresh[0] if fresh else None, "fc"),
                "fresh_candidate": seed(fresh[1] if fresh else None, "fk"),
            }
        )
    )


def _calls(c: GuardComparison) -> dict[str, str]:
    return {m.name: m.call for m in c.measures}


def test_a_control_at_exactly_047_is_uninformative_and_calls_nothing() -> None:
    c = _compare([_agg(0.47), _agg(0.9)], [_agg(0.1), _agg(0.1)])

    assert c.verdict == "uninformative"
    assert c.measures == []


def test_a_control_at_048_is_informative() -> None:
    c = _compare([_agg(0.48), _agg(0.9)], [_agg(0.9), _agg(0.9)])

    assert c.verdict == "no_regression_called"
    assert set(_calls(c)) == {"action_accuracy", "extension_target_accuracy"}


def test_seeding_values_carry_the_none_rate_beside_the_accuracies() -> None:
    s = guard_seeding("dir", _agg(0.9, 0.8, none_rate=0.33))

    assert (s.action_accuracy, s.extension_target_accuracy, s.none_rate) == (
        0.9,
        0.8,
        0.33,
    )
    assert (s.label, s.arm_name, s.runs) == ("dir", "arm", 3)


def test_both_candidates_below_on_either_accuracy_need_confirmation() -> None:
    c = _compare([_agg(0.9, 1.0), _agg(0.95, 0.9)], [_agg(0.9, 0.8), _agg(0.9, 0.7)])

    assert _calls(c) == {
        "action_accuracy": "none",
        "extension_target_accuracy": "regression",
    }
    assert c.verdict == "needs_confirmation"


def test_one_candidate_below_is_unresolved() -> None:
    c = _compare([_agg(0.9), _agg(0.95)], [_agg(0.8), _agg(0.95)])

    assert _calls(c)["action_accuracy"] == "unresolved"
    assert c.verdict == "unresolved"


def test_a_fresh_pair_with_the_candidate_strictly_below_holds() -> None:
    c = _compare(
        [_agg(0.9), _agg(0.95)], [_agg(0.8), _agg(0.7)], fresh=(_agg(0.9), _agg(0.8))
    )

    assert c.verdict == "hold_confirmed"


def test_a_fresh_pair_that_ties_is_unresolved() -> None:
    c = _compare(
        [_agg(0.9), _agg(0.95)], [_agg(0.8), _agg(0.7)], fresh=(_agg(0.9), _agg(0.9))
    )

    assert c.measures[0].confirmation == "unresolved"
    assert c.verdict == "unresolved"
