"""Frozen decision rule: the two-seeding measure call, its confirmation, the verdict."""

from __future__ import annotations

import math

import pytest
from probe_cli.comparison import (
    MeasureRead,
    Seedings,
    call_measure,
    confirm_loss,
    confirm_measure,
    read_measure,
    verdict_of,
)
from pydantic import BaseModel, ValidationError


def test_both_candidates_below_the_control_minimum_is_a_regression() -> None:
    assert call_measure([0.80, 0.90], [0.70, 0.79]) == "regression"


def test_one_candidate_below_is_unresolved() -> None:
    assert call_measure([0.80, 0.90], [0.70, 0.85]) == "unresolved"


def test_no_candidate_below_is_none() -> None:
    assert call_measure([0.80, 0.90], [0.85, 0.95]) == "none"


def test_a_tie_with_the_bound_is_inside() -> None:
    assert call_measure([0.80, 0.90], [0.80, 0.80]) == "none"


def test_a_zero_width_control_range_is_that_point() -> None:
    assert call_measure([0.80, 0.80], [0.79, 0.80]) == "unresolved"
    assert call_measure([0.80, 0.80], [0.79, 0.70]) == "regression"


@pytest.mark.parametrize(
    ("control", "candidate"),
    [([0.8], [0.7, 0.7]), ([0.8, 0.9], [0.7]), ([0.8, 0.9, 1.0], [0.7, 0.7])],
)
def test_anything_but_two_and_two_raises(
    control: list[float], candidate: list[float]
) -> None:
    with pytest.raises(ValueError, match="2 control and 2 candidate"):
        call_measure(control, candidate)


def test_a_fresh_candidate_strictly_below_its_control_confirms() -> None:
    assert confirm_measure(fresh_control=0.80, fresh_candidate=0.79) == "confirmed"
    assert confirm_measure(fresh_control=0.80, fresh_candidate=0.80) == "unresolved"
    assert confirm_measure(fresh_control=0.80, fresh_candidate=0.81) == "unresolved"


def test_one_ulp_below_is_a_tie_and_one_millionth_is_a_difference() -> None:
    ulp_below = math.nextafter(0.8, 0.0)
    micro_below = 0.8 - 1e-6

    assert call_measure([0.8, 0.9], [ulp_below, ulp_below]) == "none"
    assert confirm_measure(fresh_control=0.8, fresh_candidate=ulp_below) == (
        "unresolved"
    )
    assert call_measure([0.8, 0.9], [micro_below, micro_below]) == "regression"
    assert confirm_measure(fresh_control=0.8, fresh_candidate=micro_below) == (
        "confirmed"
    )


def test_equal_totals_over_three_runs_read_as_a_tie() -> None:
    # Averaged as aggregate_situational_runs does: 36 of 45 correct both ways.
    even = sum([12 / 15, 12 / 15, 12 / 15]) / 3
    spread = sum([10 / 15, 11 / 15, 15 / 15]) / 3

    assert spread < even
    assert call_measure([even, 0.9], [spread, spread]) == "none"
    assert confirm_measure(fresh_control=even, fresh_candidate=spread) == "unresolved"


def test_a_named_loss_recurs_iff_the_fresh_control_has_it_and_the_candidate_not() -> (
    None
):
    assert confirm_loss(fresh_control_has=True, fresh_candidate_has=False) == (
        "confirmed"
    )
    assert confirm_loss(fresh_control_has=True, fresh_candidate_has=True) == (
        "unresolved"
    )
    assert confirm_loss(fresh_control_has=False, fresh_candidate_has=False) == (
        "unresolved"
    )


def test_a_missing_value_in_any_seeding_is_unresolved_with_a_reason() -> None:
    m = read_measure("precision_floor", [0.9, None], [0.5, 0.5])

    assert m.call == "unresolved"
    assert "1 of 4" in m.reason


def test_only_a_regression_reads_the_fresh_pair() -> None:
    regressed = read_measure("c", [0.8, 0.9], [0.7, 0.7], fresh=(0.8, 0.7))
    unresolved = read_measure("c", [0.8, 0.9], [0.7, 0.9], fresh=(0.8, 0.7))
    no_fresh_value = read_measure("c", [0.8, 0.9], [0.7, 0.7], fresh=(0.8, None))

    assert regressed.confirmation == "confirmed"
    assert (unresolved.confirmation, unresolved.fresh_control) == (None, None)
    assert no_fresh_value.confirmation == "unresolved"
    assert "fresh" in no_fresh_value.reason


def _m(call: str, confirmation: str | None = None) -> MeasureRead:
    return MeasureRead.model_validate(
        {"name": "m", "control": [], "candidate": [], "call": call}
        | {"confirmation": confirmation}
    )


def test_verdicts() -> None:
    assert verdict_of([_m("regression")], [], fresh=False) == "needs_confirmation"
    assert verdict_of([_m("none")], [None], fresh=False) == "needs_confirmation"
    assert verdict_of([_m("regression", "confirmed")], [], fresh=True) == (
        "hold_confirmed"
    )
    assert verdict_of([_m("none")], ["confirmed"], fresh=True) == "hold_confirmed"
    assert verdict_of([_m("regression", "unresolved")], ["unresolved"], fresh=True) == (
        "unresolved"
    )
    assert verdict_of([_m("unresolved")], [], fresh=False) == "unresolved"
    assert verdict_of([_m("none")], [], fresh=False) == "no_regression_called"


def test_a_fresh_pair_without_a_trigger_is_not_read() -> None:
    assert verdict_of([_m("none")], [], fresh=True) == "no_regression_called"
    assert verdict_of([_m("unresolved")], [], fresh=True) == "unresolved"


class _V(BaseModel):
    v: float


def test_seedings_are_two_and_two_with_a_whole_fresh_pair() -> None:
    two = [_V(v=1), _V(v=2)]
    Seedings[_V](
        control=two, candidate=two, fresh_control=_V(v=1), fresh_candidate=_V(v=1)
    )

    with pytest.raises(ValidationError):
        Seedings[_V](control=two[:1], candidate=two)
    with pytest.raises(ValidationError, match="fresh pair"):
        Seedings[_V](control=two, candidate=two, fresh_control=_V(v=1))
