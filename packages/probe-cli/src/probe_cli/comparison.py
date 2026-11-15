"""Frozen comparison rule: two control seedings against two candidate seedings.

An operational screen, not a significance test: it emits calls, never p-values,
intervals or rates beyond the observed shares. Every measure is higher-is-better.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Sequence

MeasureCall = Literal["regression", "unresolved", "none"]
Confirmation = Literal["confirmed", "unresolved"]
Verdict = Literal[
    "needs_confirmation", "hold_confirmed", "unresolved", "no_regression_called"
]

_SEEDINGS = 2
# Means of per-run ratios at equal totals can differ in the last ulp; real
# differences on these measures are about 0.02 or more, so ties stay ties.
_DIGITS = 9


class UnvalidatedInput(BaseModel):
    # Admitted by --allow-unvalidated: an artifact older than `probe validity`.
    path: str
    state: Literal["absent"] = "absent"


class Seedings[T: BaseModel](BaseModel):
    control: list[T] = Field(min_length=_SEEDINGS, max_length=_SEEDINGS)
    candidate: list[T] = Field(min_length=_SEEDINGS, max_length=_SEEDINGS)
    fresh_control: T | None = None
    fresh_candidate: T | None = None

    @model_validator(mode="after")
    def _fresh_is_whole(self) -> Seedings[T]:
        if (self.fresh_control is None) != (self.fresh_candidate is None):
            raise ValueError("a fresh pair is one control and one candidate seeding")
        return self

    @property
    def has_fresh(self) -> bool:
        return self.fresh_control is not None


class MeasureRead(BaseModel):
    name: str
    control: list[float | None]
    candidate: list[float | None]
    call: MeasureCall
    reason: str = ""
    fresh_control: float | None = None
    fresh_candidate: float | None = None
    confirmation: Confirmation | None = None


def _check_counts(control: Sequence[object], candidate: Sequence[object]) -> None:
    if len(control) != _SEEDINGS or len(candidate) != _SEEDINGS:
        raise ValueError(
            f"expected {_SEEDINGS} control and {_SEEDINGS} candidate values, "
            f"got {len(control)} and {len(candidate)}"
        )


def call_measure(control: Sequence[float], candidate: Sequence[float]) -> MeasureCall:
    # The losing bound is min(control); a tie with it is inside.
    _check_counts(control, candidate)
    bound = round(min(control), _DIGITS)
    outside = sum(1 for value in candidate if round(value, _DIGITS) < bound)
    if outside == _SEEDINGS:
        return "regression"
    return "unresolved" if outside else "none"


def confirm_measure(*, fresh_control: float, fresh_candidate: float) -> Confirmation:
    below = round(fresh_candidate, _DIGITS) < round(fresh_control, _DIGITS)
    return "confirmed" if below else "unresolved"


def confirm_loss(*, fresh_control_has: bool, fresh_candidate_has: bool) -> Confirmation:
    confirmed = fresh_control_has and not fresh_candidate_has
    return "confirmed" if confirmed else "unresolved"


def read_measure(
    name: str,
    control: Sequence[float | None],
    candidate: Sequence[float | None],
    *,
    fresh: tuple[float | None, float | None] | None = None,
) -> MeasureRead:
    _check_counts(control, candidate)
    values = [*control, *candidate]
    read = MeasureRead(
        name=name, control=list(control), candidate=list(candidate), call="none"
    )
    missing = sum(1 for value in values if value is None)
    if missing:
        reason = f"{name} has no value in {missing} of {len(values)} seedings"
        return read.model_copy(update={"call": "unresolved", "reason": reason})
    call = call_measure(
        [v for v in control if v is not None], [v for v in candidate if v is not None]
    )
    read = read.model_copy(update={"call": call})
    if call != "regression" or fresh is None:
        return read
    return _confirmed(read, fresh)


def _confirmed(
    read: MeasureRead, fresh: tuple[float | None, float | None]
) -> MeasureRead:
    fresh_control, fresh_candidate = fresh
    update: dict[str, object] = {
        "fresh_control": fresh_control,
        "fresh_candidate": fresh_candidate,
    }
    if fresh_control is None or fresh_candidate is None:
        update |= {"confirmation": "unresolved", "reason": "fresh pair has no value"}
    else:
        update["confirmation"] = confirm_measure(
            fresh_control=fresh_control, fresh_candidate=fresh_candidate
        )
    return read.model_copy(update=update)


def verdict_of(
    measures: Sequence[MeasureRead],
    losses: Sequence[Confirmation | None],
    *,
    fresh: bool,
) -> Verdict:
    """losses: one confirmation per named loss, None when no fresh pair was read."""
    triggered = bool(losses) or any(m.call == "regression" for m in measures)
    if not triggered:
        # A fresh pair confirms only the items that triggered it: with none,
        # it is not read and nothing is called from one seeding.
        unresolved = any(m.call == "unresolved" for m in measures)
        return "unresolved" if unresolved else "no_regression_called"
    if not fresh:
        return "needs_confirmation"
    confirmed = "confirmed" in losses or any(
        m.confirmation == "confirmed" for m in measures
    )
    return "hold_confirmed" if confirmed else "unresolved"
