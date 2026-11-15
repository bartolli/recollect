"""Surfacing read of the prompt candidate against two control seedings.

Coverage is the rank cut and then the gate: the surfaced list is already the top
`max_facts_per_query` of the force-promoted pool, and a pair is covered when a
surfaced fact from its required trace passes the recall gate.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from probe_cli.comparison import (
    Confirmation,
    MeasureRead,
    Seedings,
    UnvalidatedInput,
    Verdict,
    confirm_loss,
    read_measure,
    verdict_of,
)
from probe_cli.surfacing_metrics import (
    QueryCoverage,
    SurfacingMetrics,
    compute_required_coverage,
)
from probe_cli.surfacing_runner import FactPopulation, SurfacingRunReport

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence
    from pathlib import Path

    from probe_cli.corpus import QueryEntry

UncoveredCause = Literal[
    "query_failed", "no_fact", "ranked_out", "under_floor", "unknown"
]
Pair = tuple[str, str]


class SurfacingReading(BaseModel):
    queries: str
    overlap_pairs: list[Pair] = Field(default_factory=list)


class SurfacingInput(BaseModel):
    label: str
    report: SurfacingRunReport
    summary: SurfacingMetrics | None = None


class PairState(BaseModel):
    query_id: str
    required_id: str
    covered: bool
    cause: UncoveredCause | None = None


class GatePasses(BaseModel):
    # floor pass: passes_gate and score >= floor; bypass pass: passes_gate below it.
    floor_passes: int = Field(ge=0)
    relevant_floor_passes: int = Field(ge=0)
    bypass_passes: int = Field(ge=0)
    relevant_bypass_passes: int = Field(ge=0)


class ScoreMeans(BaseModel):
    relevant: float
    distractor: float


class SurfacingSeeding(BaseModel):
    label: str
    arm_name: str
    pairs: list[PairState]
    covered_all: int = Field(ge=0)
    pairs_all: int = Field(ge=0)
    coverage_all: float = Field(ge=0.0, le=1.0)
    covered_without_overlap: int = Field(ge=0)
    pairs_without_overlap: int = Field(ge=0)
    coverage_without_overlap: float = Field(ge=0.0, le=1.0)
    gate: GatePasses
    precision_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    # Descriptive: no losing side, calls nothing.
    fact_population: FactPopulation
    score_means: ScoreMeans | None = None


def load_reading(path: Path) -> SurfacingReading:
    return SurfacingReading.model_validate(tomllib.loads(path.read_text("utf-8")))


def required_pairs(queries: Sequence[QueryEntry]) -> list[Pair]:
    return [(q.id, rid) for q in queries for rid in q.relevant_trace_ids]


def _cause(
    row: QueryCoverage, required_id: str, report: SurfacingRunReport
) -> UncoveredCause:
    answered = {q.query_id for q in report.queries} - set(report.failed_query_ids)
    if row.query_id not in answered:
        return "query_failed"
    # An empty pool marks a report recorded before the pool was read.
    if not report.surfacing_pool:
        return "unknown"
    if required_id in row.returned:
        return "under_floor"
    pooled = {f.source_eval_id for f in report.surfacing_pool}
    return "ranked_out" if required_id in pooled else "no_fact"


def _pair_states(
    rows: Sequence[QueryCoverage], report: SurfacingRunReport
) -> list[PairState]:
    return [
        PairState(
            query_id=row.query_id,
            required_id=rid,
            covered=rid in row.covered,
            cause=None if rid in row.covered else _cause(row, rid, report),
        )
        for row in rows
        for rid in row.required
    ]


def _gate_passes(report: SurfacingRunReport, floor: float) -> GatePasses:
    gated = [f for q in report.queries for f in q.surfaced if f.passes_gate]
    above = [f for f in gated if f.score >= floor]
    below = [f for f in gated if f.score < floor]
    return GatePasses(
        floor_passes=len(above),
        relevant_floor_passes=sum(1 for f in above if f.relevant),
        bypass_passes=len(below),
        relevant_bypass_passes=sum(1 for f in below if f.relevant),
    )


def _share(covered: int, total: int) -> float:
    return covered / total if total else 0.0


def _score_means(summary: SurfacingMetrics | None) -> ScoreMeans | None:
    if summary is None:
        return None
    return ScoreMeans(
        relevant=summary.relevant.mean, distractor=summary.distractor.mean
    )


def read_seeding(
    seeding: SurfacingInput,
    queries: Sequence[QueryEntry],
    *,
    floor: float,
    overlap: Collection[Pair],
) -> SurfacingSeeding:
    report = seeding.report
    coverage = compute_required_coverage(report, queries, floor=floor)
    pairs = _pair_states(coverage.queries, report)
    kept = [p for p in pairs if (p.query_id, p.required_id) not in overlap]
    kept_covered = sum(1 for p in kept if p.covered)
    gate = _gate_passes(report, floor)
    return SurfacingSeeding(
        label=seeding.label,
        arm_name=report.arm_name,
        pairs=pairs,
        covered_all=coverage.covered_pairs,
        pairs_all=coverage.required_pairs,
        coverage_all=coverage.coverage,
        covered_without_overlap=kept_covered,
        pairs_without_overlap=len(kept),
        coverage_without_overlap=_share(kept_covered, len(kept)),
        gate=gate,
        precision_floor=(
            gate.relevant_floor_passes / gate.floor_passes
            if gate.floor_passes
            else None
        ),
        fact_population=report.fact_population,
        score_means=_score_means(seeding.summary),
    )


class NamedLoss(BaseModel):
    query_id: str
    required_id: str
    # One per candidate seeding, in input order.
    candidate_causes: list[UncoveredCause | None]
    confirmation: Confirmation | None = None


class NamedGain(BaseModel):
    # Calls nothing; a gain on an exemplar-overlap pair is not evidence for the
    # candidate.
    query_id: str
    required_id: str
    exemplar_overlap: bool


class SurfacingComparison(BaseModel):
    floor: float
    reading: SurfacingReading
    seedings: Seedings[SurfacingSeeding]
    measures: list[MeasureRead]
    named_losses: list[NamedLoss]
    named_gains: list[NamedGain]
    verdict: Verdict
    unvalidated_inputs: list[UnvalidatedInput] = Field(default_factory=list)


_MEASURES: dict[str, Callable[[SurfacingSeeding], float | None]] = {
    "coverage_all": lambda s: s.coverage_all,
    "coverage_without_overlap": lambda s: s.coverage_without_overlap,
    "precision_floor": lambda s: s.precision_floor,
}


def _measure(
    name: str,
    value: Callable[[SurfacingSeeding], float | None],
    seedings: Seedings[SurfacingSeeding],
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


def _covered(seeding: SurfacingSeeding) -> set[Pair]:
    return {(p.query_id, p.required_id) for p in seeding.pairs if p.covered}


def _cause_of(seeding: SurfacingSeeding, pair: Pair) -> UncoveredCause | None:
    return next(p.cause for p in seeding.pairs if (p.query_id, p.required_id) == pair)


def _loss_confirmation(
    seedings: Seedings[SurfacingSeeding], pair: Pair
) -> Confirmation | None:
    if seedings.fresh_control is None or seedings.fresh_candidate is None:
        return None
    return confirm_loss(
        fresh_control_has=pair in _covered(seedings.fresh_control),
        fresh_candidate_has=pair in _covered(seedings.fresh_candidate),
    )


def _in_both_not_either(
    held: Sequence[SurfacingSeeding], absent: Sequence[SurfacingSeeding]
) -> list[Pair]:
    both = _covered(held[0]).intersection(*(_covered(s) for s in held[1:]))
    found = both.difference(*(_covered(s) for s in absent))
    order = [(p.query_id, p.required_id) for p in held[0].pairs]
    return [pair for pair in order if pair in found]


def _named_losses(seedings: Seedings[SurfacingSeeding]) -> list[NamedLoss]:
    return [
        NamedLoss(
            query_id=pair[0],
            required_id=pair[1],
            candidate_causes=[_cause_of(s, pair) for s in seedings.candidate],
            confirmation=_loss_confirmation(seedings, pair),
        )
        for pair in _in_both_not_either(seedings.control, seedings.candidate)
    ]


def _named_gains(
    seedings: Seedings[SurfacingSeeding], overlap: set[Pair]
) -> list[NamedGain]:
    return [
        NamedGain(
            query_id=pair[0], required_id=pair[1], exemplar_overlap=pair in overlap
        )
        for pair in _in_both_not_either(seedings.candidate, seedings.control)
    ]


def _check_overlap(reading: SurfacingReading, queries: Sequence[QueryEntry]) -> None:
    # A mistyped overlap pair would silently make both coverages equal.
    unknown = sorted(set(reading.overlap_pairs) - set(required_pairs(queries)))
    if unknown:
        raise ValueError(f"overlap pairs absent from the query labels: {unknown}")


def _recorded_floor(seeding: SurfacingInput) -> float | None:
    if seeding.summary is None or seeding.summary.required_coverage is None:
        return None
    return seeding.summary.required_coverage.floor


def _check_floor(inputs: Seedings[SurfacingInput], floor: float) -> None:
    # passes_gate was decided at the run's floor: split at another floor, floor
    # passes and bypass passes would be miscounted.
    fresh = [s for s in (inputs.fresh_control, inputs.fresh_candidate) if s is not None]
    differ = [
        f"{s.label} (recorded {recorded})"
        for s in [*inputs.control, *inputs.candidate, *fresh]
        if (recorded := _recorded_floor(s)) is not None and recorded != floor
    ]
    if differ:
        raise ValueError(
            f"recorded floor differs from the config floor {floor}: {differ}"
        )


def compare_surfacing(
    inputs: Seedings[SurfacingInput],
    queries: Sequence[QueryEntry],
    *,
    reading: SurfacingReading,
    floor: float,
) -> SurfacingComparison:
    _check_overlap(reading, queries)
    _check_floor(inputs, floor)

    def read(s: SurfacingInput | None) -> SurfacingSeeding | None:
        if s is None:
            return None
        return read_seeding(s, queries, floor=floor, overlap=reading.overlap_pairs)

    seedings = Seedings[SurfacingSeeding].model_validate(
        {
            "control": [read(s) for s in inputs.control],
            "candidate": [read(s) for s in inputs.candidate],
            "fresh_control": read(inputs.fresh_control),
            "fresh_candidate": read(inputs.fresh_candidate),
        }
    )
    measures = [_measure(name, value, seedings) for name, value in _MEASURES.items()]
    losses = _named_losses(seedings)
    return SurfacingComparison(
        floor=floor,
        reading=reading,
        seedings=seedings,
        measures=measures,
        named_losses=losses,
        named_gains=_named_gains(seedings, set(reading.overlap_pairs)),
        verdict=verdict_of(
            measures, [loss.confirmation for loss in losses], fresh=seedings.has_fresh
        ),
    )
