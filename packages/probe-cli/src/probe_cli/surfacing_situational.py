"""Story-3 spike metric: does source_trace token-activation predict relevance?

Restricts to the below-floor band -- the facts the flat recall floor drops --
and asks whether token-activated source_traces there carry more relevance than
unactivated ones (the recall-recovery signal) without re-admitting distractor
noise. readmitted_distractor (below-floor activated facts on distractor queries)
is the ADR no-build trigger; queries_with_activation + distinct_activated_traces
report substrate density so a token-poor corpus reads as a legible null, not a
false no-signal. Probe-side and conservative: the seed omits think_about's
entity supplement, so it undercounts activation -- a positive signal is
trustworthy, only a null is ambiguous.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from probe_cli.surfacing_runner import SurfacedFact, SurfacingRunReport

if TYPE_CHECKING:
    from recollect.models import PersonaFact

    from probe_cli.corpus import SeedGroup


class GroupedFactCoverage(BaseModel):
    # Substrate gate for slice-1b: a fact can only be hop-activated if its
    # source_trace is a token-group member. Coverage near zero => wrong
    # substrate, no query design recovers it (slice 1's lesson, one level up).
    total_facts: int
    facts_on_grouped_traces: int
    distinct_grouped_source_traces: int
    per_group: dict[str, int] = Field(default_factory=dict)


def compute_grouped_fact_coverage(
    facts: list[PersonaFact],
    groups: list[SeedGroup],
    seed_id_map: dict[str, str],
) -> GroupedFactCoverage:
    # seed_id_map: eval-corpus id -> db trace id. Group membership is authored
    # in eval ids; map to db ids to match fact.source_trace_id.
    group_db_members = {
        g.group_id: {seed_id_map[m] for m in g.member_trace_ids if m in seed_id_map}
        for g in groups
    }
    all_grouped: set[str] = set().union(*group_db_members.values()) if groups else set()
    on_grouped = [f for f in facts if f.source_trace_id in all_grouped]
    return GroupedFactCoverage(
        total_facts=len(facts),
        facts_on_grouped_traces=len(on_grouped),
        distinct_grouped_source_traces=len(
            {f.source_trace_id for f in on_grouped if f.source_trace_id}
        ),
        per_group={
            gid: sum(1 for f in facts if f.source_trace_id in members)
            for gid, members in group_db_members.items()
        },
    )


class SituationalCohort(BaseModel):
    n: int
    relevant: int
    precision: float = Field(ge=0.0, le=1.0)


class SituationalLift(BaseModel):
    floor: float
    below_floor_total: int
    below_floor_relevant: int
    baseline_precision: float = Field(ge=0.0, le=1.0)
    activated: SituationalCohort
    not_activated: SituationalCohort
    lift: float
    recovered_relevant: int
    readmitted_distractor: int
    queries_with_activation: int
    distinct_activated_traces: int


def _cohort(facts: list[SurfacedFact]) -> SituationalCohort:
    n = len(facts)
    rel = sum(1 for f in facts if f.relevant)
    return SituationalCohort(n=n, relevant=rel, precision=rel / n if n else 0.0)


# Recall safety-bypass (core._RECALL_SAFETY_BYPASS): these surface below the
# floor regardless of activation, so they are never bridge-recoverable.
_BYPASS: frozenset[str] = frozenset({"health", "dietary"})


def compute_situational_lift(
    report: SurfacingRunReport, *, floor: float
) -> SituationalLift:
    if not report.queries:
        raise ValueError("report must contain queries")

    below = [
        f
        for q in report.queries
        for f in q.surfaced
        if f.score < floor and f.category not in _BYPASS
    ]
    activated = _cohort([f for f in below if f.source_trace_activated])
    not_activated = _cohort([f for f in below if not f.source_trace_activated])
    below_relevant = sum(1 for f in below if f.relevant)

    readmitted = sum(
        1
        for q in report.queries
        if q.is_distractor
        for f in q.surfaced
        if f.score < floor
        and f.source_trace_activated
        and f.category not in _BYPASS
    )
    queries_with_activation = sum(
        1
        for q in report.queries
        if any(f.source_trace_activated for f in q.surfaced)
    )
    distinct_traces = {
        f.source_trace_id
        for q in report.queries
        for f in q.surfaced
        if f.source_trace_activated and f.source_trace_id is not None
    }

    return SituationalLift(
        floor=floor,
        below_floor_total=len(below),
        below_floor_relevant=below_relevant,
        baseline_precision=below_relevant / len(below) if below else 0.0,
        activated=activated,
        not_activated=not_activated,
        lift=activated.precision - not_activated.precision,
        recovered_relevant=sum(
            1 for f in below if f.source_trace_activated and f.relevant
        ),
        readmitted_distractor=readmitted,
        queries_with_activation=queries_with_activation,
        distinct_activated_traces=len(distinct_traces),
    )
