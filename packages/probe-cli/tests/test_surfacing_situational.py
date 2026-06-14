"""Story-3 spike: situational-grounding lift over the below-floor band."""

from __future__ import annotations

import pytest
from probe_cli.corpus import SeedGroup
from probe_cli.surfacing_runner import (
    QuerySurfacing,
    SurfacedFact,
    SurfacingRunReport,
)
from probe_cli.surfacing_situational import (
    compute_grouped_fact_coverage,
    compute_situational_lift,
)
from recollect.models import PersonaFact


def _f(
    score: float,
    relevant: bool,
    *,
    activated: bool = False,
    trace: str = "t",
    category: str = "",
) -> SurfacedFact:
    return SurfacedFact(
        fact_id="f",
        source_trace_id=trace,
        score=score,
        relevant=relevant,
        source_trace_activated=activated,
        category=category,
    )


def _q(
    qid: str, surfaced: list[SurfacedFact], *, distractor: bool = False
) -> QuerySurfacing:
    return QuerySurfacing(query_id=qid, is_distractor=distractor, surfaced=surfaced)


def _report(queries: list[QuerySurfacing]) -> SurfacingRunReport:
    return SurfacingRunReport(
        arm_name="t", model="m", seeded_traces=1, promoted_facts=1, queries=queries
    )


def test_below_floor_activated_relevant_is_recovered() -> None:
    lift = compute_situational_lift(
        _report([_q("q1", [_f(0.40, True, activated=True)])]), floor=0.65
    )
    assert lift.below_floor_total == 1
    assert lift.below_floor_relevant == 1
    assert lift.activated.n == 1
    assert lift.activated.relevant == 1
    assert lift.activated.precision == pytest.approx(1.0)
    assert lift.recovered_relevant == 1


def test_above_floor_excluded_from_band() -> None:
    # The spike only concerns the recovery band; an above-floor fact (already
    # surfaced by the floor) never enters the cohort even when activated.
    lift = compute_situational_lift(
        _report([
            _q("q1", [
                _f(0.90, True, activated=True),   # above floor -> excluded
                _f(0.40, False, activated=False),  # below floor -> not_activated
            ]),
        ]),
        floor=0.65,
    )
    assert lift.below_floor_total == 1
    assert lift.activated.n == 0
    assert lift.not_activated.n == 1
    assert lift.recovered_relevant == 0


def test_distractor_readmission_and_lift() -> None:
    # On a distractor query every surfaced fact is noise -- a below-floor
    # activated one there is a re-admitted distractor (the ADR no-build trigger),
    # disjoint from the recovered-relevant count on labeled queries.
    lift = compute_situational_lift(
        _report([
            _q("q1", [
                _f(0.40, True, activated=True, trace="a"),
                _f(0.50, False, activated=False, trace="c"),
            ]),
            _q("d1", [_f(0.30, False, activated=True, trace="b")], distractor=True),
        ]),
        floor=0.65,
    )
    assert lift.activated.n == 2
    assert lift.activated.precision == pytest.approx(0.5)
    assert lift.not_activated.precision == pytest.approx(0.0)
    assert lift.lift == pytest.approx(0.5)
    assert lift.recovered_relevant == 1
    assert lift.readmitted_distractor == 1
    assert lift.distinct_activated_traces == 2
    assert lift.queries_with_activation == 2


def test_token_dead_corpus_is_legible_null() -> None:
    # Below-floor facts exist but nothing is token-activated (token-poor corpus):
    # the activated cohort is empty (precision 0.0, no div-by-zero), density
    # counts are 0 -- the spike reads as inconclusive, not false-no-signal.
    lift = compute_situational_lift(
        _report([_q("q1", [_f(0.40, False), _f(0.30, True)])]), floor=0.65
    )
    assert lift.below_floor_total == 2
    assert lift.activated.n == 0
    assert lift.activated.precision == pytest.approx(0.0)
    assert lift.not_activated.precision == pytest.approx(0.5)
    assert lift.lift == pytest.approx(-0.5)
    assert lift.queries_with_activation == 0
    assert lift.distinct_activated_traces == 0
    assert lift.baseline_precision == pytest.approx(0.5)


def test_empty_report_rejected() -> None:
    with pytest.raises(ValueError, match="queries"):
        compute_situational_lift(_report([]), floor=0.65)


def test_below_floor_bypass_category_excluded_from_cohort() -> None:
    # A below-floor health/dietary fact is never dropped by the recall floor
    # (safety bypass surfaces it regardless), so it is not bridge-recoverable --
    # exclude it from the cohort or it pollutes activated-precision.
    lift = compute_situational_lift(
        _report([_q("q1", [
            _f(0.40, True, activated=True, category="health"),
            _f(0.40, True, activated=True, category="constraint"),
        ])]),
        floor=0.65,
    )
    assert lift.below_floor_total == 1
    assert lift.activated.n == 1
    assert lift.recovered_relevant == 1


def test_distractor_bypass_not_counted_as_readmitted() -> None:
    # On a distractor query a bypass fact surfaces anyway; only a bridge-admitted
    # (non-bypass) below-floor fact counts as a re-admission (the ADR no-go).
    lift = compute_situational_lift(
        _report([_q("d1", [
            _f(0.30, False, activated=True, category="health"),
            _f(0.30, False, activated=True, category="general"),
        ], distractor=True)]),
        floor=0.65,
    )
    assert lift.readmitted_distractor == 1


def _pf(source_trace_id: str) -> PersonaFact:
    return PersonaFact(
        subject="s", predicate="p", object="o", content="c",
        source_trace_id=source_trace_id,
    )


def _grp(gid: str, members: list[str]) -> SeedGroup:
    return SeedGroup(
        group_id=gid, person_ref="r", situation="s", implications=["i"],
        significance=0.5, member_trace_ids=members,
    )


def test_grouped_fact_coverage_counts_members_excludes_ungrouped() -> None:
    # Substrate gate: only facts whose source_trace is a token-group member can
    # ever be hop-activated; ungrouped-trace facts can't, so they don't count.
    seed_id_map = {"c1.1": "db1", "c1.2": "db2", "c5.u1": "db9"}
    groups = [_grp("G1", ["c1.1", "c1.2"])]
    facts = [_pf("db1"), _pf("db2"), _pf("db2"), _pf("db9")]
    cov = compute_grouped_fact_coverage(facts, groups, seed_id_map)
    assert cov.total_facts == 4
    assert cov.facts_on_grouped_traces == 3
    assert cov.distinct_grouped_source_traces == 2
    assert cov.per_group["G1"] == 3
