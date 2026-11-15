"""Surfacing read per seeding: required pairs, coverage, causes, floor precision."""

from __future__ import annotations

import pytest
from probe_cli.corpus import QueryEntry
from probe_cli.surfacing_compare import (
    SurfacingInput,
    SurfacingSeeding,
    read_seeding,
)
from probe_cli.surfacing_metrics import (
    ScoreStats,
    SurfacingMetrics,
    compute_required_coverage,
)
from probe_cli.surfacing_runner import (
    FactPopulation,
    PoolFact,
    QuerySurfacing,
    SurfacedFact,
    SurfacingRunReport,
)

_FLOOR = 0.65
_QUERIES = [
    QueryEntry(id="q1", text="", relevant_trace_ids=["a"]),
    QueryEntry(id="q2", text="", relevant_trace_ids=["b", "c"]),
    QueryEntry(id="q3", text=""),
    QueryEntry(id="q4", text="", relevant_trace_ids=["d"]),
]


def _f(
    src: str, score: float = 0.8, *, gate: bool = True, relevant: bool = True
) -> SurfacedFact:
    return SurfacedFact(
        fact_id=f"f-{src}",
        source_eval_id=src,
        score=score,
        relevant=relevant,
        passes_gate=gate,
    )


def _report(
    queries: list[QuerySurfacing], pool: list[str], **extra: object
) -> SurfacingRunReport:
    return SurfacingRunReport.model_validate(
        {"arm_name": "arm", "model": "m", "seeded_traces": 4, "promoted_facts": 0}
        | {
            "queries": queries,
            "surfacing_pool": [PoolFact(fact_id=s, source_eval_id=s) for s in pool],
        }
        | extra
    )


def _causes_report() -> SurfacingRunReport:
    return _report(
        [
            QuerySurfacing(query_id="q2", surfaced=[_f("b", 0.5, gate=False)]),
            QuerySurfacing(query_id="q3", surfaced=[_f("x", 0.9, relevant=False)]),
            QuerySurfacing(query_id="q4", surfaced=[_f("d")]),
        ],
        pool=["b", "d", "x"],
    )


def _read(
    report: SurfacingRunReport,
    overlap: list[tuple[str, str]] | None = None,
    summary: SurfacingMetrics | None = None,
) -> SurfacingSeeding:
    seeding = SurfacingInput(label="s", report=report, summary=summary)
    return read_seeding(seeding, _QUERIES, floor=_FLOOR, overlap=overlap or [])


def test_coverage_totals_equal_compute_required_coverage() -> None:
    report = _causes_report()

    s = _read(report)
    c = compute_required_coverage(report, _QUERIES, floor=_FLOOR)

    assert (s.covered_all, s.pairs_all) == (c.covered_pairs, c.required_pairs) == (1, 4)
    assert s.coverage_all == pytest.approx(c.coverage)


def test_each_uncovered_pair_names_its_cause_from_the_artifact() -> None:
    s = _read(_causes_report())

    causes = {(p.query_id, p.required_id): (p.covered, p.cause) for p in s.pairs}
    assert causes == {
        ("q1", "a"): (False, "query_failed"),
        ("q2", "b"): (False, "under_floor"),
        ("q2", "c"): (False, "no_fact"),
        ("q4", "d"): (True, None),
    }


def test_a_listed_failure_and_a_pooled_unsurfaced_source() -> None:
    report = _report(
        [QuerySurfacing(query_id="q1"), QuerySurfacing(query_id="q4")],
        pool=["a", "d"],
        failed_query_ids=["q1"],
    )

    s = _read(report)

    causes = {p.required_id: p.cause for p in s.pairs}
    assert (causes["a"], causes["d"]) == ("query_failed", "ranked_out")


def test_an_empty_pool_leaves_the_cause_unknown() -> None:
    report = _causes_report().model_copy(update={"surfacing_pool": []})

    s = _read(report)

    causes = {p.required_id: p.cause for p in s.pairs}
    assert causes == {"a": "query_failed", "b": "unknown", "c": "unknown", "d": None}


def test_coverage_is_reported_with_and_without_the_overlap_pairs() -> None:
    s = _read(_causes_report(), [("q2", "b"), ("q2", "c")])

    assert (s.covered_all, s.pairs_all) == (1, 4)
    assert (s.covered_without_overlap, s.pairs_without_overlap) == (1, 2)
    assert s.coverage_without_overlap == pytest.approx(0.5)


def test_floor_and_bypass_passes_are_counted_apart_and_sum_to_the_gated_facts() -> None:
    report = _report(
        [
            QuerySurfacing(
                query_id="q1", surfaced=[_f("a"), _f("h", 0.3, relevant=False)]
            ),
            QuerySurfacing(query_id="q3", surfaced=[_f("x", 0.7, relevant=False)]),
            QuerySurfacing(
                query_id="q4", surfaced=[_f("d", 0.4), _f("y", 0.6, gate=False)]
            ),
        ],
        pool=["a", "d", "h", "x", "y"],
    )

    s = _read(report)
    c = compute_required_coverage(report, _QUERIES, floor=_FLOOR)

    g = s.gate
    assert (g.floor_passes, g.relevant_floor_passes) == (2, 1)
    assert (g.bypass_passes, g.relevant_bypass_passes) == (2, 1)
    assert g.floor_passes + g.bypass_passes == c.gated_facts
    assert g.relevant_floor_passes + g.relevant_bypass_passes == c.gated_relevant
    assert s.precision_floor == pytest.approx(0.5)


def test_no_floor_pass_leaves_precision_none() -> None:
    report = _report(
        [QuerySurfacing(query_id="q1", surfaced=[_f("a", 0.3)])], pool=["a"]
    )

    s = _read(report)

    assert s.precision_floor is None
    assert s.gate.relevant_bypass_passes == 1


def test_population_and_score_means_are_copied_through() -> None:
    population = FactPopulation(total=3, by_status={"promoted": 1, "candidate": 2})
    report = _causes_report().model_copy(update={"fact_population": population})
    stats = ScoreStats(n=1, mean=0.7, median=0.7, min=0.7, max=0.7)
    summary = SurfacingMetrics.model_validate(
        {"relevant": stats, "distractor": stats.model_copy(update={"mean": 0.5})}
        | {"overlap_max_distractor": 0, "overlap_min_relevant": 0, "separable": True}
        | {"block_precision_mean": 0, "block_precision_n": 0}
        | {"distractor_query_count": 0, "distractor_queries_with_surfaced": 0}
    )

    s = _read(report, summary=summary)
    bare = _read(report)

    assert s.fact_population == population
    assert s.score_means is not None
    assert (s.score_means.relevant, s.score_means.distractor) == (0.7, 0.5)
    assert bare.score_means is None
