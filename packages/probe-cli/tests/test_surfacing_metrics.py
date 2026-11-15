"""S-margin, block precision, threshold sweep, distractor accounting."""

from __future__ import annotations

import pytest
from probe_cli.corpus import QueryEntry
from probe_cli.surfacing_metrics import (
    compute_required_coverage,
    compute_surfacing_metrics,
)
from probe_cli.surfacing_runner import (
    QuerySurfacing,
    SurfacedFact,
    SurfacingRunReport,
)


def _f(score: float, relevant: bool) -> SurfacedFact:
    return SurfacedFact(
        fact_id="f", source_trace_id="t", score=score, relevant=relevant
    )


def _q(
    qid: str, surfaced: list[SurfacedFact], *, distractor: bool = False
) -> QuerySurfacing:
    return QuerySurfacing(query_id=qid, is_distractor=distractor, surfaced=surfaced)


def _report(queries: list[QuerySurfacing]) -> SurfacingRunReport:
    return SurfacingRunReport(
        arm_name="t", model="m", seeded_traces=1, promoted_facts=1, queries=queries
    )


def test_relevant_and_distractor_distributions() -> None:
    m = compute_surfacing_metrics(
        _report([
            _q("q1", [_f(0.8, True), _f(0.4, False)]),
            _q("q2", [_f(0.6, True), _f(0.5, False)]),
        ])
    )
    assert m.relevant.n == 2
    assert m.relevant.mean == pytest.approx(0.7)
    assert m.relevant.min == pytest.approx(0.6)
    assert m.relevant.max == pytest.approx(0.8)
    assert m.distractor.n == 2
    assert m.distractor.mean == pytest.approx(0.45)


def test_overlap_separable() -> None:
    m = compute_surfacing_metrics(_report([_q("q1", [_f(0.8, True), _f(0.4, False)])]))
    assert m.overlap_max_distractor == pytest.approx(0.4)
    assert m.overlap_min_relevant == pytest.approx(0.8)
    assert m.separable is True


def test_overlap_not_separable() -> None:
    # distractor 0.6 >= relevant 0.5 -> pools overlap
    m = compute_surfacing_metrics(_report([_q("q1", [_f(0.5, True), _f(0.6, False)])]))
    assert m.separable is False


def test_block_precision_excludes_distractor_and_empty() -> None:
    m = compute_surfacing_metrics(
        _report([
            _q("q1", [_f(0.8, True), _f(0.4, False)]),   # 1/2
            _q("q2", [_f(0.6, True)]),                   # 1/1
            _q("q3", []),                                # 0/0 -> excluded
            _q("d1", [_f(0.5, False)], distractor=True), # distractor -> excluded
        ])
    )
    assert m.block_precision_n == 2
    assert m.block_precision_mean == pytest.approx(0.75)


def test_threshold_sweep() -> None:
    m = compute_surfacing_metrics(
        _report([_q("q1", [_f(0.8, True), _f(0.6, True), _f(0.6, False)])]),
        thresholds=(0.55, 0.7),
    )
    pts = {p.threshold: p for p in m.threshold_sweep}
    # T=0.55: rel {0.8,0.6}=2, dis {0.6}=1 -> precision 2/3, recall 2/2
    assert pts[0.55].kept_relevant == 2
    assert pts[0.55].kept_distractor == 1
    assert pts[0.55].precision == pytest.approx(2 / 3)
    assert pts[0.55].recall == pytest.approx(1.0)
    # T=0.7: rel {0.8}=1, dis {}=0 -> precision 1.0, recall 1/2
    assert pts[0.7].kept_relevant == 1
    assert pts[0.7].kept_distractor == 0
    assert pts[0.7].recall == pytest.approx(0.5)


def test_distractor_query_accounting() -> None:
    m = compute_surfacing_metrics(
        _report([
            _q("q1", [_f(0.8, True)]),
            _q("d1", [_f(0.5, False), _f(0.4, False)], distractor=True),
            _q("d2", [], distractor=True),
        ])
    )
    assert m.distractor_query_count == 2
    assert m.distractor_queries_with_surfaced == 1
    # distractor-query facts pool into the distractor scores
    assert m.distractor.n == 2


def test_empty_report_rejected() -> None:
    with pytest.raises(ValueError, match="queries"):
        compute_surfacing_metrics(_report([]))


_FLOOR = 0.65
_LABELED = [
    QueryEntry(id="q1", text="x", relevant_trace_ids=["e1"]),
    QueryEntry(id="q2", text="y", relevant_trace_ids=["e2"]),
]


def _src(eval_id: str, score: float, *, gate: bool | None = None) -> SurfacedFact:
    passes = score >= _FLOOR if gate is None else gate
    return SurfacedFact(
        fact_id=eval_id,
        source_eval_id=eval_id,
        score=score,
        relevant=True,
        passes_gate=passes,
    )


def test_lost_required_source_lowers_coverage_not_sweep_recall() -> None:
    whole = _report([_q("q1", [_src("e1", 0.8)]), _q("q2", [_src("e2", 0.7)])])
    lost = _report([_q("q1", [_src("e1", 0.8)]), _q("q2", [])])
    for report, coverage in ((whole, 1.0), (lost, 0.5)):
        metrics = compute_surfacing_metrics(report, thresholds=(_FLOOR,))
        (point,) = metrics.threshold_sweep
        assert point.recall == pytest.approx(1.0)
        c = compute_required_coverage(report, _LABELED, floor=_FLOOR)
        assert (c.required_pairs, c.coverage) == (2, pytest.approx(coverage))


def test_source_the_gate_drops_is_returned_not_covered() -> None:
    report = _report([_q("q1", [_src("e1", 0.8)]), _q("q2", [_src("e2", 0.5)])])
    c = compute_required_coverage(report, _LABELED, floor=_FLOOR)
    q2 = c.queries[1]
    assert (q2.query_id, q2.returned, q2.covered) == ("q2", ["e2"], [])
    assert (c.returned_pairs, c.covered_pairs) == (2, 1)


def test_coverage_follows_the_gate_not_the_bare_floor() -> None:
    # A safety-bypass fact passes the gate under the floor.
    bypass = _src("e2", 0.5, gate=True)
    noise = SurfacedFact(fact_id="n", source_eval_id="e9", score=0.9, passes_gate=True)
    report = _report([_q("q1", [_src("e1", 0.8), noise]), _q("q2", [bypass])])
    c = compute_required_coverage(report, _LABELED, floor=_FLOOR)
    assert (c.covered_pairs, c.coverage) == (2, pytest.approx(1.0))
    assert (c.gated_facts, c.gated_relevant) == (3, 2)
    assert c.gated_precision == pytest.approx(2 / 3)


def test_labeled_query_missing_from_report_is_uncovered() -> None:
    report = _report([_q("q1", [_src("e1", 0.8)])])
    c = compute_required_coverage(report, _LABELED, floor=_FLOOR)
    assert (c.required_pairs, c.covered_pairs) == (2, 1)
    assert c.queries_without_required_source == ["q2"]


def test_distractor_queries_contribute_no_pair() -> None:
    queries = [*_LABELED, QueryEntry(id="d1", text="z", relevant_trace_ids=[])]
    labeled = [_q("q1", [_src("e1", 0.8)]), _q("q2", [_src("e2", 0.7)])]
    report = _report([*labeled, _q("d1", [_src("e9", 0.9)], distractor=True)])
    c = compute_required_coverage(report, queries, floor=_FLOOR)
    assert [q.query_id for q in c.queries] == ["q1", "q2"]
    assert (c.required_pairs, c.coverage) == (2, pytest.approx(1.0))
