"""S-margin, block precision, threshold sweep, distractor accounting."""

from __future__ import annotations

import pytest
from probe_cli.surfacing_metrics import compute_surfacing_metrics
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
