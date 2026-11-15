"""Read-path surfacing metrics: relevant/distractor S-margin, block precision, sweep.

block_precision_mean is the per-query precision of the IMPORTANT CONTEXT block;
the threshold-sweep precision is a distinct pooled number (the recall-floor
calibration). They are not the same "precision" -- do not conflate.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from probe_cli.surfacing_runner import SurfacingRunReport

if TYPE_CHECKING:
    from collections.abc import Sequence

    from probe_cli.corpus import QueryEntry
    from probe_cli.surfacing_runner import SurfacedFact

_DEFAULT_THRESHOLDS: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65)


class ScoreStats(BaseModel):
    n: int
    mean: float
    median: float
    min: float
    max: float


class ThresholdPoint(BaseModel):
    threshold: float
    kept_relevant: int
    kept_distractor: int
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)


class QueryCoverage(BaseModel):
    query_id: str
    required: list[str]
    returned: list[str]
    covered: list[str]


class RequiredCoverage(BaseModel):
    # The denominator is the query corpus's labels, not the surfaced facts: a
    # query that loses every required source still counts its pairs. A pair is
    # covered when a fact from that source passes the production recall gate.
    floor: float
    required_pairs: int = Field(ge=0)
    returned_pairs: int = Field(ge=0)
    covered_pairs: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    # Every fact the gate passes, distractor queries included.
    gated_facts: int = Field(ge=0)
    gated_relevant: int = Field(ge=0)
    gated_precision: float = Field(ge=0.0, le=1.0)
    queries: list[QueryCoverage]
    queries_without_required_source: list[str]


class SurfacingMetrics(BaseModel):
    relevant: ScoreStats
    distractor: ScoreStats
    overlap_max_distractor: float
    overlap_min_relevant: float
    separable: bool
    block_precision_mean: float = Field(ge=0.0, le=1.0)
    block_precision_n: int
    distractor_query_count: int
    distractor_queries_with_surfaced: int
    threshold_sweep: list[ThresholdPoint] = Field(default_factory=list)
    required_coverage: RequiredCoverage | None = None


def _score_stats(values: list[float]) -> ScoreStats:
    if not values:
        return ScoreStats(n=0, mean=0.0, median=0.0, min=0.0, max=0.0)
    return ScoreStats(
        n=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        min=min(values),
        max=max(values),
    )


def compute_surfacing_metrics(
    report: SurfacingRunReport,
    *,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> SurfacingMetrics:
    if not report.queries:
        raise ValueError("report must contain queries")

    rel = [f.score for q in report.queries for f in q.surfaced if f.relevant]
    dis = [f.score for q in report.queries for f in q.surfaced if not f.relevant]

    # Block precision over non-distractor queries that surfaced >=1 fact. A query
    # that surfaced nothing is 0/0 (excluded); distractor queries are reported
    # separately rather than diluting the block number.
    block_precisions: list[float] = []
    for q in report.queries:
        if q.is_distractor or not q.surfaced:
            continue
        rel_count = sum(1 for f in q.surfaced if f.relevant)
        block_precisions.append(rel_count / len(q.surfaced))

    distractor_queries = [q for q in report.queries if q.is_distractor]
    distractor_with_surfaced = sum(1 for q in distractor_queries if q.surfaced)

    total_rel = len(rel)
    sweep: list[ThresholdPoint] = []
    for t in thresholds:
        kr = sum(1 for s in rel if s >= t)
        kd = sum(1 for s in dis if s >= t)
        sweep.append(
            ThresholdPoint(
                threshold=t,
                kept_relevant=kr,
                kept_distractor=kd,
                precision=kr / (kr + kd) if (kr + kd) else 0.0,
                recall=kr / total_rel if total_rel else 0.0,
            )
        )

    max_dis = max(dis) if dis else 0.0
    min_rel = min(rel) if rel else 0.0
    return SurfacingMetrics(
        relevant=_score_stats(rel),
        distractor=_score_stats(dis),
        overlap_max_distractor=max_dis,
        overlap_min_relevant=min_rel,
        separable=(not rel or not dis) or (max_dis < min_rel),
        block_precision_mean=(
            statistics.mean(block_precisions) if block_precisions else 0.0
        ),
        block_precision_n=len(block_precisions),
        distractor_query_count=len(distractor_queries),
        distractor_queries_with_surfaced=distractor_with_surfaced,
        threshold_sweep=sweep,
    )


def _query_coverage(
    query: QueryEntry, surfaced: Sequence[SurfacedFact]
) -> QueryCoverage:
    required = list(query.relevant_trace_ids)
    return QueryCoverage(
        query_id=query.id,
        required=required,
        returned=[r for r in required if any(f.source_eval_id == r for f in surfaced)],
        covered=[
            r
            for r in required
            if any(f.source_eval_id == r and f.passes_gate for f in surfaced)
        ],
    )


def compute_required_coverage(
    report: SurfacingRunReport, queries: Sequence[QueryEntry], *, floor: float
) -> RequiredCoverage:
    """One (query, required id) pair per label; distractor queries add none.

    floor is the configured recall floor, recorded beside the gate's verdicts.
    """
    surfaced = {q.query_id: q.surfaced for q in report.queries}
    rows = [
        _query_coverage(q, surfaced.get(q.id, []))
        for q in queries
        if q.relevant_trace_ids
    ]
    required = sum(len(r.required) for r in rows)
    covered = sum(len(r.covered) for r in rows)
    gated = [f for q in report.queries for f in q.surfaced if f.passes_gate]
    gated_relevant = sum(1 for f in gated if f.relevant)
    return RequiredCoverage(
        floor=floor,
        required_pairs=required,
        returned_pairs=sum(len(r.returned) for r in rows),
        covered_pairs=covered,
        coverage=covered / required if required else 0.0,
        gated_facts=len(gated),
        gated_relevant=gated_relevant,
        gated_precision=gated_relevant / len(gated) if gated else 0.0,
        queries=rows,
        queries_without_required_source=[r.query_id for r in rows if not r.covered],
    )
