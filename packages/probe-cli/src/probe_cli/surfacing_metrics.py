"""Read-path surfacing metrics: relevant/distractor S-margin, block precision, sweep.

block_precision_mean is the per-query precision of the IMPORTANT CONTEXT block;
the threshold-sweep precision is a distinct pooled number (the recall-floor
calibration). They are not the same "precision" -- do not conflate.
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, Field

from probe_cli.surfacing_runner import SurfacingRunReport

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
