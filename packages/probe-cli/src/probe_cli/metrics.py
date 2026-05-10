"""Extraction + retrieval metrics."""

from __future__ import annotations

import math
from collections import Counter
from typing import get_args

from pydantic import BaseModel, Field
from recollect.llm.types import ExtractionResult, Predicate
from recollect.models import FactCategory

ESCAPE_VALVE_PREDICATE = "is_associated_with"
PREDICATE_VOCAB_SIZE = len(get_args(Predicate))
FOREIGN_PREFIX = "__foreign__:"


class EntryResult(BaseModel):
    entry_id: str
    success: bool
    error: str = ""
    extraction: ExtractionResult | None = None
    latency_ms: float = 0.0


class RunReport(BaseModel):
    arm_name: str
    run_index: int
    model: str
    prompt_version: str
    entries: list[EntryResult] = Field(default_factory=list)


class RunMetrics(BaseModel):
    validity_rate: float
    predicate_diversity: float
    escape_valve_rate: float
    concepts_per_trace_mean: float
    valence_mean: float
    significance_mean: float
    predicate_distribution: dict[str, int] = Field(default_factory=dict)
    category_distribution: dict[str, int] = Field(default_factory=dict)
    entity_type_distribution: dict[str, int] = Field(default_factory=dict)


class AggregateMetrics(BaseModel):
    arm_name: str
    runs: int
    total_entries_per_run: int
    prompt_version: str
    model: str
    validity_rate_mean: float
    validity_rate_std: float
    predicate_diversity_mean: float
    predicate_diversity_std: float
    escape_valve_rate_mean: float
    escape_valve_rate_std: float
    concepts_per_trace_mean: float
    concepts_per_trace_std: float
    valence_mean: float
    significance_mean: float
    predicate_distribution: dict[str, int] = Field(default_factory=dict)
    category_distribution: dict[str, int] = Field(default_factory=dict)
    entity_type_distribution: dict[str, int] = Field(default_factory=dict)


def compute_run_metrics(report: RunReport) -> RunMetrics:
    successes = [e for e in report.entries if e.success and e.extraction is not None]
    n = len(report.entries)
    validity_rate = len(successes) / n if n else 0.0

    predicate_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    entity_type_counter: Counter[str] = Counter()
    concepts_counts: list[int] = []
    valences: list[float] = []
    significances: list[float] = []

    for e in successes:
        ext = e.extraction
        if ext is None:
            continue
        for rel in ext.relations:
            predicate_counter[rel.relation] += 1
            category_counter[rel.category] += 1
        for ent in ext.entities:
            entity_type_counter[ent.entity_type] += 1
        concepts_counts.append(len(ext.concepts))
        valences.append(ext.emotional_valence)
        significances.append(ext.significance)

    total_predicates = sum(predicate_counter.values())
    diversity = (
        len(predicate_counter) / PREDICATE_VOCAB_SIZE if PREDICATE_VOCAB_SIZE else 0.0
    )
    escape = (
        predicate_counter.get(ESCAPE_VALVE_PREDICATE, 0) / total_predicates
        if total_predicates
        else 0.0
    )

    return RunMetrics(
        validity_rate=validity_rate,
        predicate_diversity=diversity,
        escape_valve_rate=escape,
        concepts_per_trace_mean=_mean(concepts_counts),
        valence_mean=_mean(valences),
        significance_mean=_mean(significances),
        predicate_distribution=dict(predicate_counter),
        category_distribution=dict(category_counter),
        entity_type_distribution=dict(entity_type_counter),
    )


def aggregate_runs(
    arm_name: str,
    model: str,
    prompt_version: str,
    reports: list[RunReport],
) -> AggregateMetrics:
    if not reports:
        raise ValueError("aggregate_runs requires at least one report")
    per_run = [compute_run_metrics(r) for r in reports]

    pred_dist: Counter[str] = Counter()
    cat_dist: Counter[str] = Counter()
    ent_dist: Counter[str] = Counter()
    for m in per_run:
        pred_dist.update(m.predicate_distribution)
        cat_dist.update(m.category_distribution)
        ent_dist.update(m.entity_type_distribution)

    return AggregateMetrics(
        arm_name=arm_name,
        runs=len(reports),
        total_entries_per_run=len(reports[0].entries),
        prompt_version=prompt_version,
        model=model,
        validity_rate_mean=_mean([m.validity_rate for m in per_run]),
        validity_rate_std=_std([m.validity_rate for m in per_run]),
        predicate_diversity_mean=_mean([m.predicate_diversity for m in per_run]),
        predicate_diversity_std=_std([m.predicate_diversity for m in per_run]),
        escape_valve_rate_mean=_mean([m.escape_valve_rate for m in per_run]),
        escape_valve_rate_std=_std([m.escape_valve_rate for m in per_run]),
        concepts_per_trace_mean=_mean([m.concepts_per_trace_mean for m in per_run]),
        concepts_per_trace_std=_std([m.concepts_per_trace_mean for m in per_run]),
        valence_mean=_mean([m.valence_mean for m in per_run]),
        significance_mean=_mean([m.significance_mean for m in per_run]),
        predicate_distribution=dict(pred_dist),
        category_distribution=dict(cat_dist),
        entity_type_distribution=dict(ent_dist),
    )


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


# -- Retrieval --


class QueryResult(BaseModel):
    # ranked_corpus_ids: CorpusEntry.id in retrieved order, persona-fact Thoughts
    # filtered out (their trace.id is a synthetic UUID with no link to source trace).
    # Cross-session leaks become "{FOREIGN_PREFIX}{uuid}" so intrusion counts.
    query_id: str
    success: bool
    error: str = ""
    ranked_corpus_ids: list[str] = Field(default_factory=list)
    relevant_corpus_ids: list[str] = Field(default_factory=list)
    expected_category: FactCategory | None = None
    is_distractor: bool = False
    persona_fact_count: int = 0
    latency_ms: float = 0.0


class RetrievalRunReport(BaseModel):
    arm_name: str
    run_index: int
    model: str
    prompt_version: str
    top_k: int
    ingested: int = 0
    ingest_failures: int = 0
    queries: list[QueryResult] = Field(default_factory=list)


class RetrievalMetrics(BaseModel):
    recall_at_k_mean: float
    mrr_mean: float
    intrusion_rate_mean: float
    distractor_intrusion_mean: float
    persona_facts_per_query_mean: float
    n_scored: int
    n_distractors: int
    n_failed: int
    per_category: dict[str, dict[str, float]] = Field(default_factory=dict)


class RetrievalAggregate(BaseModel):
    arm_name: str
    runs: int
    queries_per_run: int
    top_k: int
    prompt_version: str
    model: str
    recall_at_k_mean: float
    recall_at_k_std: float
    mrr_mean: float
    mrr_std: float
    intrusion_rate_mean: float
    intrusion_rate_std: float
    distractor_intrusion_mean: float
    distractor_intrusion_std: float
    persona_facts_per_query_mean: float
    persona_facts_per_query_std: float
    per_category: dict[str, dict[str, float]] = Field(default_factory=dict)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    # |relevant ∩ top_k| / |relevant|; empty relevant ⇒ undefined, caller filters.
    if not relevant:
        raise ValueError("recall_at_k undefined for empty relevant set")
    if k <= 0:
        raise ValueError("k must be positive")
    top = set(ranked[:k])
    hits = len(relevant & top)
    return hits / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    # 1/rank of first hit, 0.0 if no hit; empty relevant ⇒ undefined, caller filters.
    if not relevant:
        raise ValueError("reciprocal_rank undefined for empty relevant set")
    for idx, rid in enumerate(ranked):
        if rid in relevant:
            return 1.0 / (idx + 1)
    return 0.0


def intrusion_rate(ranked: list[str], relevant: set[str], k: int) -> float:
    # 1 - precision@k. Denominator min(k, |returned|) — short results not penalized
    # for empty slots; distractor queries (empty relevant) ⇒ every returned id intrudes.
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked[:k]
    if not top:
        return 0.0
    hits = sum(1 for rid in top if rid in relevant)
    intruders = len(top) - hits
    return intruders / len(top)


def compute_retrieval_metrics(report: RetrievalRunReport) -> RetrievalMetrics:
    scored: list[tuple[QueryResult, float, float, float]] = []
    distractor_intrusions: list[float] = []
    failed = 0

    for q in report.queries:
        if not q.success:
            failed += 1
            continue
        relevant = set(q.relevant_corpus_ids)
        if q.is_distractor or not relevant:
            distractor_intrusions.append(
                intrusion_rate(q.ranked_corpus_ids, relevant, report.top_k)
            )
            continue
        r = recall_at_k(q.ranked_corpus_ids, relevant, report.top_k)
        rr = reciprocal_rank(q.ranked_corpus_ids, relevant)
        ir = intrusion_rate(q.ranked_corpus_ids, relevant, report.top_k)
        scored.append((q, r, rr, ir))

    per_category = _per_category_breakdown(scored)
    pf_counts = [
        float(q.persona_fact_count) for q in report.queries if q.success
    ]

    return RetrievalMetrics(
        recall_at_k_mean=_mean([r for _, r, _, _ in scored]),
        mrr_mean=_mean([rr for _, _, rr, _ in scored]),
        intrusion_rate_mean=_mean([ir for _, _, _, ir in scored]),
        distractor_intrusion_mean=_mean(distractor_intrusions),
        persona_facts_per_query_mean=_mean(pf_counts),
        n_scored=len(scored),
        n_distractors=len(distractor_intrusions),
        n_failed=failed,
        per_category=per_category,
    )


def aggregate_retrieval_runs(
    arm_name: str,
    model: str,
    prompt_version: str,
    top_k: int,
    reports: list[RetrievalRunReport],
) -> RetrievalAggregate:
    if not reports:
        raise ValueError("aggregate_retrieval_runs requires at least one report")
    per_run = [compute_retrieval_metrics(r) for r in reports]

    cat_keys: set[str] = set()
    for m in per_run:
        cat_keys.update(m.per_category.keys())
    per_category: dict[str, dict[str, float]] = {}
    for cat in sorted(cat_keys):
        recalls = [
            m.per_category[cat]["recall_at_k_mean"]
            for m in per_run
            if cat in m.per_category
        ]
        mrrs = [
            m.per_category[cat]["mrr_mean"]
            for m in per_run
            if cat in m.per_category
        ]
        per_category[cat] = {
            "recall_at_k_mean": _mean(recalls),
            "recall_at_k_std": _std(recalls),
            "mrr_mean": _mean(mrrs),
            "mrr_std": _std(mrrs),
        }

    return RetrievalAggregate(
        arm_name=arm_name,
        runs=len(reports),
        queries_per_run=len(reports[0].queries),
        top_k=top_k,
        prompt_version=prompt_version,
        model=model,
        recall_at_k_mean=_mean([m.recall_at_k_mean for m in per_run]),
        recall_at_k_std=_std([m.recall_at_k_mean for m in per_run]),
        mrr_mean=_mean([m.mrr_mean for m in per_run]),
        mrr_std=_std([m.mrr_mean for m in per_run]),
        intrusion_rate_mean=_mean([m.intrusion_rate_mean for m in per_run]),
        intrusion_rate_std=_std([m.intrusion_rate_mean for m in per_run]),
        distractor_intrusion_mean=_mean([m.distractor_intrusion_mean for m in per_run]),
        distractor_intrusion_std=_std([m.distractor_intrusion_mean for m in per_run]),
        persona_facts_per_query_mean=_mean(
            [m.persona_facts_per_query_mean for m in per_run]
        ),
        persona_facts_per_query_std=_std(
            [m.persona_facts_per_query_mean for m in per_run]
        ),
        per_category=per_category,
    )


def _per_category_breakdown(
    scored: list[tuple[QueryResult, float, float, float]],
) -> dict[str, dict[str, float]]:
    by_cat: dict[str, list[tuple[float, float]]] = {}
    for q, r, rr, _ir in scored:
        if q.expected_category is None:
            continue
        by_cat.setdefault(q.expected_category, []).append((r, rr))
    return {
        cat: {
            "recall_at_k_mean": _mean([r for r, _ in pairs]),
            "mrr_mean": _mean([rr for _, rr in pairs]),
            "n": float(len(pairs)),
        }
        for cat, pairs in by_cat.items()
    }
