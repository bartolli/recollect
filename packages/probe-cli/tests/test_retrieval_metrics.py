"""Retrieval metric tests."""

from __future__ import annotations

import pytest
from probe_cli.metrics import (
    QueryResult,
    RetrievalRunReport,
    aggregate_retrieval_runs,
    compute_retrieval_metrics,
    intrusion_rate,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_full_hit() -> None:
    assert recall_at_k(["a", "b", "c", "d", "e"], {"a", "b"}, 5) == 1.0


def test_recall_at_k_partial() -> None:
    assert recall_at_k(["a", "x", "y", "z", "w"], {"a", "b"}, 5) == 0.5


def test_recall_at_k_miss() -> None:
    assert recall_at_k(["x", "y", "z", "w", "v"], {"a"}, 5) == 0.0


def test_recall_at_k_top_k_truncates() -> None:
    # "b" at index 5 ⇒ outside top-5
    assert recall_at_k(["x", "y", "z", "w", "v", "b"], {"b"}, 5) == 0.0


def test_recall_at_k_empty_relevant_raises() -> None:
    with pytest.raises(ValueError, match="empty relevant set"):
        recall_at_k(["a", "b"], set(), 5)


def test_mrr_first_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_third_position() -> None:
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3)


def test_mrr_no_hit() -> None:
    assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


def test_intrusion_rate_clean() -> None:
    assert intrusion_rate(["a", "b", "c"], {"a", "b", "c"}, 3) == 0.0


def test_intrusion_rate_full() -> None:
    assert intrusion_rate(["x", "y", "z"], {"a"}, 3) == 1.0


def test_intrusion_rate_distractor_query() -> None:
    assert intrusion_rate(["x", "y", "z", "w", "v"], set(), 5) == 1.0


def test_intrusion_rate_short_result_uses_returned_length() -> None:
    # |returned|=2 < k=5 ⇒ denom=2, not 5
    assert intrusion_rate(["x", "y"], {"a"}, 5) == 1.0


def test_intrusion_rate_empty_result() -> None:
    assert intrusion_rate([], {"a"}, 5) == 0.0


def _query(
    qid: str,
    ranked: list[str],
    relevant: list[str],
    *,
    category: str | None = None,
    is_distractor: bool = False,
    persona_fact_count: int = 0,
) -> QueryResult:
    return QueryResult(
        query_id=qid,
        success=True,
        ranked_corpus_ids=ranked,
        relevant_corpus_ids=relevant,
        expected_category=category,  # type: ignore[arg-type]
        is_distractor=is_distractor,
        persona_fact_count=persona_fact_count,
    )


def _report(queries: list[QueryResult], *, run_index: int = 0) -> RetrievalRunReport:
    return RetrievalRunReport(
        arm_name="test",
        run_index=run_index,
        model="test",
        prompt_version="0.0.0",
        top_k=5,
        ingested=10,
        queries=queries,
    )


def test_compute_retrieval_metrics_mixed_set() -> None:
    # q1 r=1 mrr=1 | q2 r=1 mrr=0.5 | q3 r=0 mrr=0 | q4 distractor 1.0
    queries = [
        _query("q1", ["a", "b", "x", "y", "z"], ["a"], category="health"),
        _query("q2", ["x", "a", "y", "z", "w"], ["a"], category="health"),
        _query("q3", ["x", "y", "z", "w", "v"], ["a"], category="dietary"),
        _query("q4", ["x", "y", "z", "w", "v"], [], is_distractor=True),
    ]
    metrics = compute_retrieval_metrics(_report(queries))
    assert metrics.n_scored == 3
    assert metrics.n_distractors == 1
    assert metrics.n_failed == 0
    assert metrics.recall_at_k_mean == pytest.approx(2 / 3)
    assert metrics.mrr_mean == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert metrics.intrusion_rate_mean == pytest.approx((4 / 5 + 4 / 5 + 5 / 5) / 3)
    assert metrics.distractor_intrusion_mean == 1.0
    assert "health" in metrics.per_category
    assert metrics.per_category["health"]["recall_at_k_mean"] == 1.0
    assert metrics.per_category["dietary"]["recall_at_k_mean"] == 0.0


def test_compute_retrieval_metrics_failed_query() -> None:
    failed = QueryResult(
        query_id="qf",
        success=False,
        error="boom",
        relevant_corpus_ids=["a"],
    )
    ok = _query("q1", ["a", "b"], ["a"], category="health")
    metrics = compute_retrieval_metrics(_report([failed, ok]))
    assert metrics.n_failed == 1
    assert metrics.n_scored == 1
    assert metrics.recall_at_k_mean == 1.0


def test_compute_retrieval_metrics_empty_relevant_treated_as_distractor() -> None:
    # is_distractor=False but relevant=[] ⇒ still routed to distractor bucket
    q = _query("q1", ["x", "y"], [])
    metrics = compute_retrieval_metrics(_report([q]))
    assert metrics.n_scored == 0
    assert metrics.n_distractors == 1


def test_aggregate_retrieval_runs_mean_and_std() -> None:
    r1 = _report(
        [_query("q1", ["a", "b", "c", "d", "e"], ["a"], category="health")],
        run_index=0,
    )
    r2 = _report(
        [_query("q1", ["x", "y", "a", "b", "c"], ["a"], category="health")],
        run_index=1,
    )
    agg = aggregate_retrieval_runs("test", "m", "0.0.0", 5, [r1, r2])
    assert agg.runs == 2
    assert agg.queries_per_run == 1
    assert agg.recall_at_k_mean == 1.0
    assert agg.mrr_mean == pytest.approx((1.0 + 1 / 3) / 2)
    assert agg.mrr_std > 0
    assert "health" in agg.per_category


def test_aggregate_requires_at_least_one_report() -> None:
    with pytest.raises(ValueError, match="at least one report"):
        aggregate_retrieval_runs("test", "m", "0.0.0", 5, [])


def test_persona_fact_count_aggregated() -> None:
    queries = [
        _query("q1", ["a", "b"], ["a"], category="health", persona_fact_count=2),
        _query("q2", ["c", "d"], ["c"], category="dietary", persona_fact_count=0),
        _query("q3", ["e", "f"], ["e"], category="health", persona_fact_count=1),
    ]
    metrics = compute_retrieval_metrics(_report(queries))
    assert metrics.persona_facts_per_query_mean == pytest.approx(1.0)
    # recall and intrusion not affected — persona facts already filtered upstream
    assert metrics.recall_at_k_mean == 1.0


def test_persona_fact_count_per_run_in_aggregate() -> None:
    r1 = _report(
        [_query("q1", ["a"], ["a"], category="health", persona_fact_count=2)],
        run_index=0,
    )
    r2 = _report(
        [_query("q1", ["a"], ["a"], category="health", persona_fact_count=4)],
        run_index=1,
    )
    agg = aggregate_retrieval_runs("test", "m", "0.0.0", 5, [r1, r2])
    assert agg.persona_facts_per_query_mean == pytest.approx(3.0)
    assert agg.persona_facts_per_query_std > 0
