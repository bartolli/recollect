"""Metrics computation tests with synthetic ExtractionResult fixtures."""

from __future__ import annotations

import pytest
from probe_cli.metrics import (
    PREDICATE_VOCAB_SIZE,
    EntryResult,
    RunReport,
    aggregate_runs,
    compute_run_metrics,
)
from recollect.llm.types import (
    Entity,
    EntityType,
    ExtractionResult,
    Predicate,
    Relation,
)


def _result(
    predicates: list[Predicate],
    *,
    concepts: list[str] | None = None,
    valence: float = 0.0,
    significance: float = 0.5,
    entity_types: list[EntityType] | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        concepts=concepts or [],
        relations=[
            Relation(source="x", relation=p, target="y", category="general")
            for p in predicates
        ],
        entities=[
            Entity(name=f"e{i}", entity_type=t)
            for i, t in enumerate(entity_types or [])
        ],
        emotional_valence=valence,
        significance=significance,
    )


def _entry(
    eid: str, ext: ExtractionResult | None, *, success: bool = True
) -> EntryResult:
    return EntryResult(entry_id=eid, success=success, extraction=ext, latency_ms=1.0)


def _report(entries: list[EntryResult], run_index: int = 0) -> RunReport:
    return RunReport(
        arm_name="t",
        run_index=run_index,
        model="mock",
        prompt_version="0.0.0-test",
        entries=entries,
    )


class TestRunMetrics:
    def test_validity_rate_all_success(self) -> None:
        r = _report(
            [
                _entry("a", _result(["works_at"])),
                _entry("b", _result(["studies"])),
            ]
        )
        m = compute_run_metrics(r)
        assert m.validity_rate == 1.0

    def test_validity_rate_partial_failure(self) -> None:
        r = _report(
            [
                _entry("a", _result(["works_at"])),
                _entry("b", None, success=False),
            ]
        )
        m = compute_run_metrics(r)
        assert m.validity_rate == 0.5

    def test_predicate_diversity_normalized(self) -> None:
        preds: list[Predicate] = ["works_at", "studies", "prefers"]
        r = _report([_entry(f"e{i}", _result([p])) for i, p in enumerate(preds)])
        m = compute_run_metrics(r)
        assert m.predicate_diversity == pytest.approx(3 / PREDICATE_VOCAB_SIZE)

    def test_escape_valve_rate(self) -> None:
        r = _report(
            [
                _entry("a", _result(["is_associated_with"])),
                _entry("b", _result(["is_associated_with"])),
                _entry("c", _result(["works_at"])),
            ]
        )
        m = compute_run_metrics(r)
        assert m.escape_valve_rate == pytest.approx(2 / 3)

    def test_concepts_distribution(self) -> None:
        r = _report(
            [
                _entry("a", _result(["works_at"], concepts=["c1", "c2"])),
                _entry("b", _result(["works_at"], concepts=["c1", "c2", "c3", "c4"])),
            ]
        )
        m = compute_run_metrics(r)
        assert m.concepts_per_trace_mean == 3.0

    def test_distributions_recorded(self) -> None:
        r = _report(
            [
                _entry(
                    "a",
                    _result(["works_at"], entity_types=["person", "organization"]),
                ),
            ]
        )
        m = compute_run_metrics(r)
        assert m.predicate_distribution == {"works_at": 1}
        assert m.entity_type_distribution == {"person": 1, "organization": 1}
        assert m.category_distribution == {"general": 1}


class TestAggregate:
    def test_mean_and_std_across_runs(self) -> None:
        r1 = _report([_entry("a", _result(["works_at"]))], run_index=0)
        r2 = _report([_entry("a", _result(["works_at"]))], run_index=1)
        r3 = _report([_entry("a", None, success=False)], run_index=2)
        agg = aggregate_runs("t", "mock", "1.0.0", [r1, r2, r3])
        assert agg.runs == 3
        assert agg.validity_rate_mean == pytest.approx(2 / 3)
        assert agg.validity_rate_std > 0.0

    def test_predicate_distribution_aggregates(self) -> None:
        r1 = _report([_entry("a", _result(["works_at"]))])
        r2 = _report([_entry("a", _result(["works_at", "studies"]))])
        agg = aggregate_runs("t", "mock", "1.0.0", [r1, r2])
        assert agg.predicate_distribution["works_at"] == 2
        assert agg.predicate_distribution["studies"] == 1

    def test_empty_reports_raises(self) -> None:
        with pytest.raises(ValueError):
            aggregate_runs("t", "mock", "1.0.0", [])

    def test_single_run_zero_std(self) -> None:
        r1 = _report([_entry("a", _result(["works_at"]))])
        agg = aggregate_runs("t", "mock", "1.0.0", [r1])
        assert agg.validity_rate_std == 0.0
