"""Action-accuracy + extension-target accuracy + aggregate tests."""

from __future__ import annotations

import pytest
from probe_cli.corpus import ExpectedAction
from probe_cli.situational_metrics import (
    aggregate_situational_runs,
    compute_situational_metrics,
)
from probe_cli.situational_runner import EvalResult, SituationalRunReport


def _r(
    expected: ExpectedAction,
    actual: ExpectedAction | None,
    *,
    expected_gid: str | None = None,
    actual_gid: str | None = None,
    success: bool = True,
) -> EvalResult:
    return EvalResult(
        entry_id="x",
        expected_action=expected,
        expected_group_id=expected_gid,
        actual_action=actual,
        actual_group_id=actual_gid,
        success=success,
    )


class TestComputeSituationalMetrics:
    def test_perfect_run(self) -> None:
        results = [
            _r("extend", "extend", expected_gid="G1", actual_gid="G1"),
            _r("revise", "revise", expected_gid="G2", actual_gid="G2"),
            _r("create", "create"),
            _r("none", "none"),
        ]
        m = compute_situational_metrics(results)
        assert m.action_accuracy == 1.0
        assert m.extension_target_accuracy == 1.0
        assert m.success_count == 4

    def test_action_accuracy_partial(self) -> None:
        results = [
            _r("extend", "extend", expected_gid="G1", actual_gid="G1"),
            _r("extend", "none", expected_gid="G2"),
            _r("none", "create"),
            _r("none", "none"),
        ]
        m = compute_situational_metrics(results)
        assert m.action_accuracy == pytest.approx(0.5)
        # extension_target denominator = correct-action extends only (1)
        assert m.extension_target_accuracy == 1.0
        assert m.extension_eligible == 2

    def test_extension_target_partial(self) -> None:
        results = [
            _r("extend", "extend", expected_gid="G1", actual_gid="G1"),
            _r("extend", "extend", expected_gid="G2", actual_gid="G3"),
        ]
        m = compute_situational_metrics(results)
        assert m.action_accuracy == 1.0
        assert m.extension_target_accuracy == pytest.approx(0.5)

    def test_extension_target_zero_when_no_correct_action(self) -> None:
        results = [_r("extend", "none", expected_gid="G1")]
        m = compute_situational_metrics(results)
        assert m.extension_target_accuracy == 0.0
        assert m.extension_correct_action_count == 0

    def test_failed_results_excluded_from_actual_distribution(self) -> None:
        results = [
            _r("extend", None, success=False),
            _r("none", "none"),
        ]
        m = compute_situational_metrics(results)
        assert m.success_count == 1
        assert m.actual_distribution["none"] == 1
        assert m.actual_distribution["extend"] == 0

    def test_distribution_counts_match_corpus(self) -> None:
        results = [
            _r("extend", "extend"),
            _r("extend", "extend"),
            _r("revise", "revise"),
            _r("none", "none"),
        ]
        m = compute_situational_metrics(results)
        assert m.expected_distribution == {
            "extend": 2, "revise": 1, "create": 0, "none": 1,
        }
        assert m.none_rate_expected == pytest.approx(0.25)

    def test_empty_results_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compute_situational_metrics([])


def _report(run_index: int, results: list[EvalResult]) -> SituationalRunReport:
    return SituationalRunReport(
        arm_name="t", run_index=run_index, model="m",
        seed_groups_restored=5, seed_traces_ingested=20,
        eval_results=results,
    )


class TestAggregateSituationalRuns:
    def test_mean_and_std(self) -> None:
        r1 = _report(0, [_r("extend", "extend", expected_gid="G1", actual_gid="G1")])
        r2 = _report(1, [_r("extend", "none", expected_gid="G1")])
        agg = aggregate_situational_runs("t", "m", [r1, r2])
        assert agg.runs == 2
        assert agg.action_accuracy_mean == pytest.approx(0.5)
        assert agg.action_accuracy_std == pytest.approx(0.7071, abs=0.001)

    def test_single_run_zero_std(self) -> None:
        r1 = _report(0, [_r("none", "none")])
        agg = aggregate_situational_runs("t", "m", [r1])
        assert agg.action_accuracy_mean == 1.0
        assert agg.action_accuracy_std == 0.0

    def test_empty_reports_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            aggregate_situational_runs("t", "m", [])
