"""Surfacing calls: measures against two controls, named losses and gains, verdicts."""

from __future__ import annotations

from pathlib import Path

import pytest
from probe_cli.comparison import Seedings
from probe_cli.corpus import QueryEntry, load_query_corpus
from probe_cli.surfacing_compare import (
    SurfacingComparison,
    SurfacingInput,
    SurfacingReading,
    compare_surfacing,
    load_reading,
    required_pairs,
)
from probe_cli.surfacing_runner import (
    PoolFact,
    QuerySurfacing,
    SurfacedFact,
    SurfacingRunReport,
)

# reading.toml names its query corpus from the repository root.
_REPO = Path(__file__).resolve().parents[3]
_IDS = ("a", "b", "c", "d")
_QUERIES = [QueryEntry(id=f"q-{i}", text="", relevant_trace_ids=[i]) for i in _IDS]
_ALL = set(_IDS)


def _run(
    covered: set[str], *, under: set[str] | None = None, noise: int = 0
) -> SurfacingInput:
    """covered: surfaced and gated; under: surfaced below the gate; rest ranked out."""
    surfaced = covered | (under or set())

    def fact(src: str, *, gate: bool, relevant: bool) -> SurfacedFact:
        return SurfacedFact(
            fact_id=src,
            source_eval_id=src,
            score=0.8,
            relevant=relevant,
            passes_gate=gate,
        )

    queries = [
        QuerySurfacing(
            query_id=f"q-{i}",
            surfaced=[fact(i, gate=i in covered, relevant=True)]
            if i in surfaced
            else [],
        )
        for i in _IDS
    ]
    queries.append(
        QuerySurfacing(
            query_id="q-x", surfaced=[fact("x", gate=True, relevant=False)] * noise
        )
    )
    report = SurfacingRunReport(
        arm_name="arm",
        model="m",
        seeded_traces=4,
        promoted_facts=0,
        queries=queries,
        surfacing_pool=[PoolFact(fact_id=i, source_eval_id=i) for i in _IDS],
    )
    return SurfacingInput(label="run", report=report)


def _compare(
    control: list[SurfacingInput],
    candidate: list[SurfacingInput],
    fresh: tuple[SurfacingInput, SurfacingInput] | None = None,
    overlap: list[tuple[str, str]] | None = None,
) -> SurfacingComparison:
    inputs = Seedings[SurfacingInput](
        control=control,
        candidate=candidate,
        fresh_control=fresh[0] if fresh else None,
        fresh_candidate=fresh[1] if fresh else None,
    )
    reading = SurfacingReading(queries="q.jsonl", overlap_pairs=overlap or [])
    return compare_surfacing(inputs, _QUERIES, reading=reading, floor=0.65)


def _calls(c: SurfacingComparison) -> dict[str, str]:
    return {m.name: m.call for m in c.measures}


def test_identical_seedings_call_nothing() -> None:
    c = _compare([_run(_ALL)] * 2, [_run(_ALL)] * 2)

    assert set(_calls(c).values()) == {"none"}
    assert (c.named_losses, c.named_gains) == ([], [])
    assert c.verdict == "no_regression_called"


def test_a_named_loss_lists_its_query_and_the_cause_in_each_candidate() -> None:
    c = _compare(
        [_run(_ALL)] * 2, [_run(_ALL - {"a"}), _run(_ALL - {"a"}, under={"a"})]
    )

    assert [(n.query_id, n.required_id) for n in c.named_losses] == [("q-a", "a")]
    assert c.named_losses[0].candidate_causes == ["ranked_out", "under_floor"]
    assert c.named_losses[0].confirmation is None
    assert _calls(c)["coverage_all"] == "regression"
    assert c.verdict == "needs_confirmation"


def test_one_candidate_below_without_a_named_loss_is_unresolved() -> None:
    c = _compare([_run(_ALL)] * 2, [_run(_ALL), _run(_ALL - {"b"})])

    assert c.named_losses == []
    assert _calls(c)["coverage_all"] == "unresolved"
    assert c.verdict == "unresolved"


def test_a_gain_on_an_overlap_pair_is_flagged_and_calls_nothing() -> None:
    c = _compare(
        [_run(_ALL - {"a", "b"})] * 2, [_run(_ALL)] * 2, overlap=[("q-a", "a")]
    )

    flags = {g.required_id: g.exemplar_overlap for g in c.named_gains}
    assert flags == {"a": True, "b": False}
    assert c.verdict == "no_regression_called"


def test_coverage_without_overlap_excludes_the_overlap_pair() -> None:
    c = _compare(
        [_run(_ALL)] * 2, [_run(_ALL - {"a"}), _run(_ALL)], overlap=[("q-a", "a")]
    )

    assert _calls(c)["coverage_all"] == "unresolved"
    assert _calls(c)["coverage_without_overlap"] == "none"


def test_no_floor_pass_in_a_seeding_leaves_precision_unresolved_with_a_reason() -> None:
    c = _compare([_run(_ALL)] * 2, [_run(_ALL), _run(set(), under=_ALL)])

    precision = next(m for m in c.measures if m.name == "precision_floor")
    assert precision.call == "unresolved"
    assert "no value" in precision.reason


def test_precision_counts_floor_passes_from_distractor_queries() -> None:
    c = _compare([_run(_ALL)] * 2, [_run(_ALL, noise=1)] * 2)

    assert _calls(c)["precision_floor"] == "regression"
    assert c.seedings.candidate[0].precision_floor == pytest.approx(0.8)


def test_a_fresh_pair_that_repeats_the_loss_holds() -> None:
    lost = _run(_ALL - {"a"})
    c = _compare([_run(_ALL)] * 2, [lost, lost], fresh=(_run(_ALL), lost))

    assert c.named_losses[0].confirmation == "confirmed"
    assert next(m for m in c.measures if m.name == "coverage_all").confirmation == (
        "confirmed"
    )
    assert c.verdict == "hold_confirmed"


def test_a_fresh_pair_that_does_not_repeat_the_loss_is_unresolved() -> None:
    lost = _run(_ALL - {"a"})
    c = _compare([_run(_ALL)] * 2, [lost, lost], fresh=(lost, lost))

    assert c.named_losses[0].confirmation == "unresolved"
    assert c.verdict == "unresolved"


def test_an_overlap_pair_absent_from_the_labels_is_refused() -> None:
    with pytest.raises(ValueError, match="absent from the query labels"):
        _compare([_run(_ALL)] * 2, [_run(_ALL)] * 2, overlap=[("q-a", "b")])


def test_the_frozen_fixture_has_26_pairs_over_23_queries_and_both_overlap_pairs() -> (
    None
):
    reading = load_reading(
        _REPO / "packages/probe-cli/fixtures/surfacing-h55/reading.toml"
    )
    pairs = required_pairs(load_query_corpus(_REPO / reading.queries).entries)

    assert len(pairs) == len(set(pairs)) == 26
    assert len({qid for qid, _ in pairs}) == 23
    assert reading.overlap_pairs == [("q15", "p3"), ("q17", "c2")]
    assert set(reading.overlap_pairs) <= set(pairs)
