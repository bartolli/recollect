"""Surfacing fixture integrity: query labels resolve to traces; arm toml loads."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from probe_cli.arm import load_arm
from probe_cli.corpus import load_corpus, load_query_corpus
from probe_cli.surfacing_runner import SurfacingArmRunner

_FIX = Path(__file__).parent.parent / "fixtures"


def test_query_labels_resolve_to_traces() -> None:
    # A dangling relevant_trace_id silently un-scores a query (it can never be
    # labeled relevant), inflating the distractor pool -- guard against it.
    trace_ids = {t.id for t in load_corpus(_FIX / "eval_traces.jsonl").entries}
    queries = load_query_corpus(_FIX / "eval_queries.jsonl").entries
    for q in queries:
        for tid in q.relevant_trace_ids:
            assert tid in trace_ids, f"query {q.id} references missing trace {tid}"


def test_distractor_queries_present() -> None:
    queries = load_query_corpus(_FIX / "eval_queries.jsonl").entries
    distractors = [q for q in queries if not q.relevant_trace_ids]
    assert len(distractors) == 7


def test_surfacing_arm_toml_loads() -> None:
    arm = load_arm(_FIX / "surfacing.toml")
    assert arm.surfacing.enabled is True
    assert arm.surfacing.traces_corpus_path.endswith("eval_traces.jsonl")
    assert arm.surfacing.query_corpus_path.endswith("eval_queries.jsonl")
    runner = SurfacingArmRunner.from_arm(arm, MagicMock(model_name="m"))
    assert runner.model == "m"
