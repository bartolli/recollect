"""Surfacing fixture integrity: query labels resolve to traces; arm toml loads."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from probe_cli.arm import load_arm
from probe_cli.corpus import load_corpus, load_query_corpus
from probe_cli.surfacing_runner import SurfacingArmRunner
from recollect.prompts import load_packaged_default

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


def test_surfacing_arm_toml_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fixture references the probe DB by env name (.env single source of
    # truth); load_arm's guard raises on an unset var, so provide it here.
    monkeypatch.setenv("RECOLLECT_PROBE_DB_URL", "postgresql://localhost/probe_eval")
    arm = load_arm(_FIX / "surfacing.toml")
    assert arm.surfacing.enabled is True
    assert arm.surfacing.traces_corpus_path.endswith("eval_traces.jsonl")
    assert arm.surfacing.query_corpus_path.endswith("eval_queries.jsonl")
    runner = SurfacingArmRunner.from_arm(arm, MagicMock(model_name="m"))
    assert runner.model == "m"


def test_h55_arms_equal_the_surfacing_arm_except_model_prompt_and_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECOLLECT_PROBE_DB_URL", "postgresql://localhost/probe_eval")
    base = load_arm(_FIX / "surfacing.toml")
    candidate_prompt = "qualification/extraction.1.3.0.md"
    packaged = load_packaged_default("extraction.default.md").version
    for prompt in ("p121", "p130"):
        arm = load_arm(_FIX / "surfacing-h55" / f"{prompt}.toml")
        assert arm.name == f"surfacing-h55-{prompt}"
        assert arm.extraction.pydantic_ai_model == "anthropic:claude-haiku-5-5"
        template = arm.extraction.template_path
        assert template.endswith(candidate_prompt) == (prompt == "p130")
        assert bool(template) == (prompt == "p130")
        runner = SurfacingArmRunner.from_arm(arm, MagicMock(model_name="m"))
        loaded = runner._extractor.template_version
        assert loaded == ("1.3.0" if prompt == "p130" else packaged)
        assert arm.extraction.instructions
        same = {"pydantic_ai_model": "", "template_path": "", "instructions": ""}
        assert arm.extraction.model_copy(update=same) == (
            base.extraction.model_copy(update=same)
        )
        assert (arm.corpus, arm.surfacing, arm.recollect_overrides) == (
            base.corpus,
            base.surfacing,
            base.recollect_overrides,
        )
