"""Arm config loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from probe_cli.arm import load_arm
from pydantic import ValidationError


def _write_arm(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "arm.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_minimal_arm_loads(tmp_path: Path) -> None:
    p = _write_arm(
        tmp_path,
        '[arm]\nname = "baseline"\nruns = 2\n[corpus]\npath = "x.jsonl"\n',
    )
    arm = load_arm(p)
    assert arm.name == "baseline"
    assert arm.runs == 2
    assert arm.corpus.path == "x.jsonl"


def test_extraction_block_overrides_defaults(tmp_path: Path) -> None:
    p = _write_arm(
        tmp_path,
        '[arm]\nname = "dense"\n'
        "[extraction]\n"
        'pydantic_ai_model = "anthropic:claude-haiku-4-5"\n'
        'template_path = "/tmp/dense.md"\n'
        "max_concepts = 7\n"
        "[corpus]\npath = \"x.jsonl\"\n",
    )
    arm = load_arm(p)
    assert arm.extraction.pydantic_ai_model == "anthropic:claude-haiku-4-5"
    assert arm.extraction.template_path == "/tmp/dense.md"
    assert arm.extraction.max_concepts == 7


def test_to_memory_config_isolated(tmp_path: Path) -> None:
    p = _write_arm(
        tmp_path,
        '[arm]\nname = "x"\n'
        '[extraction]\nmax_concepts = 9\nmax_relations = 4\n'
        "[corpus]\npath = \"x.jsonl\"\n",
    )
    arm = load_arm(p)
    cfg = arm.to_memory_config()
    assert cfg.get("extraction.max_concepts") == 9
    assert cfg.get("extraction.max_relations") == 4
    # Mutating the arm config does not bleed into a fresh one
    cfg2 = arm.to_memory_config()
    cfg._set("extraction.max_concepts", 1)
    assert cfg2.get("extraction.max_concepts") == 9


def test_runs_must_be_positive(tmp_path: Path) -> None:
    p = _write_arm(
        tmp_path,
        '[arm]\nname = "x"\nruns = 0\n[corpus]\npath = "x.jsonl"\n',
    )
    with pytest.raises(ValidationError):
        load_arm(p)


def test_default_runs_is_three(tmp_path: Path) -> None:
    p = _write_arm(tmp_path, '[arm]\nname = "x"\n[corpus]\npath = "x.jsonl"\n')
    arm = load_arm(p)
    assert arm.runs == 3


def test_default_output_dir(tmp_path: Path) -> None:
    p = _write_arm(tmp_path, '[arm]\nname = "x"\n[corpus]\npath = "x.jsonl"\n')
    arm = load_arm(p)
    assert arm.output.dir == "out"


def test_retrieval_block_loads(tmp_path: Path) -> None:
    p = _write_arm(
        tmp_path,
        '[arm]\nname = "eval"\n'
        '[corpus]\npath = "x.jsonl"\n'
        "[retrieval]\n"
        "enabled = true\n"
        'db_url = "postgresql://localhost/probe"\n'
        'traces_corpus_path = "traces.jsonl"\n'
        'query_corpus_path = "queries.jsonl"\n'
        "token_budget = 50000\n"
        "top_k = 10\n",
    )
    arm = load_arm(p)
    assert arm.retrieval.enabled is True
    assert arm.retrieval.token_budget == 50000
    assert arm.retrieval.top_k == 10
    assert arm.retrieval.traces_corpus_path == "traces.jsonl"


def test_retrieval_defaults_when_absent(tmp_path: Path) -> None:
    p = _write_arm(
        tmp_path, '[arm]\nname = "x"\n[corpus]\npath = "x.jsonl"\n'
    )
    arm = load_arm(p)
    assert arm.retrieval.enabled is False
    assert arm.retrieval.top_k == 5
    assert arm.retrieval.token_budget == 100_000


def test_recollect_overrides_apply_to_memory_config(tmp_path: Path) -> None:
    p = _write_arm(
        tmp_path,
        '[arm]\nname = "x"\n[corpus]\npath = "x.jsonl"\n'
        "[recollect_overrides]\n"
        '"persona.auto_extract" = false\n'
        '"recall_tokens.enabled" = false\n'
        '"retrieval.spread_seed_count" = 7\n',
    )
    arm = load_arm(p)
    cfg = arm.to_memory_config()
    assert cfg.get("persona.auto_extract") is False
    assert cfg.get("recall_tokens.enabled") is False
    assert cfg.get("retrieval.spread_seed_count") == 7


def test_retrieval_max_retrievals_buffer_overrides_default(tmp_path: Path) -> None:
    p = _write_arm(
        tmp_path,
        '[arm]\nname = "x"\n[corpus]\npath = "x.jsonl"\n'
        "[retrieval]\n"
        "enabled = true\n"
        'traces_corpus_path = "t.jsonl"\n'
        'query_corpus_path = "q.jsonl"\n'
        "max_retrievals_buffer = 75\n",
    )
    arm = load_arm(p)
    cfg = arm.to_memory_config()
    assert cfg.get("retrieval.max_retrievals") == 75


def test_situational_block_loads(tmp_path: Path) -> None:
    p = _write_arm(
        tmp_path,
        '[arm]\nname = "s"\n[corpus]\npath = ""\n'
        "[situational]\n"
        "enabled = true\n"
        'db_url = "postgresql://x/y"\n'
        'seed_traces_path = "seed.jsonl"\n'
        'seed_groups_path = "groups.jsonl"\n'
        'eval_corpus_path = "eval.jsonl"\n',
    )
    arm = load_arm(p)
    assert arm.situational.enabled
    assert arm.situational.db_url == "postgresql://x/y"
    assert arm.situational.seed_traces_path == "seed.jsonl"


def test_situational_defaults_when_absent(tmp_path: Path) -> None:
    p = _write_arm(tmp_path, '[arm]\nname = "x"\n[corpus]\npath = "x.jsonl"\n')
    arm = load_arm(p)
    assert arm.situational.enabled is False
    assert arm.situational.seed_traces_path == ""
