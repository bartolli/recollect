"""Associative corpus integrity: every cross-reference resolves (slice-1c).

A dangling member/relevant/forbid id silently distorts the per-case metric, so
guard the corpus before the expensive live run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from probe_cli.arm import load_arm
from probe_cli.corpus import (
    load_corpus,
    load_ground_truth,
    load_query_corpus,
    load_seed_groups,
)

_FIX = Path(__file__).parent.parent / "fixtures" / "associative"


def _trace_ids() -> set[str]:
    return {e.id for e in load_corpus(_FIX / "seed_traces.jsonl").entries}


def test_seed_group_members_exist() -> None:
    ids = _trace_ids()
    for g in load_seed_groups(_FIX / "seed_groups.jsonl"):
        for m in g.member_trace_ids:
            assert m in ids, f"group {g.group_id} references missing trace {m}"


def test_query_relevant_ids_exist() -> None:
    ids = _trace_ids()
    for q in load_query_corpus(_FIX / "queries.jsonl").entries:
        for tid in q.relevant_trace_ids:
            assert tid in ids, f"query {q.id} references missing trace {tid}"


def test_ground_truth_covers_queries_and_forbid_resolves() -> None:
    ids = _trace_ids()
    query_ids = {q.id for q in load_query_corpus(_FIX / "queries.jsonl").entries}
    gt_ids = {g.query_id for g in load_ground_truth(_FIX / "ground_truth.jsonl")}
    assert gt_ids == query_ids, "ground truth and queries must cover the same ids"
    for g in load_ground_truth(_FIX / "ground_truth.jsonl"):
        for tid in g.forbid:
            assert tid in ids, f"{g.query_id} forbids missing trace {tid}"


def test_surface_and_forbid_disjoint() -> None:
    # A fact cannot be both expected-to-surface and forbidden for one query.
    surface = {
        q.id: set(q.relevant_trace_ids)
        for q in load_query_corpus(_FIX / "queries.jsonl").entries
    }
    for g in load_ground_truth(_FIX / "ground_truth.jsonl"):
        overlap = surface.get(g.query_id, set()) & set(g.forbid)
        assert not overlap, f"{g.query_id}: surface/forbid overlap {overlap}"


def test_arm_toml_routes_situational_surfacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECOLLECT_PROBE_DB_URL", "postgresql://localhost/probe_eval")
    arm = load_arm(
        Path(__file__).parent.parent / "fixtures" / "surfacing-associative.toml"
    )
    assert arm.surfacing.enabled
    assert arm.surfacing.seed_groups_path.endswith("seed_groups.jsonl")
    assert arm.surfacing.ground_truth_path.endswith("ground_truth.jsonl")
