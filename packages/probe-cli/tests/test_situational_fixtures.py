"""P6 fixture sanity checks: schema-drift + cross-reference integrity."""

from __future__ import annotations

from pathlib import Path

from probe_cli.corpus import load_eval_corpus, load_seed_groups, load_seed_traces

_FIX = Path(__file__).parent.parent / "fixtures" / "situational"


def test_seed_traces_load() -> None:
    traces = load_seed_traces(_FIX / "seed_traces.jsonl")
    assert len(traces) == 20
    grouped = [t for t in traces if t.seed_group_id is not None]
    assert len(grouped) == 14


def test_seed_groups_load() -> None:
    groups = load_seed_groups(_FIX / "seed_groups.jsonl")
    assert {g.group_id for g in groups} == {"G1", "G2", "G3", "G4", "G5"}


def test_eval_action_distribution() -> None:
    entries = load_eval_corpus(_FIX / "eval.jsonl")
    assert len(entries) == 15
    counts = dict.fromkeys(("extend", "revise", "create", "none"), 0)
    for e in entries:
        counts[e.expected_action] += 1
    assert counts == {"extend": 5, "revise": 3, "create": 1, "none": 6}


def test_member_ids_resolve_to_seed_traces() -> None:
    traces = load_seed_traces(_FIX / "seed_traces.jsonl")
    groups = load_seed_groups(_FIX / "seed_groups.jsonl")
    trace_ids = {t.id for t in traces}
    for g in groups:
        for m in g.member_trace_ids:
            assert m in trace_ids, f"{g.group_id} references missing trace {m}"


def test_eval_group_refs_resolve_to_seed_groups() -> None:
    entries = load_eval_corpus(_FIX / "eval.jsonl")
    groups = load_seed_groups(_FIX / "seed_groups.jsonl")
    group_ids = {g.group_id for g in groups}
    for e in entries:
        if e.expected_action in ("extend", "revise"):
            assert e.expected_group_id in group_ids


def test_eval_create_links_resolve_to_seed_traces() -> None:
    traces = load_seed_traces(_FIX / "seed_traces.jsonl")
    entries = load_eval_corpus(_FIX / "eval.jsonl")
    trace_ids = {t.id for t in traces}
    for e in entries:
        if e.expected_action == "create":
            assert e.expected_linked_trace_ids
            for tid in e.expected_linked_trace_ids:
                assert tid in trace_ids
