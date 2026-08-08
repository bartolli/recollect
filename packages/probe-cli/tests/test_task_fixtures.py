"""Task-eval fixture sanity: schema-drift + cross-reference + tier invariants."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from probe_cli.corpus import load_task_questions, load_task_seed_traces

_FIX = Path(__file__).parent.parent / "fixtures" / "task"


def test_seed_traces_load() -> None:
    seeds = load_task_seed_traces(_FIX / "seed_traces.jsonl")
    assert len(seeds) == 208
    tiers = Counter(s.density_tier for s in seeds)
    assert (tiers[0], tiers[1], tiers[2]) == (66, 71, 71)
    assert len({s.id for s in seeds}) == len(seeds)


def test_questions_load() -> None:
    questions = load_task_questions(_FIX / "questions.jsonl")
    assert len(questions) == 40
    # Post-verification distribution: t3q02 and t1q06 demoted to any by the
    # 2026-08-07 reachability run (multi-path reach; see story-5 record).
    # Batch-2 verification chains add t3q07-t3q10. Stable-drift demotion
    # (five seedings): t3q09/t3q10 to t2 -- t3 is n=7, six composition-cut
    # plus the one true hop case (t3q01).
    labels = Counter(q.tier_label for q in questions)
    assert labels == {"t3": 7, "t1": 5, "t2": 8, "any": 14, "none": 6}


def test_required_traces_resolve_and_are_core_tier() -> None:
    seeds = {s.id: s for s in load_task_seed_traces(_FIX / "seed_traces.jsonl")}
    for q in load_task_questions(_FIX / "questions.jsonl"):
        for tid in q.requires_trace_ids:
            assert tid in seeds, f"{q.id} references missing trace {tid}"
            assert seeds[tid].density_tier == 0, f"{q.id} requires non-core trace {tid}"


def test_abstain_questions_require_no_traces() -> None:
    for q in load_task_questions(_FIX / "questions.jsonl"):
        if q.answer_type == "abstain":
            assert q.tier_label == "none"
            assert q.requires_trace_ids == []
        else:
            assert q.requires_trace_ids, f"{q.id} has no required traces"
