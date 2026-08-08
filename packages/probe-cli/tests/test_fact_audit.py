"""classify_tail gate-stage branches, aggregation, scratch-DB guard."""

from __future__ import annotations

import pytest
from probe_cli.fact_audit import (
    AuditRound,
    Classification,
    TailOutcome,
    _require_scratch_db,
    aggregate_outcomes,
    classify_tail,
)
from recollect.models import FactStatus, PersonaFact

_ALIASES = ["6pm", "6 pm", "after 6pm"]


def _fact(status: FactStatus, obj: str, content: str = "") -> PersonaFact:
    return PersonaFact(
        subject="Oak Street parking",
        predicate="is_available_at",
        object=obj,
        content=content or f"Oak Street parking is_available_at {obj}",
        status=status,
        source_trace_id="t1",
        user_id="u1",
    )


def _rel(target: str, confidence: float = 0.9, context: str = "") -> dict[str, object]:
    return {
        "source": "Oak Street parking",
        "relation": "is_available_at",
        "target": target,
        "confidence": confidence,
        "context": context,
    }


def _classify(pattern: dict[str, object], facts: list[PersonaFact]) -> TailOutcome:
    return classify_tail(
        tail_id="chain-c-6",
        question_id="t3q06",
        pattern=pattern,
        facts=facts,
        aliases=_ALIASES,
        confidence_threshold=0.6,
    )


def test_promoted_spo_carries_answer() -> None:
    out = _classify(
        {"fact_type": "semantic", "relations": [_rel("after 6pm")]},
        [_fact("promoted", "free after 6pm")],
    )
    assert out.classification == "promoted"
    assert out.gate_stage == ""


def test_promoted_but_answer_only_in_content_is_spo_lossy() -> None:
    out = _classify(
        {"fact_type": "semantic", "relations": [_rel("after 6pm")]},
        [_fact("promoted", "metered weekdays", content="free after 6pm on Oak")],
    )
    assert out.classification == "gated"
    assert out.gate_stage == "spo_lossy"


def test_candidate_row_is_gated() -> None:
    out = _classify(
        {"fact_type": "semantic", "relations": [_rel("after 6pm")]},
        [_fact("candidate", "free after 6pm")],
    )
    assert out.classification == "gated"
    assert out.gate_stage == "candidate"


def test_episodic_fact_type_gates_extracted_relation() -> None:
    out = _classify(
        {"fact_type": "episodic", "relations": [_rel("after 6pm")]},
        [],
    )
    assert out.classification == "gated"
    assert out.gate_stage == "fact_type"


def test_low_confidence_relation_gates_pre_write() -> None:
    out = _classify(
        {"fact_type": "semantic", "relations": [_rel("after 6pm", confidence=0.5)]},
        [],
    )
    assert out.classification == "gated"
    assert out.gate_stage == "confidence"


def test_confident_semantic_relation_without_row_is_swallowed() -> None:
    out = _classify(
        {"fact_type": "semantic", "relations": [_rel("after 6pm")]},
        [],
    )
    assert out.classification == "gated"
    assert out.gate_stage == "swallowed"


def test_no_answer_signal_is_never_extracted() -> None:
    out = _classify(
        {"fact_type": "semantic", "relations": [_rel("metered parking")]},
        [_fact("candidate", "metered until evening")],
    )
    assert out.classification == "never_extracted"
    assert out.answer_relations == 0


def test_alias_matches_at_word_boundary_only() -> None:
    # "16pm" must not match the "6pm" alias.
    out = _classify(
        {"fact_type": "semantic", "relations": [_rel("16pm")]},
        [],
    )
    assert out.classification == "never_extracted"


def test_aggregate_counts_per_tail() -> None:
    def outcome(cls: Classification) -> TailOutcome:
        return TailOutcome(
            tail_id="chain-c-6",
            question_id="t3q06",
            classification=cls,
        )

    rounds = [
        AuditRound(round_index=0, seeded=1, outcomes=[outcome("promoted")]),
        AuditRound(round_index=1, seeded=1, outcomes=[outcome("gated")]),
        AuditRound(round_index=2, seeded=1, outcomes=[outcome("gated")]),
    ]
    assert aggregate_outcomes(rounds) == {"chain-c-6": {"promoted": 1, "gated": 2}}


def test_scratch_db_guard_rejects_other_databases() -> None:
    with pytest.raises(ValueError, match="probe_fact_audit"):
        _require_scratch_db("postgresql://u@localhost:5432/probe_task")
    with pytest.raises(ValueError, match="probe_fact_audit"):
        _require_scratch_db("")
    _require_scratch_db("postgresql://u@localhost:5432/probe_fact_audit")
