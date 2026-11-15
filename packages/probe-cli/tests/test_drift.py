"""Drift: per-statement shape, named events, committed relation loss and recurrence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from probe_cli.drift import (
    DriftInput,
    DriftInputs,
    DriftReport,
    ReaderKeyRow,
    committed_statements,
    compare_drift,
    load_reader_key,
)
from probe_cli.representations import (
    RelationRow,
    RepresentationsReport,
    StatementRepresentation,
)
from recollect.llm.types import ExtractionResult

_KEY_PATH = Path(__file__).parent.parent / "fixtures/qualification/reader_key.jsonl"
_THRESHOLD = 0.6
_KEY = [
    ReaderKeyRow(id="s", certainty="committed"),
    ReaderKeyRow(id="h", certainty="possible"),
]


def _stmt(
    sid: str = "s", *weights: float, **extraction: Any
) -> StatementRepresentation:
    return StatementRepresentation(
        id=sid,
        trace_content="text",
        trace_status="active",
        extraction=ExtractionResult.model_validate(extraction),
        degraded=False,
        entity_relations=[
            RelationRow(triple="a b c", context="", weight=w) for w in weights
        ],
        persona_facts=[],
        pinned_fact_lines=[],
    )


def _in(label: str, *statements: StatementRepresentation) -> DriftInput:
    report = RepresentationsReport(
        arm_name=label,
        template_db="db",
        user_id="u",
        traces_path="t.jsonl",
        traces_sha256="0",
        degraded_statements=[],
        statements=list(statements),
    )
    return DriftInput(label=label, report=report)


def _drift(
    control: StatementRepresentation,
    candidates: tuple[StatementRepresentation, StatementRepresentation],
    fresh: tuple[StatementRepresentation, StatementRepresentation] | None = None,
) -> DriftReport:
    inputs = DriftInputs(
        control=_in("c", control),
        candidate=[_in("k1", candidates[0]), _in("k2", candidates[1])],
        fresh_control=_in("fc", fresh[0]) if fresh else None,
        fresh_candidate=_in("fk", fresh[1]) if fresh else None,
    )
    return compare_drift(inputs, key=_KEY, threshold=_THRESHOLD)


def test_shape_values_and_diffs_are_reported_without_a_threshold() -> None:
    control = _stmt(
        "s",
        0.9,
        0.3,
        entities=[{"name": "Ana"}, {"name": "Bo"}],
        significance=0.3,
        domains=["travel", "social"],
    )
    candidate = _stmt(
        "s",
        0.9,
        entities=[{"name": "Ana"}, {"name": "Cy"}],
        significance=0.35,
        domains=["travel", "general"],
    )

    d = _drift(control, (candidate, control)).candidates[0].statements[0]

    assert (d.entities_added, d.entities_removed) == (["Cy"], ["Bo"])
    assert (d.domains_added, d.domains_removed) == (["general"], ["social"])
    assert (d.control.significance, d.candidate.significance) == (0.3, 0.35)
    assert (d.control.relations, d.candidate.relations) == (2, 1)
    assert (d.control.relations_at_threshold, d.candidate.relations_at_threshold) == (
        1,
        1,
    )
    assert d.events == []


def test_a_changed_fact_type_is_named() -> None:
    semantic = _stmt(fact_type="semantic")
    d = _drift(semantic, (_stmt(fact_type="episodic"), semantic))

    assert d.candidates[0].statements[0].events == ["fact_type_changed"]
    assert d.candidates[1].statements[0].events == []


def test_the_threshold_is_crossed_in_both_directions_and_a_tie_counts() -> None:
    up = _drift(_stmt("s", 0.5), (_stmt("s", 0.6), _stmt("s", 0.59)))
    down = _drift(_stmt("s", 0.6), (_stmt("s", 0.5), _stmt("s", 0.7)))

    assert [c.statements[0].events for c in up.candidates] == [
        ["threshold_crossed"],
        [],
    ]
    assert [c.statements[0].events for c in down.candidates] == [
        ["threshold_crossed"],
        [],
    ]


def test_a_statement_left_with_no_relation_is_named() -> None:
    d = _drift(_stmt("s", 0.3), (_stmt("s"), _stmt("s", 0.3)))

    assert d.candidates[0].statements[0].events == ["no_relation"]


def test_the_committed_statements_of_the_frozen_key() -> None:
    assert committed_statements(load_reader_key(_KEY_PATH)) == {
        "will-1",
        "will-2",
        "will-3",
        "corrected-1a",
        "corrected-1b",
        "corrected-2a",
        "corrected-2b",
        "screen-4",
    }


def test_a_committed_statement_without_relation_in_both_candidates_is_a_loss() -> None:
    d = _drift(_stmt("s", 0.9), (_stmt("s", 0.3), _stmt("s")))

    loss = d.relation_losses[0]
    assert (loss.statement_id, loss.status) == ("s", "named_loss")
    assert (loss.control_at_threshold, loss.candidate_at_threshold) == (1, [0, 0])
    assert loss.confirmation is None


def test_a_loss_in_one_candidate_is_unresolved() -> None:
    d = _drift(_stmt("s", 0.9), (_stmt("s", 0.3), _stmt("s", 0.9)))

    assert [loss.status for loss in d.relation_losses] == ["unresolved"]


def test_a_hedged_statement_losing_its_relation_is_recorded() -> None:
    d = _drift(_stmt("h", 0.9), (_stmt("h"), _stmt("h")))

    assert [(loss.certainty, loss.status) for loss in d.relation_losses] == [
        ("possible", "recorded")
    ]


def test_a_statement_that_keeps_its_relation_has_no_loss_entry() -> None:
    assert (
        _drift(_stmt("s", 0.9), (_stmt("s", 0.9), _stmt("s", 0.8))).relation_losses
        == []
    )


def test_the_loss_recurs_iff_the_fresh_control_holds_and_the_candidate_not() -> None:
    lost = (_stmt("s"), _stmt("s"))

    recurs = _drift(_stmt("s", 0.9), lost, fresh=(_stmt("s", 0.9), _stmt("s", 0.3)))
    kept = _drift(_stmt("s", 0.9), lost, fresh=(_stmt("s", 0.9), _stmt("s", 0.9)))
    control_lost = _drift(_stmt("s", 0.9), lost, fresh=(_stmt("s"), _stmt("s")))

    assert recurs.relation_losses[0].confirmation == "confirmed"
    assert recurs.relation_losses[0].fresh_control_at_threshold == 1
    assert kept.relation_losses[0].confirmation == "unresolved"
    assert control_lost.relation_losses[0].confirmation == "unresolved"


def test_a_statement_missing_from_the_key_or_a_candidate_is_refused() -> None:
    with pytest.raises(ValueError, match="reader key"):
        _drift(_stmt("x", 0.9), (_stmt("x"), _stmt("x")))
    with pytest.raises(ValueError, match="k1"):
        _drift(_stmt("s", 0.9), (_stmt("h"), _stmt("s")))
