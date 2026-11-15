"""Drift: each statement's stored shape at the control prompt against each candidate.

No threshold on drift. Significance is reported as raw values: the SDK defines no
tier function over it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from probe_cli.comparison import Confirmation, UnvalidatedInput, confirm_loss
from probe_cli.corpus import _load_jsonl
from probe_cli.representations import RepresentationsReport, StatementRepresentation

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

Certainty = Literal["possible", "conditional", "committed", "opinion"]
DriftEvent = Literal["fact_type_changed", "threshold_crossed", "no_relation"]
RelationLossStatus = Literal["named_loss", "unresolved", "recorded"]

_CANDIDATES = 2


class ReaderKeyRow(BaseModel):
    id: str
    certainty: Certainty


class DriftInput(BaseModel):
    label: str
    report: RepresentationsReport


class DriftInputs(BaseModel):
    control: DriftInput
    candidate: list[DriftInput] = Field(min_length=_CANDIDATES, max_length=_CANDIDATES)
    fresh_control: DriftInput | None = None
    fresh_candidate: DriftInput | None = None

    @model_validator(mode="after")
    def _fresh_is_whole(self) -> DriftInputs:
        if (self.fresh_control is None) != (self.fresh_candidate is None):
            raise ValueError("a fresh pair is one control and one candidate seeding")
        return self


class StatementShape(BaseModel):
    entities: list[str]
    fact_type: str
    significance: float = Field(ge=0.0, le=1.0)
    domains: list[str]
    # Stored entity_relations rows, not the extraction's relation list.
    relations: int = Field(ge=0)
    relations_at_threshold: int = Field(ge=0)


class StatementDrift(BaseModel):
    statement_id: str
    control: StatementShape
    candidate: StatementShape
    entities_added: list[str]
    entities_removed: list[str]
    domains_added: list[str]
    domains_removed: list[str]
    events: list[DriftEvent]


class CandidateDrift(BaseModel):
    label: str
    arm_name: str
    statements: list[StatementDrift]


class RelationLoss(BaseModel):
    # named_loss and unresolved apply to committed statements only; any other
    # certainty is recorded and calls nothing.
    statement_id: str
    certainty: Certainty
    control_at_threshold: int = Field(ge=0)
    candidate_at_threshold: list[int]
    status: RelationLossStatus
    fresh_control_at_threshold: int | None = None
    fresh_candidate_at_threshold: int | None = None
    confirmation: Confirmation | None = None


class DriftReport(BaseModel):
    threshold: float
    control: str
    candidates: list[CandidateDrift]
    relation_losses: list[RelationLoss]
    fresh_control: str | None = None
    fresh_candidate: str | None = None
    unvalidated_inputs: list[UnvalidatedInput] = Field(default_factory=list)


def load_reader_key(path: Path) -> list[ReaderKeyRow]:
    rows, _ = _load_jsonl(path, ReaderKeyRow, "reader key row")
    return rows


def committed_statements(rows: Iterable[ReaderKeyRow]) -> frozenset[str]:
    return frozenset(row.id for row in rows if row.certainty == "committed")


def statement_shape(
    statement: StatementRepresentation, *, threshold: float
) -> StatementShape:
    relations = statement.entity_relations
    return StatementShape(
        entities=[e.name for e in statement.extraction.entities],
        fact_type=statement.extraction.fact_type,
        significance=statement.extraction.significance,
        domains=list(statement.extraction.domains),
        relations=len(relations),
        relations_at_threshold=sum(1 for r in relations if r.weight >= threshold),
    )


def _events(control: StatementShape, candidate: StatementShape) -> list[DriftEvent]:
    events: list[DriftEvent] = []
    if control.fact_type != candidate.fact_type:
        events.append("fact_type_changed")
    if (control.relations_at_threshold > 0) != (candidate.relations_at_threshold > 0):
        events.append("threshold_crossed")
    if control.relations > 0 and candidate.relations == 0:
        events.append("no_relation")
    return events


def _statement_drift(
    statement_id: str, control: StatementShape, candidate: StatementShape
) -> StatementDrift:
    return StatementDrift(
        statement_id=statement_id,
        control=control,
        candidate=candidate,
        entities_added=sorted(set(candidate.entities) - set(control.entities)),
        entities_removed=sorted(set(control.entities) - set(candidate.entities)),
        domains_added=sorted(set(candidate.domains) - set(control.domains)),
        domains_removed=sorted(set(control.domains) - set(candidate.domains)),
        events=_events(control, candidate),
    )


def _shapes(
    seeding: DriftInput, ids: Sequence[str], *, threshold: float
) -> list[StatementShape]:
    by_id = {s.id: s for s in seeding.report.statements}
    missing = [sid for sid in ids if sid not in by_id]
    if missing:
        raise ValueError(f"{seeding.label}: no representation of {missing}")
    return [statement_shape(by_id[sid], threshold=threshold) for sid in ids]


def _status(certainty: Certainty, lost: int) -> RelationLossStatus:
    if certainty != "committed":
        return "recorded"
    return "named_loss" if lost == _CANDIDATES else "unresolved"


def _relation_loss(
    statement_id: str,
    certainty: Certainty,
    control: int,
    candidates: list[int],
) -> RelationLoss | None:
    lost = sum(1 for n in candidates if n == 0)
    if control == 0 or lost == 0:
        return None
    return RelationLoss(
        statement_id=statement_id,
        certainty=certainty,
        control_at_threshold=control,
        candidate_at_threshold=candidates,
        status=_status(certainty, lost),
    )


def _recurrence(
    loss: RelationLoss, fresh_control: int, fresh_candidate: int
) -> RelationLoss:
    return loss.model_copy(
        update={
            "fresh_control_at_threshold": fresh_control,
            "fresh_candidate_at_threshold": fresh_candidate,
            "confirmation": confirm_loss(
                fresh_control_has=fresh_control > 0,
                fresh_candidate_has=fresh_candidate > 0,
            ),
        }
    )


def _certainties(
    key: Sequence[ReaderKeyRow], ids: Sequence[str]
) -> dict[str, Certainty]:
    """Statement id -> certainty, in statement order."""
    by_id = {row.id: row.certainty for row in key}
    missing = [sid for sid in ids if sid not in by_id]
    if missing:
        raise ValueError(f"no reader key row for {missing}")
    return {sid: by_id[sid] for sid in ids}


def _fresh_counts(
    inputs: DriftInputs, ids: Sequence[str], *, threshold: float
) -> list[tuple[int, int]] | None:
    if inputs.fresh_control is None or inputs.fresh_candidate is None:
        return None
    fc = _shapes(inputs.fresh_control, ids, threshold=threshold)
    fk = _shapes(inputs.fresh_candidate, ids, threshold=threshold)
    return [
        (c.relations_at_threshold, k.relations_at_threshold)
        for c, k in zip(fc, fk, strict=True)
    ]


def _relation_losses(
    certainties: dict[str, Certainty],
    control: list[StatementShape],
    candidates: list[list[StatementShape]],
    fresh: list[tuple[int, int]] | None,
) -> list[RelationLoss]:
    losses: list[RelationLoss] = []
    for i, (sid, certainty) in enumerate(certainties.items()):
        at = [shapes[i].relations_at_threshold for shapes in candidates]
        loss = _relation_loss(sid, certainty, control[i].relations_at_threshold, at)
        if loss is None:
            continue
        if fresh is not None and loss.status == "named_loss":
            loss = _recurrence(loss, *fresh[i])
        losses.append(loss)
    return losses


def compare_drift(
    inputs: DriftInputs, *, key: Sequence[ReaderKeyRow], threshold: float
) -> DriftReport:
    ids = [s.id for s in inputs.control.report.statements]
    certainties = _certainties(key, ids)
    control = _shapes(inputs.control, ids, threshold=threshold)
    candidates = [_shapes(c, ids, threshold=threshold) for c in inputs.candidate]
    return DriftReport(
        threshold=threshold,
        control=inputs.control.label,
        candidates=[
            CandidateDrift(
                label=seeding.label,
                arm_name=seeding.report.arm_name,
                statements=[
                    _statement_drift(sid, c, k)
                    for sid, c, k in zip(ids, control, shapes, strict=True)
                ],
            )
            for seeding, shapes in zip(inputs.candidate, candidates, strict=True)
        ],
        relation_losses=_relation_losses(
            certainties,
            control,
            candidates,
            _fresh_counts(inputs, ids, threshold=threshold),
        ),
        fresh_control=inputs.fresh_control.label if inputs.fresh_control else None,
        fresh_candidate=inputs.fresh_candidate.label
        if inputs.fresh_candidate
        else None,
    )
