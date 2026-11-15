"""Reading tally: per reader, qualification-check states and the context verdict."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from probe_cli.reading_packet import (
    ItemKind,
    KeyEntry,
    PacketItem,
    ReaderKeyRow,
    SourceListings,
)
from probe_cli.reading_reader import Answer, ReaderRecord

State = Literal["pass", "fail", "unrepresented"]
# unanswered: items exist, none with a successful answer. no_line: relations but
# no line read against the statement and no fallback line pinned under it.
Cause = Literal[
    "no_relation",
    "fallback_only",
    "empty_context",
    "all_not_applicable",
    "unanswered",
    "no_line",
]
Verdict = Literal["holds", "does_not_hold", "undetermined", "not_applicable"]
_STATES: tuple[State, ...] = ("pass", "fail", "unrepresented")


class TallyError(ValueError):
    pass


class TallyInputs(BaseModel):
    key: list[KeyEntry]
    items: list[PacketItem]
    listings: list[SourceListings]
    rows: list[ReaderKeyRow]
    # Reader name -> its answer records.
    answers: dict[str, list[ReaderRecord]]


class TallyRequest(BaseModel):
    control: str
    candidates: list[str] = Field(default_factory=list)
    confidence_threshold: float = Field(ge=0.0, le=1.0)


class CheckState(BaseModel):
    statement_id: str
    check: int = Field(ge=0)
    state: State
    cause: Cause | None = None


class AddedClaim(BaseModel):
    item_id: str
    statement_id: str
    quotes: list[str]
    quote_missing: bool


class KindTally(BaseModel):
    states: list[CheckState]
    # Observed (statement, check) states; unrepresented is never a pass.
    counts: dict[str, int]
    added_claims: list[AddedClaim]


class TripleFailure(BaseModel):
    item_id: str
    statement_id: str
    check: int = Field(ge=0)
    context_item_id: str


class LineCounts(BaseModel):
    passing: int = Field(ge=0)
    failing: int = Field(ge=0)
    failing_only_through_triple: int = Field(ge=0)


class LineTally(BaseModel):
    all_lines: LineCounts
    at_or_above_threshold: LineCounts
    # Counted in all_lines only.
    weight_none: list[str]
    fails_only_through_triple: list[TripleFailure]


class LabelTally(BaseModel):
    label: str
    context: KindTally
    line: KindTally
    # Decides nothing.
    lines: LineTally


class Regression(BaseModel):
    statement_id: str
    check: int = Field(ge=0)
    check_text: str
    control_state: State
    candidate_label: str
    candidate_state: State
    cause: Cause | None = None


class ReaderTally(BaseModel):
    reader: str
    labels: list[LabelTally]
    incomplete: list[str]
    regressions: list[Regression]
    unrepresented_where_control_passed: list[Regression]
    verdict: Verdict


class Disagreement(BaseModel):
    item_id: str
    label: str
    statement_id: str
    kind: ItemKind
    answers: dict[str, list[Answer]]


class TallyReport(BaseModel):
    control_label: str
    candidate_labels: list[str]
    confidence_threshold: float
    readers: list[ReaderTally]
    disagreements: list[Disagreement]
    # Joint: the readers' common verdict, else undetermined.
    verdict: Verdict


@dataclass(frozen=True)
class _Packet:
    key: list[KeyEntry]
    checks: dict[str, list[str]]
    representations: dict[str, str]
    listings: dict[str, SourceListings]
    # Corpus order: every statement a source holds.
    statements: list[str]
    # (label, kind, statement id) -> key entries.
    groups: dict[tuple[str, str, str], list[KeyEntry]]

    def entries(self, label: str, kind: ItemKind, statement_id: str) -> list[KeyEntry]:
        return self.groups.get((label, kind, statement_id), [])

    def of_kind(self, label: str, kind: ItemKind) -> list[KeyEntry]:
        return [e for e in self.key if e.label == label and e.kind == kind]


@dataclass(frozen=True)
class _Reading:
    packet: _Packet
    # Item id -> successful record.
    answers: dict[str, ReaderRecord]

    def answer(self, entry: KeyEntry, check: int) -> Answer | None:
        record = self.answers.get(entry.item_id)
        return None if record is None else record.answers[check]


def _packet(inputs: TallyInputs) -> _Packet:
    present = {e.statement_id for e in inputs.key if e.kind == "context"}
    for s in inputs.listings:
        present |= set(s.no_relation) | {r.statement_id for r in s.empty_context}
    groups: dict[tuple[str, str, str], list[KeyEntry]] = {}
    for e in inputs.key:
        groups.setdefault((e.label, e.kind, e.statement_id), []).append(e)
    return _Packet(
        key=inputs.key,
        checks={row.id: row.checks for row in inputs.rows},
        representations={i.item_id: i.representation for i in inputs.items},
        listings={s.label: s for s in inputs.listings},
        statements=[row.id for row in inputs.rows if row.id in present],
        groups=groups,
    )


def _reading(packet: _Packet, records: list[ReaderRecord]) -> _Reading:
    """Refuses records of another packet or another key."""
    statement = {e.item_id: e.statement_id for e in packet.key}
    answers: dict[str, ReaderRecord] = {}
    for r in records:
        if r.item_id not in statement:
            raise TallyError(f"answer for {r.item_id}, which is not in the key")
        if r.success:
            if len(r.answers) != len(packet.checks[statement[r.item_id]]):
                raise TallyError(f"{r.item_id}: answer count differs from its checks")
            answers[r.item_id] = r
    return _Reading(packet=packet, answers=answers)


def _missing_cause(
    listings: SourceListings, kind: ItemKind, statement_id: str
) -> Cause:
    no_relation = statement_id in listings.no_relation
    if kind == "context":
        return "no_relation" if no_relation else "empty_context"
    if any(f.pinned_under == statement_id for f in listings.fallback):
        return "fallback_only"
    return "no_relation" if no_relation else "no_line"


def _check_state(
    statement_id: str, check: int, given: list[Answer], missing: Cause
) -> CheckState:
    if "no" in given:
        return CheckState(statement_id=statement_id, check=check, state="fail")
    if "yes" in given:
        return CheckState(statement_id=statement_id, check=check, state="pass")
    cause: Cause = "all_not_applicable" if given else missing
    return CheckState(
        statement_id=statement_id, check=check, state="unrepresented", cause=cause
    )


def _statement_states(
    reading: _Reading, label: str, kind: ItemKind, statement_id: str
) -> list[CheckState]:
    entries = reading.packet.entries(label, kind, statement_id)
    if entries:
        missing: Cause = "unanswered"
    else:
        missing = _missing_cause(reading.packet.listings[label], kind, statement_id)
    states: list[CheckState] = []
    # The last check is the added-claim check; every earlier one qualifies.
    for check in range(len(reading.packet.checks[statement_id]) - 1):
        given = [a for e in entries if (a := reading.answer(e, check)) is not None]
        states.append(_check_state(statement_id, check, given, missing))
    return states


def _added_claims(reading: _Reading, label: str, kind: ItemKind) -> list[AddedClaim]:
    claims: list[AddedClaim] = []
    for e in reading.packet.of_kind(label, kind):
        last = len(reading.packet.checks[e.statement_id]) - 1
        if reading.answer(e, last) == "no":
            quotes = reading.answers[e.item_id].added_claim_quotes
            claims.append(
                AddedClaim(
                    item_id=e.item_id,
                    statement_id=e.statement_id,
                    quotes=quotes,
                    quote_missing=not quotes,
                )
            )
    return claims


def _kind_tally(reading: _Reading, label: str, kind: ItemKind) -> KindTally:
    states = [
        state
        for statement_id in reading.packet.statements
        for state in _statement_states(reading, label, kind, statement_id)
    ]
    observed = Counter(s.state for s in states)
    return KindTally(
        states=states,
        counts={state: observed[state] for state in _STATES},
        added_claims=_added_claims(reading, label, kind),
    )


def _context_twin(reading: _Reading, line: KeyEntry) -> KeyEntry | None:
    """The context item whose text the printed line ends with."""
    text = reading.packet.representations[line.item_id]
    for e in reading.packet.entries(line.label, "context", line.statement_id):
        if text.endswith(f" -- {reading.packet.representations[e.item_id]}"):
            return e
    return None


def _triple_failures(reading: _Reading, line: KeyEntry) -> list[TripleFailure]:
    twin = _context_twin(reading, line)
    if twin is None:
        return []
    failures: list[TripleFailure] = []
    for check in range(len(reading.packet.checks[line.statement_id]) - 1):
        if reading.answer(line, check) == "no" and reading.answer(twin, check) in (
            "yes",
            "not_applicable",
        ):
            failures.append(
                TripleFailure(
                    item_id=line.item_id,
                    statement_id=line.statement_id,
                    check=check,
                    context_item_id=twin.item_id,
                )
            )
    return failures


def _line_counts(outcomes: list[str]) -> LineCounts:
    return LineCounts(
        passing=outcomes.count("passing"),
        failing=outcomes.count("failing") + outcomes.count("triple"),
        failing_only_through_triple=outcomes.count("triple"),
    )


def _line_tally(reading: _Reading, label: str, threshold: float) -> LineTally:
    lines = reading.packet.of_kind(label, "line")
    failures = {e.item_id: _triple_failures(reading, e) for e in lines}
    outcome: dict[str, str] = {}
    for e in lines:
        record = reading.answers.get(e.item_id)
        if record is None:
            continue
        failed = {n for n, a in enumerate(record.answers) if a == "no"}
        through = {f.check for f in failures[e.item_id]}
        # A failed added-claim check is never a triple failure: through holds
        # qualification checks only.
        outcome[e.item_id] = (
            "passing" if not failed else "triple" if failed <= through else "failing"
        )
    above = [e for e in lines if e.weight is not None and e.weight >= threshold]
    return LineTally(
        all_lines=_line_counts(
            [outcome[e.item_id] for e in lines if e.item_id in outcome]
        ),
        at_or_above_threshold=_line_counts(
            [outcome[e.item_id] for e in above if e.item_id in outcome]
        ),
        weight_none=[e.item_id for e in lines if e.weight is None],
        fails_only_through_triple=[f for e in lines for f in failures[e.item_id]],
    )


def _label_tally(reading: _Reading, label: str, threshold: float) -> LabelTally:
    return LabelTally(
        label=label,
        context=_kind_tally(reading, label, "context"),
        line=_kind_tally(reading, label, "line"),
        lines=_line_tally(reading, label, threshold),
    )


def _compared(
    reading: _Reading, labels: dict[str, LabelTally], request: TallyRequest
) -> tuple[list[Regression], list[Regression]]:
    """(pass at control -> fail at a candidate, pass at control -> unrepresented)."""
    regressions: list[Regression] = []
    unrepresented: list[Regression] = []
    passed = [s for s in labels[request.control].context.states if s.state == "pass"]
    for candidate in request.candidates:
        at = {(s.statement_id, s.check): s for s in labels[candidate].context.states}
        for control in passed:
            state = at[(control.statement_id, control.check)]
            entry = Regression(
                statement_id=control.statement_id,
                check=control.check,
                check_text=reading.packet.checks[control.statement_id][control.check],
                control_state=control.state,
                candidate_label=candidate,
                candidate_state=state.state,
                cause=state.cause,
            )
            if state.state == "fail":
                regressions.append(entry)
            elif state.state == "unrepresented":
                unrepresented.append(entry)
    return regressions, unrepresented


def _reader_tally(reading: _Reading, name: str, request: TallyRequest) -> ReaderTally:
    labels = {
        label: _label_tally(reading, label, request.confidence_threshold)
        for label in reading.packet.listings
    }
    incomplete = [
        e.item_id for e in reading.packet.key if e.item_id not in reading.answers
    ]
    verdict: Verdict = "not_applicable"
    regressions: list[Regression] = []
    unrepresented: list[Regression] = []
    if request.candidates:
        regressions, unrepresented = _compared(reading, labels, request)
        claims = sum(len(labels[c].context.added_claims) for c in request.candidates)
        if claims or regressions:
            verdict = "does_not_hold"
        elif incomplete or unrepresented:
            verdict = "undetermined"
        else:
            verdict = "holds"
    return ReaderTally(
        reader=name,
        labels=list(labels.values()),
        incomplete=incomplete,
        regressions=regressions,
        unrepresented_where_control_passed=unrepresented,
        verdict=verdict,
    )


def _disagreements(
    packet: _Packet, readings: dict[str, _Reading]
) -> list[Disagreement]:
    found: list[Disagreement] = []
    for e in packet.key:
        given = {
            name: r.answers[e.item_id].answers
            for name, r in readings.items()
            if e.item_id in r.answers
        }
        if len({tuple(a) for a in given.values()}) > 1:
            found.append(
                Disagreement(
                    item_id=e.item_id,
                    label=e.label,
                    statement_id=e.statement_id,
                    kind=e.kind,
                    answers=given,
                )
            )
    return found


def _checked(inputs: TallyInputs, request: TallyRequest) -> None:
    if len(request.candidates) not in (0, 2):
        raise TallyError(
            f"{len(request.candidates)} candidate labels; give zero or two"
        )
    labels = {s.label for s in inputs.listings}
    unknown = sorted({request.control, *request.candidates} - labels)
    if unknown:
        raise TallyError(f"labels not in the packet: {', '.join(unknown)}")


def tally(inputs: TallyInputs, request: TallyRequest) -> TallyReport:
    _checked(inputs, request)
    packet = _packet(inputs)
    readings = {
        name: _reading(packet, records)
        for name, records in sorted(inputs.answers.items())
    }
    readers = [_reader_tally(r, name, request) for name, r in readings.items()]
    verdicts = {r.verdict for r in readers}
    joint: Verdict = "not_applicable"
    if request.candidates:
        joint = verdicts.pop() if len(verdicts) == 1 else "undetermined"
    return TallyReport(
        control_label=request.control,
        candidate_labels=request.candidates,
        confidence_threshold=request.confidence_threshold,
        readers=readers,
        disagreements=_disagreements(packet, readings),
        verdict=joint,
    )


def summary(report: TallyReport) -> list[str]:
    candidates = ",".join(report.candidate_labels) or "-"
    lines = [
        f"tally control={report.control_label} candidates={candidates}"
        f"  threshold={report.confidence_threshold:.2f}"
    ]
    for reader in report.readers:
        lines.append(
            f"reader {reader.reader}: verdict={reader.verdict}"
            f"  incomplete={len(reader.incomplete)}"
            f"  regressions={len(reader.regressions)}"
            f"  unrepresented={len(reader.unrepresented_where_control_passed)}"
        )
        lines += [_label_line(t) for t in reader.labels]
    lines.append(f"disagreements={len(report.disagreements)}  verdict={report.verdict}")
    return lines


def _kind_text(k: KindTally) -> str:
    c = k.counts
    return (
        f"pass={c['pass']} fail={c['fail']} unrep={c['unrepresented']}"
        f" claims={len(k.added_claims)}"
    )


def _counts_text(c: LineCounts) -> str:
    return f"{c.passing}/{c.failing}/{c.failing_only_through_triple}"


def _label_line(t: LabelTally) -> str:
    return (
        f"  {t.label}  context {_kind_text(t.context)}  line {_kind_text(t.line)}"
        f"  lines pass/fail/triple all={_counts_text(t.lines.all_lines)}"
        f" at_threshold={_counts_text(t.lines.at_or_above_threshold)}"
        f" weight_none={len(t.lines.weight_none)}"
    )
