"""Reading packet: every relation context and distinct pinned line, shuffled."""

from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

from probe_cli.representations import RepresentationsReport

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from probe_cli.representations import StatementRepresentation

logger = logging.getLogger(__name__)

# Last check of every key row, held once: rows carry one to three kind checks first.
ADDED_CLAIM_CHECK = "Does it stay within what the statement says, adding no claim?"

ItemKind = Literal["context", "line"]
_KIND_ORDER: dict[ItemKind, int] = {"context": 0, "line": 1}


class PacketError(ValueError):
    pass


class ReaderKeyRow(BaseModel):
    id: str
    checks: list[str] = Field(min_length=2)

    @field_validator("checks")
    @classmethod
    def _added_claim_last(cls, checks: list[str]) -> list[str]:
        if checks[-1] != ADDED_CLAIM_CHECK or checks.count(ADDED_CLAIM_CHECK) != 1:
            raise ValueError(
                "the added-claim check must appear once, as the last check"
            )
        return checks


class PacketItem(BaseModel):
    item_id: str
    statement: str
    checks: list[str]
    kind: ItemKind
    representation: str


class KeyEntry(BaseModel):
    item_id: str
    label: str
    statement_id: str
    kind: ItemKind
    # context: position in entity_relations; line: first-encounter ordinal among
    # the lines read against statement_id.
    index: int = Field(ge=0)
    pinned_under: list[str] = Field(default_factory=list)
    weight: float | None = Field(default=None, ge=0.0, le=1.0)
    pin_finding: bool = False


class LineListing(BaseModel):
    pinned_under: str
    source: str
    line: str


class RelationListing(BaseModel):
    statement_id: str
    index: int = Field(ge=0)
    triple: str


class SourceListings(BaseModel):
    label: str
    fallback: list[LineListing] = Field(default_factory=list)
    empty_source: list[LineListing] = Field(default_factory=list)
    no_relation: list[str] = Field(default_factory=list)
    empty_context: list[RelationListing] = Field(default_factory=list)
    # Also read, once, against their source.
    pin_findings: list[LineListing] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "fallback": len(self.fallback),
            "empty_source": len(self.empty_source),
            "no_relation": len(self.no_relation),
            "empty_context": len(self.empty_context),
            "pin_findings": len(self.pin_findings),
        }


class Packet(BaseModel):
    items: list[PacketItem]
    key: list[KeyEntry]
    listings: list[SourceListings]


class SourceFile(BaseModel):
    label: str
    path: str
    sha256: str
    validity: Literal["VALID", "INVALID", "absent"]


class PacketMeta(BaseModel):
    seed: int
    sources: list[SourceFile]
    reader_key_sha256: str
    items_by_kind: dict[str, int]
    # Label -> listing class -> count.
    listing_counts: dict[str, dict[str, int]]


class _Validity(BaseModel):
    state: Literal["VALID", "INVALID"]


_LISTINGS = TypeAdapter(list[SourceListings])


_Draft = tuple[tuple[int, int, int, int], PacketItem, KeyEntry]


@dataclass(frozen=True)
class _Source:
    position: int
    label: str
    checks: dict[str, list[str]]
    # Statement id -> (corpus position, statement).
    statements: dict[str, tuple[int, StatementRepresentation]]
    listings: SourceListings

    def draft(self, entry: KeyEntry, representation: str) -> _Draft:
        number, statement = self.statements[entry.statement_id]
        item = PacketItem(
            item_id="",
            statement=statement.trace_content,
            checks=self.checks[entry.statement_id],
            kind=entry.kind,
            representation=representation,
        )
        return (
            (self.position, number, _KIND_ORDER[entry.kind], entry.index),
            item,
            entry,
        )


def parse_reader_key(text: str) -> list[ReaderKeyRow]:
    rows: list[ReaderKeyRow] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        try:
            rows.append(ReaderKeyRow.model_validate_json(line))
        except ValidationError as exc:
            raise PacketError(f"reader key line {number}: {line[:40]}: {exc}") from exc
    return rows


def _context_drafts(src: _Source, statement: StatementRepresentation) -> list[_Draft]:
    if not statement.entity_relations:
        src.listings.no_relation.append(statement.id)
    drafts: list[_Draft] = []
    for index, rel in enumerate(statement.entity_relations):
        if not rel.context:
            src.listings.empty_context.append(
                RelationListing(
                    statement_id=statement.id, index=index, triple=rel.triple
                )
            )
            continue
        entry = KeyEntry(
            item_id="",
            label=src.label,
            statement_id=statement.id,
            kind="context",
            index=index,
            weight=rel.weight,
        )
        drafts.append(src.draft(entry, rel.context))
    return drafts


def _read_lines(
    report: RepresentationsReport, listings: SourceListings
) -> dict[tuple[str, str], list[str]]:
    """(source, line) -> statements it was pinned under, in encounter order."""
    read: dict[tuple[str, str], list[str]] = {}
    for statement in report.statements:
        for pinned in statement.pinned_fact_lines:
            seen = LineListing(
                pinned_under=statement.id, source=pinned.source, line=pinned.line
            )
            if pinned.fallback:
                listings.fallback.append(seen)
                continue
            if not pinned.source:
                listings.empty_source.append(seen)
                continue
            if pinned.source != statement.id:
                listings.pin_findings.append(seen)
            under = read.setdefault((pinned.source, pinned.line), [])
            if statement.id not in under:
                under.append(statement.id)
    return read


def _line_weight(statement: StatementRepresentation, line: str) -> float | None:
    for rel in statement.entity_relations:
        if line.endswith(f"{rel.triple} -- {rel.context}"):
            return rel.weight
    return None


def _line_drafts(src: _Source, report: RepresentationsReport) -> list[_Draft]:
    ordinals: dict[str, int] = {}
    drafts: list[_Draft] = []
    for (source, line), under in _read_lines(report, src.listings).items():
        if source not in src.statements:
            raise PacketError(
                f"{src.label}: pinned line source {source} is no statement"
            )
        index = ordinals.get(source, 0)
        ordinals[source] = index + 1
        entry = KeyEntry(
            item_id="",
            label=src.label,
            statement_id=source,
            kind="line",
            index=index,
            pinned_under=under,
            weight=_line_weight(src.statements[source][1], line),
            pin_finding=any(s != source for s in under),
        )
        drafts.append(src.draft(entry, line))
    return drafts


def _source_drafts(
    position: int,
    label: str,
    report: RepresentationsReport,
    checks: dict[str, list[str]],
) -> tuple[list[_Draft], SourceListings]:
    missing = [s.id for s in report.statements if s.id not in checks]
    if missing:
        raise PacketError(f"{label}: no reader key row for {', '.join(missing)}")
    src = _Source(
        position=position,
        label=label,
        checks=checks,
        statements={s.id: (n, s) for n, s in enumerate(report.statements)},
        listings=SourceListings(label=label),
    )
    drafts: list[_Draft] = []
    for statement in report.statements:
        drafts += _context_drafts(src, statement)
    drafts += _line_drafts(src, report)
    return drafts, src.listings


def build_packet(
    sources: Sequence[tuple[str, RepresentationsReport]],
    rows: Sequence[ReaderKeyRow],
    *,
    seed: int,
) -> Packet:
    """Same sources, key and seed give the same packet."""
    checks = {row.id: row.checks for row in rows}
    drafts: list[_Draft] = []
    listings: list[SourceListings] = []
    for position, (label, report) in enumerate(sources):
        source_drafts, source_listings = _source_drafts(position, label, report, checks)
        drafts += source_drafts
        listings.append(source_listings)
    drafts.sort(key=lambda d: d[0])
    random.Random(seed).shuffle(drafts)  # noqa: S311 -- determinism, not crypto
    items: list[PacketItem] = []
    key: list[KeyEntry] = []
    for number, (_, item, entry) in enumerate(drafts, start=1):
        item_id = f"item-{number:04d}"
        items.append(item.model_copy(update={"item_id": item_id}))
        key.append(entry.model_copy(update={"item_id": item_id}))
    return Packet(items=items, key=key, listings=listings)


def _validity(path: Path) -> Literal["VALID", "INVALID", "absent"]:
    if not path.exists():
        return "absent"
    try:
        return _Validity.model_validate_json(path.read_bytes()).state
    except ValidationError as exc:
        raise PacketError(f"{path}: no VALID or INVALID state: {exc}") from exc


def read_source(label: str, path: Path) -> tuple[SourceFile, RepresentationsReport]:
    """Refuses a source whose sibling validity.json says INVALID."""
    validity = _validity(path.parent / "validity.json")
    if validity == "INVALID":
        raise PacketError(f"{label}: {path.parent / 'validity.json'} says INVALID")
    data = path.read_bytes()
    try:
        report = RepresentationsReport.model_validate_json(data)
    except ValidationError as exc:
        raise PacketError(f"{label}: {path} is not a representations report") from exc
    source = SourceFile(
        label=label,
        path=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
        validity=validity,
    )
    return source, report


def packet_meta(
    packet: Packet,
    *,
    seed: int,
    sources: Sequence[SourceFile],
    reader_key_sha256: str,
) -> PacketMeta:
    kinds = Counter(entry.kind for entry in packet.key)
    return PacketMeta(
        seed=seed,
        sources=list(sources),
        reader_key_sha256=reader_key_sha256,
        items_by_kind={kind: kinds[kind] for kind in _KIND_ORDER},
        listing_counts={s.label: s.counts() for s in packet.listings},
    )


def _jsonl(rows: Sequence[BaseModel]) -> str:
    return "".join(f"{row.model_dump_json()}\n" for row in rows)


def write_packet(packet: Packet, meta: PacketMeta, out_dir: Path) -> None:
    """packet.jsonl goes to readers; key, listings and meta stay with the tally."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "packet.jsonl").write_text(_jsonl(packet.items), encoding="utf-8")
    (out_dir / "key.jsonl").write_text(_jsonl(packet.key), encoding="utf-8")
    (out_dir / "listings.json").write_bytes(
        _LISTINGS.dump_json(packet.listings, indent=2)
    )
    (out_dir / "packet_meta.json").write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("wrote packet items=%d to %s", len(packet.items), out_dir)


def parse_packet_items(text: str) -> list[PacketItem]:
    lines = text.splitlines()
    return [PacketItem.model_validate_json(line) for line in lines if line.strip()]


def parse_key(text: str) -> list[KeyEntry]:
    lines = text.splitlines()
    return [KeyEntry.model_validate_json(line) for line in lines if line.strip()]


def parse_listings(data: bytes) -> list[SourceListings]:
    return _LISTINGS.validate_json(data)
