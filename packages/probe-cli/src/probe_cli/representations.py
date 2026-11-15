"""Representations: every stored form of each corpus statement, and its pinned lines."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import asyncpg
from pydantic import BaseModel, Field, ValidationError
from recollect.config import config as sdk_config
from recollect.core import CognitiveMemory
from recollect.llm.types import ExtractionResult
from recollect.models import PersonaFact
from recollect.storage_context import create_storage_context
from recollect_mcp.server import _safety_net_line

from probe_cli.corpus import load_corpus
from probe_cli.fusion_screen import (
    _DROP_SCREEN,
    SCREEN_DB,
    ScreenPreconditionError,
    _copy_store,
    _maintenance,
    _with_database,
)
from probe_cli.weak_match_screen import _database_names, check_template

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from probe_cli.arm import Arm
    from probe_cli.corpus import CorpusEntry

logger = logging.getLogger(__name__)

_TRACES_SQL = """
    SELECT id, COALESCE(content, '') AS content, status,
           COALESCE(pattern, '{}'::jsonb)::text AS pattern
    FROM memory_traces
    WHERE ($1::text IS NULL OR user_id = $1)
"""
# entity_relations carries no user_id: the asserting trace's user scopes the row.
_RELATIONS_SQL = """
    SELECT r.source_trace_id, r.source_entity, r.relation, r.target_entity,
           COALESCE(r.context, '') AS context, r.weight
    FROM entity_relations r
    JOIN memory_traces t ON t.id = r.source_trace_id
    WHERE ($1::text IS NULL OR t.user_id = $1)
    ORDER BY r.created_at, r.id
"""
# pin's statement fallback predicate; the closed predicate enum lacks it, so no
# extraction relation can carry it.
_FALLBACK_PREDICATE = "noted"


class RelationRow(BaseModel):
    triple: str
    context: str
    weight: float = Field(ge=0.0, le=1.0)


class FactLine(BaseModel):
    line: str
    status: str
    category: str


class PinnedLine(BaseModel):
    line: str
    # Corpus id of the fact's source trace: pin can flip a twin another
    # statement stated. "" when that trace is not a corpus statement.
    source: str
    # pin's `user noted` statement fact: faithful by construction, not a reading
    # of the extraction.
    fallback: bool


class StatementRepresentation(BaseModel):
    id: str
    trace_content: str
    trace_status: str
    extraction: ExtractionResult
    # count_degraded's predicate on the extraction: no concepts.
    degraded: bool
    entity_relations: list[RelationRow]
    # Before any pin.
    persona_facts: list[FactLine]
    pinned_fact_lines: list[PinnedLine]


class RepresentationsReport(BaseModel):
    arm_name: str
    template_db: str
    user_id: str
    traces_path: str
    traces_sha256: str
    # Corpus order.
    degraded_statements: list[str]
    statements: list[StatementRepresentation]


class StoredTrace(BaseModel):
    id: str
    content: str
    status: str
    # The JSONB text: a raw asyncpg connection has no JSON codec.
    pattern: str


class StoredRelation(BaseModel):
    source_trace_id: str
    source_entity: str
    relation: str
    target_entity: str
    context: str
    weight: float = Field(ge=0.0, le=1.0)


class StoredState(BaseModel):
    traces: list[StoredTrace]
    relations: list[StoredRelation]
    facts: list[PersonaFact]


def match_statements(
    entries: Sequence[CorpusEntry], traces: Sequence[StoredTrace]
) -> dict[str, StoredTrace]:
    """Corpus id -> the one trace holding its exact text."""
    by_content: dict[str, list[StoredTrace]] = defaultdict(list)
    for trace in traces:
        by_content[trace.content].append(trace)
    matched: dict[str, StoredTrace] = {}
    for entry in entries:
        found = by_content.get(entry.text, [])
        if len(found) != 1:
            raise ScreenPreconditionError(
                f"{entry.id}: {len(found)} traces hold its text, expected 1"
            )
        matched[entry.id] = found[0]
    return matched


def _extraction(corpus_id: str, trace: StoredTrace) -> ExtractionResult:
    try:
        return ExtractionResult.model_validate_json(trace.pattern)
    except ValidationError as exc:
        raise ScreenPreconditionError(
            f"{corpus_id}: stored pattern is not an extraction"
        ) from exc


def _relation_rows(relations: Sequence[StoredRelation]) -> dict[str, list[RelationRow]]:
    by_trace: dict[str, list[RelationRow]] = defaultdict(list)
    for r in relations:
        by_trace[r.source_trace_id].append(
            RelationRow(
                triple=f"{r.source_entity} {r.relation} {r.target_entity}",
                context=r.context,
                weight=r.weight,
            )
        )
    return by_trace


def _fact_lines(facts: Sequence[PersonaFact]) -> dict[str, list[FactLine]]:
    by_trace: dict[str, list[FactLine]] = defaultdict(list)
    for f in facts:
        if f.source_trace_id is not None:
            by_trace[f.source_trace_id].append(
                FactLine(line=_safety_net_line(f), status=f.status, category=f.category)
            )
    return by_trace


def _pinned_lines(
    pinned: Sequence[PersonaFact], sources: dict[str, str]
) -> list[PinnedLine]:
    return [
        PinnedLine(
            line=_safety_net_line(f),
            source=sources.get(f.source_trace_id or "", ""),
            fallback=f.predicate == _FALLBACK_PREDICATE,
        )
        for f in pinned
    ]


async def represent(
    entries: Sequence[CorpusEntry],
    state: StoredState,
    *,
    pin: Callable[[str], Awaitable[list[PersonaFact]]],
) -> list[StatementRepresentation]:
    """Every statement is matched and its pattern read before the first pin.

    Pins run in corpus order.
    """
    matched = match_statements(entries, state.traces)
    extractions = {cid: _extraction(cid, trace) for cid, trace in matched.items()}
    sources = {trace.id: cid for cid, trace in matched.items()}
    relations = _relation_rows(state.relations)
    facts = _fact_lines(state.facts)
    statements: list[StatementRepresentation] = []
    for entry in entries:
        trace = matched[entry.id]
        pinned = await pin(trace.id)
        logger.info("representations statement=%s pinned=%d", entry.id, len(pinned))
        statements.append(
            StatementRepresentation(
                id=entry.id,
                trace_content=trace.content,
                trace_status=trace.status,
                extraction=extractions[entry.id],
                degraded=not extractions[entry.id].concepts,
                entity_relations=relations.get(trace.id, []),
                persona_facts=facts.get(trace.id, []),
                pinned_fact_lines=_pinned_lines(pinned, sources),
            )
        )
    return statements


async def _read_rows(
    screen_url: str, user_id: str
) -> tuple[list[StoredTrace], list[StoredRelation]]:
    """The user's traces and relations in one read; refuses all but the copy."""
    conn = await asyncpg.connect(screen_url)
    try:
        name = await conn.fetchval("SELECT current_database()")
        if name != SCREEN_DB:
            raise ScreenPreconditionError(f"connected to {name!r}, not {SCREEN_DB!r}")
        async with conn.transaction(readonly=True):
            traces = await conn.fetch(_TRACES_SQL, user_id)
            relations = await conn.fetch(_RELATIONS_SQL, user_id)
    finally:
        await conn.close()
    return (
        [StoredTrace.model_validate(dict(row.items())) for row in traces],
        [StoredRelation.model_validate(dict(row.items())) for row in relations],
    )


async def _represented(
    arm: Arm, entries: Sequence[CorpusEntry], *, screen_url: str, user_id: str
) -> list[StatementRepresentation]:
    traces, relations = await _read_rows(screen_url, user_id)
    storage = create_storage_context(screen_url)
    memory = CognitiveMemory(storage=storage, config=arm.to_memory_config())
    try:
        await memory.connect()
        # Read before the first pin: pin flips stored statuses.
        facts = await memory.facts(user_id=user_id, limit=None)
        state = StoredState(traces=traces, relations=relations, facts=facts)
        return await represent(entries, state, pin=memory.pin)
    finally:
        await memory.close()


async def read(
    arm: Arm, *, template: str, user_id: str, traces_path: Path
) -> RepresentationsReport:
    """Every read and pin runs on the store copy, dropped before return."""
    entries = load_corpus(traces_path).entries
    if not entries:
        raise ScreenPreconditionError(f"no statements in {traces_path}")
    digest = hashlib.sha256(traces_path.read_bytes()).hexdigest()
    base_url = arm.task.db_url or sdk_config.database_url
    check_template(template, await _database_names(base_url))
    try:
        await _copy_store(base_url, template)
        statements = await _represented(
            arm,
            entries,
            screen_url=_with_database(base_url, SCREEN_DB),
            user_id=user_id,
        )
    finally:
        await _maintenance(base_url, _DROP_SCREEN)
    return RepresentationsReport(
        arm_name=arm.name,
        template_db=template,
        user_id=user_id,
        traces_path=str(traces_path),
        traces_sha256=digest,
        degraded_statements=[s.id for s in statements if s.degraded],
        statements=statements,
    )
