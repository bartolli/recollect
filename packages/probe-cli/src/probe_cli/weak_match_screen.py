"""Weak-match screen: fixed rules label one captured think_about window per query."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from statistics import fmean, median, pstdev
from typing import TYPE_CHECKING, Literal

import asyncpg
from pydantic import BaseModel, ConfigDict, Field
from recollect.config import config as sdk_config
from recollect.core import _estimate_tokens
from recollect.storage_context import create_storage_context
from recollect.storage_utils import embedding_to_pgvector
from recollect_mcp.server import _format_thoughts

from probe_cli.corpus import (
    TierLabel,
    _load_jsonl,
    load_task_questions,
    load_task_seed_traces,
)
from probe_cli.fusion_screen import (
    _DROP_SCREEN,
    _TOKEN_BUDGET,
    SCREEN_DB,
    CallCapture,
    ScreenPreconditionError,
    WindowMismatch,
    _copy_store,
    _maintenance,
    _read_screen_traces,
    _ScreenMemory,
    _window_key,
    _with_database,
    seed_keys,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from pathlib import Path

    from recollect.config import MemoryConfig
    from recollect.models import Thought

    from probe_cli.arm import Arm
    from probe_cli.corpus import TaskQuestion, TaskSeedTrace

logger = logging.getLogger(__name__)

RuleName = Literal["current", "fixed-floor", "store-relative", "top-relative"]
CELLS: tuple[tuple[RuleName, float | None], ...] = (
    ("current", None),
    ("fixed-floor", 0.60),
    ("fixed-floor", 0.65),
    ("fixed-floor", 0.70),
    ("store-relative", 1.0),
    ("store-relative", 1.5),
    ("store-relative", 2.0),
    ("top-relative", 0.10),
    ("top-relative", 0.15),
    ("top-relative", 0.20),
)
# 1.0 - distance in Python: the exact base score search_semantic hands the fusion.
_SIMILARITY_SQL = """
    SELECT id, embedding <=> $1::vector AS distance
    FROM memory_traces
    WHERE embedding IS NOT NULL
      AND status <> 'forgotten'
      AND ($2::text IS NULL OR user_id = $2)
"""


class ScreenQuery(BaseModel):
    """One --queries line; expected ids are trace id prefixes."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    token_budget: int = Field(default=_TOKEN_BUDGET, ge=1)
    expected_strong: list[str] = Field(default_factory=list)
    expected_weak: list[str] = Field(default_factory=list)


class ExpectedErrors(BaseModel):
    strong_labeled_weak: list[str] = Field(default_factory=list)
    weak_left_strong: list[str] = Field(default_factory=list)
    not_delivered: list[str] = Field(default_factory=list)


class QueryCell(BaseModel):
    query_id: str
    tier_label: TierLabel | None = None
    delivered: int = Field(ge=0)
    weak: int = Field(ge=0)
    window: list[str]
    cost_current: int = Field(ge=1)
    cost_cut: int = Field(ge=1)
    expected: ExpectedErrors | None = None


class RequiredLost(BaseModel):
    # token: the lost trace carried a positive recall token bonus.
    token: list[str] = Field(default_factory=list)
    direct: list[str] = Field(default_factory=list)
    by_tier: dict[TierLabel, int] = Field(default_factory=dict)


class RuleCell(BaseModel):
    rule: RuleName
    parameter: float | None = None
    weak_total: int = Field(ge=0)
    required_lost: RequiredLost
    none_tier_strong: dict[str, int] = Field(default_factory=dict)
    none_tier_strong_total: int = Field(ge=0)
    cost_current_total: int = Field(ge=0)
    cost_cut_total: int = Field(ge=0)
    cost_current_mean: float = Field(ge=0.0)
    cost_cut_mean: float = Field(ge=0.0)
    queries: list[QueryCell]


class ScreenValidity(BaseModel):
    matched: int = Field(ge=0)
    checks: int = Field(ge=0)
    # The replay's lifecycle gate re-reads these with their captured status.
    archived_candidates: int = Field(ge=0)
    mismatches: list[WindowMismatch] = Field(default_factory=list)


class WeakMatchScreenReport(BaseModel):
    arm_name: str
    template_db: str
    user_id: str
    # None: the arm's task questions at its density tier.
    queries_path: str | None = None
    density_tier: int | None = Field(default=None, ge=0, le=2)
    store_traces: int = Field(ge=0)
    validity: ScreenValidity
    required_in_window: list[str]
    none_tier_delivered: dict[str, int]
    delivered_mean: float = Field(ge=0.0)
    cells: list[RuleCell]


@dataclass(frozen=True)
class QueryCapture:
    query: ScreenQuery
    window: list[Thought]
    token_bonuses: dict[str, float]
    # Store-relative population: every non-forgotten embedded trace of the user.
    similarities: dict[str, float]
    tier_label: TierLabel | None = None
    required: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _WeakPlan:
    arm: Arm
    config: MemoryConfig
    template: str
    user_id: str
    queries: list[ScreenQuery]
    queries_path: Path | None = None
    questions: dict[str, TaskQuestion] = field(default_factory=dict)
    seeds: list[TaskSeedTrace] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return self.arm.task.db_url or sdk_config.database_url

    @property
    def screen_url(self) -> str:
        return _with_database(self.base_url, SCREEN_DB)


@dataclass(frozen=True)
class _Pass:
    captures: list[QueryCapture]
    validity: ScreenValidity
    keys: dict[str, str]


def _is_fact(thought: Thought) -> bool:
    return bool(thought.trace.pattern.get("persona_fact"))


def _trace_thoughts(window: list[Thought]) -> list[Thought]:
    return [t for t in window if not _is_fact(t)]


def store_cut(similarities: Collection[float], k: float) -> float:
    values = list(similarities)
    return median(values) + k * pstdev(values)


def weak_traces(
    capture: QueryCapture, *, rule: RuleName, parameter: float | None
) -> frozenset[str]:
    """Trace ids the rule labels weak; persona-fact thoughts are never labeled."""
    traces = _trace_thoughts(capture.window)
    if rule == "current" or parameter is None or not traces:
        return frozenset()
    if rule == "fixed-floor":
        return frozenset(t.trace.id for t in traces if t.relevance < parameter)
    if rule == "top-relative":
        top = max(t.relevance for t in traces)
        return frozenset(t.trace.id for t in traces if top - t.relevance > parameter)
    cut = store_cut(capture.similarities.values(), parameter)
    sims = capture.similarities
    return frozenset(t.trace.id for t in traces if sims[t.trace.id] < cut)


def window_cost(thoughts: list[Thought]) -> int:
    """SDK token estimate of the text MCP recall prints for these thoughts."""
    return _estimate_tokens(_format_thoughts(thoughts, safety_facts=()))


def cut_window(window: list[Thought], weak: frozenset[str]) -> list[Thought]:
    return [t for t in window if t.trace.id not in weak]


def _slot(
    n: int, thought: Thought, *, keys: dict[str, str], weak: frozenset[str] | None
) -> str:
    tid = thought.trace.id
    if _is_fact(thought):
        return f"{n}. ~{tid[:8]} {thought.relevance:.3f}"
    line = f"{n}. {keys.get(tid, tid[:8])} {thought.relevance:.3f}"
    if weak is None:
        return line
    return f"{line} {'weak' if tid in weak else 'strong'}"


def window_lines(
    window: list[Thought],
    *,
    keys: dict[str, str],
    weak: frozenset[str] | None = None,
) -> list[str]:
    return [_slot(n, t, keys=keys, weak=weak) for n, t in enumerate(window, start=1)]


def _match(prefix: str, ids: list[str]) -> str | None:
    return next((tid for tid in ids if tid.startswith(prefix)), None)


def _expected_errors(
    query: ScreenQuery, *, traces: list[Thought], weak: frozenset[str]
) -> ExpectedErrors | None:
    prefixes = [*query.expected_strong, *query.expected_weak]
    if not prefixes:
        return None
    ids = [t.trace.id for t in traces]
    found = {p: _match(p, ids) for p in prefixes}
    labeled = {p: tid in weak for p, tid in found.items() if tid is not None}
    return ExpectedErrors(
        strong_labeled_weak=[p for p in query.expected_strong if labeled.get(p)],
        weak_left_strong=[p for p in query.expected_weak if labeled.get(p) is False],
        not_delivered=[p for p, tid in found.items() if tid is None],
    )


def _query_cell(
    capture: QueryCapture, *, weak: frozenset[str], keys: dict[str, str]
) -> QueryCell:
    traces = _trace_thoughts(capture.window)
    return QueryCell(
        query_id=capture.query.id,
        tier_label=capture.tier_label,
        delivered=len(traces),
        weak=len(weak),
        window=window_lines(capture.window, keys=keys, weak=weak),
        cost_current=window_cost(capture.window),
        cost_cut=window_cost(cut_window(capture.window, weak)),
        expected=_expected_errors(capture.query, traces=traces, weak=weak),
    )


def _delivered_ids(capture: QueryCapture) -> set[str]:
    return {t.trace.id for t in _trace_thoughts(capture.window)}


def required_in_window(captures: Sequence[QueryCapture]) -> list[str]:
    return [c.query.id for c in captures if c.required & _delivered_ids(c)]


def _route(capture: QueryCapture, weak: frozenset[str]) -> str | None:
    lost = capture.required & weak
    if not lost:
        return None
    bonuses = capture.token_bonuses
    return "token" if any(bonuses.get(tid, 0.0) > 0.0 for tid in lost) else "direct"


def required_lost(
    captures: Sequence[QueryCapture], weak_sets: Sequence[frozenset[str]]
) -> RequiredLost:
    routes = [(c, _route(c, w)) for c, w in zip(captures, weak_sets, strict=True)]
    lost = [(c, route) for c, route in routes if route is not None]
    return RequiredLost(
        token=[c.query.id for c, route in lost if route == "token"],
        direct=[c.query.id for c, route in lost if route == "direct"],
        by_tier=dict(Counter(c.tier_label for c, _ in lost if c.tier_label)),
    )


def rule_cell(
    captures: Sequence[QueryCapture],
    *,
    rule: RuleName,
    parameter: float | None,
    keys: dict[str, str],
) -> RuleCell:
    weak_sets = [weak_traces(c, rule=rule, parameter=parameter) for c in captures]
    pairs = zip(captures, weak_sets, strict=True)
    queries = [_query_cell(c, weak=w, keys=keys) for c, w in pairs]
    none_tier = {
        q.query_id: q.delivered - q.weak for q in queries if q.tier_label == "none"
    }
    return RuleCell(
        rule=rule,
        parameter=parameter,
        weak_total=sum(q.weak for q in queries),
        required_lost=required_lost(captures, weak_sets),
        none_tier_strong=none_tier,
        none_tier_strong_total=sum(none_tier.values()),
        cost_current_total=sum(q.cost_current for q in queries),
        cost_cut_total=sum(q.cost_cut for q in queries),
        cost_current_mean=fmean(q.cost_current for q in queries),
        cost_cut_mean=fmean(q.cost_cut for q in queries),
        queries=queries,
    )


def screen_cells(
    captures: Sequence[QueryCapture], *, keys: dict[str, str]
) -> list[RuleCell]:
    """Every cell from the one capture list; no retrieval runs here."""
    return [rule_cell(captures, rule=r, parameter=p, keys=keys) for r, p in CELLS]


def check_template(template: str, existing: Collection[str]) -> None:
    # The closing cleanup drops the screen copy, which must never be the template.
    if template == SCREEN_DB:
        raise ScreenPreconditionError(f"template {template!r} is the screen copy")
    if template not in existing:
        raise ScreenPreconditionError(f"template {template!r} does not exist")


def load_screen_queries(path: Path | str) -> list[ScreenQuery]:
    entries, _ = _load_jsonl(path, ScreenQuery, "screen query")
    return entries


def _plan_queries(
    arm: Arm, queries_path: Path | None
) -> tuple[list[ScreenQuery], list[TaskQuestion]]:
    if queries_path is not None:
        return load_screen_queries(queries_path), []
    tier = arm.task.density_tier
    questions = [
        q
        for q in load_task_questions(arm.task.questions_path)
        if tier in q.density_tiers
    ]
    return [ScreenQuery(id=q.id, query=q.question) for q in questions], questions


def _plan(
    arm: Arm, *, template: str, user_id: str, queries_path: Path | None
) -> _WeakPlan:
    queries, questions = _plan_queries(arm, queries_path)
    if not queries:
        raise ScreenPreconditionError("no queries to screen")
    config = arm.to_memory_config()
    config._set("recall_tokens.enabled", arm.task.with_tokens)
    seeds = load_task_seed_traces(arm.task.seed_traces_path) if questions else []
    return _WeakPlan(
        arm=arm,
        config=config,
        template=template,
        user_id=user_id,
        queries=queries,
        queries_path=queries_path,
        questions={q.id: q for q in questions},
        seeds=seeds,
    )


async def _database_names(base_url: str) -> set[str]:
    conn = await asyncpg.connect(_with_database(base_url, "postgres"))
    try:
        rows = await conn.fetch("SELECT datname FROM pg_database")
    finally:
        await conn.close()
    return {str(row["datname"]) for row in rows}


async def _content_similarities(
    memory: _ScreenMemory, query: str, *, user_id: str
) -> dict[str, float]:
    embedding = await memory._embeddings.generate_embedding(query, task="search_query")
    pool = await memory.storage.pool.get_pool()
    async with pool.acquire() as conn, conn.transaction(readonly=True):
        rows = await conn.fetch(
            _SIMILARITY_SQL, embedding_to_pgvector(embedding), user_id
        )
    if not rows:
        raise ScreenPreconditionError(f"no embedded traces for user {user_id!r}")
    return {str(row["id"]): 1.0 - float(row["distance"]) for row in rows}


async def _capture(
    memory: _ScreenMemory, query: ScreenQuery, *, user_id: str
) -> tuple[CallCapture, dict[str, float]]:
    similarities = await _content_similarities(memory, query.query, user_id=user_id)
    call = await memory.capture_call(
        query.query, label=query.id, token_budget=query.token_budget, user_id=user_id
    )
    unread = [
        t.trace.id[:8]
        for t in _trace_thoughts(call.window)
        if t.trace.id not in similarities
    ]
    if unread:
        raise ScreenPreconditionError(
            f"{query.id}: window traces without a content similarity: {unread}"
        )
    return call, similarities


def _query_capture(
    query: ScreenQuery,
    *,
    call: CallCapture,
    similarities: dict[str, float],
    question: TaskQuestion | None,
    by_key: dict[str, str],
) -> QueryCapture:
    tokens = call.inputs.token_bonuses or {}
    if question is None:
        return QueryCapture(query, call.window, tokens, similarities)
    required = frozenset(
        by_key[key] for key in question.requires_trace_ids if key in by_key
    )
    return QueryCapture(
        query, call.window, tokens, similarities, question.tier_label, required
    )


async def _validity(
    memory: _ScreenMemory,
    pairs: list[tuple[ScreenQuery, CallCapture]],
    *,
    keys: dict[str, str],
) -> ScreenValidity:
    mismatches: list[WindowMismatch] = []
    for query, call in pairs:
        replay = await memory.replay(
            call, form="prod", weight=None, token_budget=query.token_budget
        )
        if _window_key(replay.window) != _window_key(call.window):
            mismatches.append(
                WindowMismatch(
                    question_id=query.id,
                    think_about=window_lines(call.window, keys=keys),
                    offline=window_lines(replay.window, keys=keys),
                )
            )
    archived = {
        tid
        for _, call in pairs
        for tid, trace in call.inputs.traces_by_id.items()
        if trace.status == "archived"
    }
    return ScreenValidity(
        matched=len(pairs) - len(mismatches),
        checks=len(pairs),
        archived_candidates=len(archived),
        mismatches=mismatches,
    )


async def _capture_pass(plan: _WeakPlan) -> _Pass:
    contents = await _read_screen_traces(plan.screen_url, plan.user_id)
    keys = seed_keys(contents, plan.seeds) if plan.questions else {}
    by_key = {key: tid for tid, key in keys.items()}
    storage = create_storage_context(plan.screen_url)
    memory = _ScreenMemory(storage=storage, config=plan.config)
    pairs: list[tuple[ScreenQuery, CallCapture]] = []
    captures: list[QueryCapture] = []
    try:
        await memory.connect()
        logger.info("weak-match-screen queries=%d", len(plan.queries))
        for query in plan.queries:
            call, sims = await _capture(memory, query, user_id=plan.user_id)
            pairs.append((query, call))
            captures.append(
                _query_capture(
                    query,
                    call=call,
                    similarities=sims,
                    question=plan.questions.get(query.id),
                    by_key=by_key,
                )
            )
        validity = await _validity(memory, pairs, keys=keys)
    finally:
        await memory.close()
    return _Pass(captures, validity, keys)


def _report(plan: _WeakPlan, result: _Pass) -> WeakMatchScreenReport:
    captures = result.captures
    delivered = {c.query.id: len(_trace_thoughts(c.window)) for c in captures}
    return WeakMatchScreenReport(
        arm_name=plan.arm.name,
        template_db=plan.template,
        user_id=plan.user_id,
        queries_path=None if plan.queries_path is None else str(plan.queries_path),
        density_tier=plan.arm.task.density_tier if plan.questions else None,
        store_traces=len(captures[0].similarities),
        validity=result.validity,
        required_in_window=required_in_window(captures),
        none_tier_delivered={
            c.query.id: delivered[c.query.id]
            for c in captures
            if c.tier_label == "none"
        },
        delivered_mean=fmean(delivered.values()),
        cells=screen_cells(captures, keys=result.keys),
    )


async def _captured(plan: _WeakPlan) -> _Pass:
    check_template(plan.template, await _database_names(plan.base_url))
    try:
        await _copy_store(plan.base_url, plan.template)
        return await _capture_pass(plan)
    finally:
        await _maintenance(plan.base_url, _DROP_SCREEN)


async def screen(
    arm: Arm, *, template: str, user_id: str, queries_path: Path | None
) -> WeakMatchScreenReport:
    """Raises ScreenPreconditionError before any copy on a missing or self template."""
    plan = _plan(arm, template=template, user_id=user_id, queries_path=queries_path)
    return _report(plan, await _captured(plan))
