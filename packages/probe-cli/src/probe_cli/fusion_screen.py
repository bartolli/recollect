"""Fusion screen: required traces in the think_about window under other concept fusions.

Runs on a template copy of one seeded task-arm store; the template is never
connected to. Every cell of a pool replays the fusion inputs captured from one
real think_about pass, so cells differ by the concept input alone.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse, urlunparse

import asyncpg
from pydantic import BaseModel, Field
from recollect.config import config as sdk_config
from recollect.core import CognitiveMemory, _cosine_similarity, _estimate_tokens
from recollect.models import MemoryTrace, Thought
from recollect.storage_context import StorageContext, create_storage_context

from probe_cli.corpus import TierLabel, load_task_questions, load_task_seed_traces
from probe_cli.store_snapshot import _quoted
from probe_cli.task_runner import TaskArmRunner, TaskVerifyReport

if TYPE_CHECKING:
    from collections.abc import Callable

    from recollect.config import MemoryConfig

    from probe_cli.arm import Arm
    from probe_cli.corpus import TaskQuestion, TaskSeedTrace

logger = logging.getLogger(__name__)

SCREEN_DB = "probe_screen"
PoolName = Literal["pool-prod", "pool-concept10"]
FusionForm = Literal["prod", "zscore", "quantile", "zscore-content", "quantile-content"]
_POOLS: tuple[PoolName, ...] = ("pool-prod", "pool-concept10")
_CELLS: tuple[tuple[FusionForm, float | None], ...] = (
    ("prod", None),
    ("zscore", 0.7),
    ("zscore", 0.5),
    ("zscore", 0.3),
    ("quantile", 0.7),
    ("quantile", 0.5),
    ("quantile", 0.3),
    ("zscore-content", 0.7),
    ("zscore-content", 0.5),
    ("zscore-content", 0.3),
    ("quantile-content", 0.7),
    ("quantile-content", 0.5),
    ("quantile-content", 0.3),
)
_CONTENT_SUFFIX = "-content"
# Lowest relevance shown as 1.000 at the window's 3 decimals.
_CLAMP_SHOWN = 0.9995
_FUSED_SIGNATURE = inspect.signature(CognitiveMemory._compute_fused_scores)
_TOKEN_BUDGET = int(
    inspect.signature(CognitiveMemory.think_about).parameters["token_budget"].default
)
_DROP_SCREEN = f"DROP DATABASE IF EXISTS {_quoted(SCREEN_DB)} WITH (FORCE)"
_USER_TRACES_SQL = """
    SELECT id, COALESCE(content, '') AS content FROM memory_traces
    WHERE ($1::text IS NULL OR user_id = $1)
"""


class ScreenPreconditionError(ValueError):
    pass


class WindowDiff(BaseModel):
    gained: list[str] = Field(default_factory=list)
    lost: list[str] = Field(default_factory=list)


class QuestionWindow(BaseModel):
    question_id: str
    tier_label: TierLabel
    required: list[str]
    in_window: bool
    # position: window shown; fused_rank: post-gate fused list (both 1-indexed);
    # relevance: the fused score at fused_rank, shown or not.
    position: int | None = Field(default=None, ge=1)
    fused_rank: int | None = Field(default=None, ge=1)
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    window: list[str]


class CellResult(BaseModel):
    pool: PoolName
    form: FusionForm
    weight: float = Field(ge=0.0, le=1.0)
    required_in_window: int = Field(ge=0)
    versus_prod: WindowDiff = Field(default_factory=WindowDiff)
    # Members whose rescaled concept is <= 0 replay as concept-less; their
    # entity term then vanishes (it is gated by concept_sim).
    concepts_dropped: int = Field(default=0, ge=0)
    entity_terms_dropped: int = Field(default=0, ge=0)
    questions_dropped: list[str] = Field(default_factory=list)
    slots_at_clamp: int = Field(default=0, ge=0)
    # Questions whose slot-1 occupant differs from pool-prod/prod.
    slot1_changed: list[str] = Field(default_factory=list)
    # Median over questions of std(base)/std(concept) on the member set.
    stretch: float | None = Field(default=None, ge=0.0)
    questions: list[QuestionWindow]


class WindowMismatch(BaseModel):
    question_id: str
    think_about: list[str]
    offline: list[str]


class PoolValidity(BaseModel):
    pool: PoolName
    matched: int = Field(ge=0)
    checks: int = Field(ge=0)
    concept_appended: int = Field(ge=0)
    mismatches: list[WindowMismatch] = Field(default_factory=list)


class ReferenceDiff(BaseModel):
    path: str
    reference_in_window: int = Field(ge=0)
    screen_in_window: int = Field(ge=0)
    diff: WindowDiff


class FusionScreenReport(BaseModel):
    arm_name: str
    template_db: str
    user_id: str
    store_traces: int = Field(ge=0)
    density_tier: int = Field(ge=0, le=2)
    token_budget: int = Field(ge=1)
    validity: list[PoolValidity]
    reference: ReferenceDiff
    cells: list[CellResult]


@dataclass(frozen=True)
class FusionInputs:
    """One production _compute_fused_scores call, keyed by parameter name."""

    traces_by_id: dict[str, MemoryTrace]
    scores_by_id: dict[str, float]
    activation_levels: dict[str, float]
    entity_sims: dict[str, float]
    spread_bonus: float
    entity_bonus: float
    significance_weight: float
    valence_weight: float
    concept_sims: dict[str, float] | None
    concept_weight: float
    token_bonuses: dict[str, float] | None
    propagation_blend: float
    witness_evidence: dict[str, float] | None
    witness_bound_margin: float


@dataclass(frozen=True)
class CellConcepts:
    concept_sims: dict[str, float]
    entity_sims: dict[str, float]
    dropped: list[str] = field(default_factory=list)
    entity_dropped: int = 0
    stretch: float | None = None


@dataclass(frozen=True)
class CallCapture:
    inputs: FusionInputs
    # The fact thoughts assembly saw, cut ones included: their ids are minted
    # per call, so only these objects can match the real window.
    facts: list[Thought]
    window: list[Thought]
    # Base is level * spread_bonus, not a content cosine.
    activation_only: frozenset[str]


@dataclass(frozen=True)
class QuestionCapture(CallCapture):
    question: TaskQuestion


@dataclass(frozen=True)
class _Replay:
    gated: list[tuple[MemoryTrace, float]]
    window: list[Thought]
    concepts: CellConcepts


@dataclass
class _Capture:
    fusions: list[FusionInputs] = field(default_factory=list)
    assemblies: list[list[Thought]] = field(default_factory=list)
    activation_only: list[frozenset[str]] = field(default_factory=list)


@dataclass(frozen=True)
class _ScreenPlan:
    arm: Arm
    config: MemoryConfig
    template: str
    questions: list[TaskQuestion]
    seeds: list[TaskSeedTrace]

    @property
    def user_id(self) -> str:
        return f"probe-{self.arm.name}"

    @property
    def base_url(self) -> str:
        return self.arm.task.db_url or sdk_config.database_url

    @property
    def screen_url(self) -> str:
        return _with_database(self.base_url, SCREEN_DB)


@dataclass(frozen=True)
class _PoolPass:
    validity: PoolValidity
    cells: list[CellResult]
    store_traces: int


# The SDK statics are reached through self; a staticmethod override has no
# instance, so the capture rides a context variable set per think_about call.
_CAPTURE: ContextVar[_Capture] = ContextVar("fusion_screen_capture")


def _members(concept_sims: dict[str, float]) -> list[str]:
    return [tid for tid, sim in concept_sims.items() if sim > 0.0]


def stretch(concept_sims: dict[str, float], base: dict[str, float]) -> float | None:
    """std(base)/std(concept) over members; None when concept has no spread."""
    members = _members(concept_sims)
    concepts = [concept_sims[tid] for tid in members]
    if len(members) < 2 or pstdev(concepts) == 0.0:
        return None
    return pstdev([base[tid] for tid in members]) / pstdev(concepts)


def zscore_rescale(
    concept_sims: dict[str, float], base: dict[str, float]
) -> dict[str, float]:
    """Members' concept sims moved onto their base mean and std."""
    scale = stretch(concept_sims, base)
    if scale is None:
        return dict(concept_sims)
    members = _members(concept_sims)
    concepts = [concept_sims[tid] for tid in members]
    concept_mean = fmean(concepts)
    base_mean = fmean(base[tid] for tid in members)
    out = dict(concept_sims)
    out.update(
        {tid: base_mean + (concept_sims[tid] - concept_mean) * scale for tid in members}
    )
    return out


def quantile_rescale(
    concept_sims: dict[str, float], base: dict[str, float]
) -> dict[str, float]:
    """Rank matching: the i-th highest concept takes the i-th highest member base."""
    members = _members(concept_sims)
    by_concept = sorted(members, key=concept_sims.__getitem__, reverse=True)
    bases = sorted((base[tid] for tid in members), reverse=True)
    out = dict(concept_sims)
    out.update(zip(by_concept, bases, strict=True))
    return out


_RESCALES: dict[
    str, Callable[[dict[str, float], dict[str, float]], dict[str, float]]
] = {"zscore": zscore_rescale, "quantile": quantile_rescale}


def member_sims(
    concept_sims: dict[str, float],
    *,
    form: FusionForm,
    activation_only: frozenset[str],
) -> dict[str, float]:
    """The form's member candidates: content forms leave out activation-only traces."""
    if not form.endswith(_CONTENT_SUFFIX):
        return dict(concept_sims)
    return {t: sim for t, sim in concept_sims.items() if t not in activation_only}


def rescale_members(
    concept_sims: dict[str, float],
    *,
    form: FusionForm,
    base: dict[str, float],
    activation_only: frozenset[str],
) -> dict[str, float]:
    """The form's rescale over its members; every other trace keeps its raw concept."""
    members = member_sims(concept_sims, form=form, activation_only=activation_only)
    out = dict(concept_sims)
    out.update(_RESCALES[form.removesuffix(_CONTENT_SUFFIX)](members, base))
    return out


def cell_concepts(
    concept_sims: dict[str, float],
    *,
    rescaled: dict[str, float],
    entity_sims: dict[str, float],
) -> CellConcepts:
    concepts = dict(concept_sims)
    entities = dict(entity_sims)
    dropped: list[str] = []
    for tid in _members(concept_sims):
        if rescaled[tid] <= 0.0:
            dropped.append(tid)
            del concepts[tid]
            continue
        concepts[tid] = rescaled[tid]
        if tid in entities:
            # Entity bonus is gated by concept_sim: scaling by c/c' keeps the
            # term numerically unchanged, so only the concept input varies.
            entities[tid] = entity_sims[tid] * concept_sims[tid] / rescaled[tid]
    return CellConcepts(
        concept_sims=concepts,
        entity_sims=entities,
        dropped=dropped,
        entity_dropped=sum(1 for tid in dropped if tid in entity_sims),
    )


def fuse_cell(
    inputs: FusionInputs,
    *,
    form: FusionForm,
    weight: float | None,
    activation_only: frozenset[str],
) -> tuple[list[tuple[MemoryTrace, float]], CellConcepts]:
    concepts = inputs.concept_sims or {}
    if form == "prod" or weight is None:
        replay = inputs
        cell = CellConcepts(concept_sims=concepts, entity_sims=inputs.entity_sims)
    else:
        base = inputs.scores_by_id
        rescaled = rescale_members(
            concepts, form=form, base=base, activation_only=activation_only
        )
        members = member_sims(concepts, form=form, activation_only=activation_only)
        cell = dataclasses.replace(
            cell_concepts(concepts, rescaled=rescaled, entity_sims=inputs.entity_sims),
            stretch=stretch(members, base),
        )
        replay = dataclasses.replace(
            inputs,
            concept_sims=cell.concept_sims,
            entity_sims=cell.entity_sims,
            concept_weight=weight,
        )
    return CognitiveMemory._compute_fused_scores(**vars(replay)), cell


def window_diff(reference: dict[str, bool], other: dict[str, bool]) -> WindowDiff:
    return WindowDiff(
        gained=[q for q, hit in other.items() if hit and not reference.get(q, False)],
        lost=[q for q, hit in reference.items() if hit and not other.get(q, False)],
    )


def in_window(questions: list[QuestionWindow]) -> dict[str, bool]:
    return {q.question_id: q.in_window for q in questions}


def _slot1(question: QuestionWindow) -> str | None:
    return question.window[0].split()[1] if question.window else None


def slot1_changes(
    reference: list[QuestionWindow], other: list[QuestionWindow]
) -> list[str]:
    # Compares window labels: a fact occupant's "~id8" is minted per call, so
    # one fact in slot 1 of both pools would read as a change.
    first = {q.question_id: _slot1(q) for q in reference}
    return [q.question_id for q in other if _slot1(q) != first.get(q.question_id)]


def seed_keys(contents: dict[str, str], seeds: list[TaskSeedTrace]) -> dict[str, str]:
    """Stored trace id -> fixture seed key, by exact content."""
    by_text = {seed.text: seed.id for seed in seeds}
    keys: dict[str, str] = {}
    for trace_id, content in contents.items():
        if content not in by_text:
            raise ScreenPreconditionError(
                f"store trace {trace_id} matches no fixture seed text"
            )
        keys[trace_id] = by_text[content]
    return keys


def question_window(
    question: TaskQuestion,
    *,
    gated: list[tuple[MemoryTrace, float]],
    window: list[Thought],
    id_map: dict[str, str],
) -> QuestionWindow:
    required = {id_map[key] for key in question.requires_trace_ids if key in id_map}
    ranked = [
        (rank, score)
        for rank, (trace, score) in enumerate(gated, start=1)
        if trace.id in required
    ]
    shown = [i for i, t in enumerate(window, start=1) if t.trace.id in required]
    return QuestionWindow(
        question_id=question.id,
        tier_label=question.tier_label,
        required=list(question.requires_trace_ids),
        in_window=bool(shown),
        position=shown[0] if shown else None,
        fused_rank=ranked[0][0] if ranked else None,
        relevance=ranked[0][1] if ranked else None,
        window=TaskArmRunner._format_window(window, id_map),
    )


def _trace_thought(trace: MemoryTrace, relevance: float) -> Thought:
    content = trace.content or str(trace.pattern)
    return Thought(
        trace=trace,
        relevance=relevance,
        token_count=_estimate_tokens(content),
        reconstructed=trace.content is None,
        reconstruction=content,
    )


def _window_key(thoughts: list[Thought]) -> list[tuple[str, str]]:
    return [(t.trace.id, f"{t.relevance:.3f}") for t in thoughts]


def _with_database(url: str, database: str) -> str:
    return urlunparse(urlparse(url)._replace(path=f"/{database}"))


def _content_cosine(query_embedding: list[float], trace: MemoryTrace) -> float:
    if trace.embedding is None:
        return 0.0
    return _cosine_similarity(query_embedding, trace.embedding)


class _ConceptChannel:
    """VectorIndex whose search_semantic also admits the top concept-MaxSim traces."""

    def __init__(self, storage: StorageContext, owner_ids: list[str]) -> None:
        self._storage = storage
        self._owner_ids = owner_ids
        self.appended = 0

    async def search_semantic(
        self,
        query_embedding: list[float],
        limit: int = 10,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[tuple[MemoryTrace, float]]:
        scored = await self._storage.vectors.search_semantic(
            query_embedding, limit, session_id=session_id, user_id=user_id
        )
        sims = await self._storage.concept_embeddings.get_max_sim_per_owner(
            query_embedding, owner_type="trace", owner_ids=self._owner_ids
        )
        present = {trace.id for trace, _ in scored}
        top = sorted(sims, key=sims.__getitem__, reverse=True)[:limit]
        fresh = [tid for tid in top if tid not in present]
        by_id = {t.id: t for t in await self._storage.traces.get_traces_bulk(fresh)}
        # Content cosine as score, in MaxSim order: think_about's entity append.
        appended = [
            (by_id[tid], _content_cosine(query_embedding, by_id[tid]))
            for tid in fresh
            if tid in by_id
        ]
        self.appended += len(appended)
        return scored + appended

    async def spread_activation(
        self, seed_id: str, max_depth: int = 2, *, user_id: str | None = None
    ) -> list[tuple[MemoryTrace, float]]:
        return await self._storage.vectors.spread_activation(
            seed_id, max_depth, user_id=user_id
        )


class _ScreenMemory(CognitiveMemory):
    @staticmethod
    def _compute_fused_scores(
        *args: Any, **kwargs: Any
    ) -> list[tuple[MemoryTrace, float]]:
        bound = _FUSED_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        _CAPTURE.get().fusions.append(FusionInputs(**bound.arguments))
        return CognitiveMemory._compute_fused_scores(*args, **kwargs)

    @staticmethod
    def _assemble_with_trace_guarantee(
        thoughts: list[Thought], max_total: int, min_traces: int
    ) -> list[Thought]:
        _CAPTURE.get().assemblies.append(list(thoughts))
        return CognitiveMemory._assemble_with_trace_guarantee(
            thoughts, max_total, min_traces
        )

    async def _merge_candidates(
        self,
        query_embedding: list[float],
        wm: list[tuple[MemoryTrace, float]],
        storage: list[tuple[MemoryTrace, float]],
        activated: list[tuple[MemoryTrace, float]],
        entity_matches: list[tuple[str, float]] | None = None,
        token_activated: dict[str, float] | None = None,
        witness_evidence: dict[str, float] | None = None,
        *,
        user_id: str | None = None,
    ) -> list[tuple[MemoryTrace, float]]:
        # Membership, not score equality: _collect_activation_levels keeps the
        # max level but bases the score on the first level seen.
        content = {t.id for t, _ in wm} | {t.id for t, _ in storage}
        activation_only = frozenset(t.id for t, _ in activated) - content
        _CAPTURE.get().activation_only.append(activation_only)
        return await super()._merge_candidates(
            query_embedding,
            wm,
            storage,
            activated,
            entity_matches,
            token_activated,
            witness_evidence,
            user_id=user_id,
        )

    async def capture_call(
        self,
        query: str,
        *,
        label: str,
        token_budget: int = _TOKEN_BUDGET,
        user_id: str,
    ) -> CallCapture:
        capture = _Capture()
        token = _CAPTURE.set(capture)
        try:
            window = await self.think_about(
                query, token_budget=token_budget, user_id=user_id
            )
        finally:
            _CAPTURE.reset(token)
        counts = [
            len(capture.fusions),
            len(capture.assemblies),
            len(capture.activation_only),
        ]
        if counts != [1, 1, 1]:
            raise ScreenPreconditionError(
                f"{label}: captured {counts} fusions, assemblies, merges; "
                "expected one each"
            )
        facts = [
            t for t in capture.assemblies[0] if t.trace.pattern.get("persona_fact")
        ]
        return CallCapture(
            capture.fusions[0], facts, window, capture.activation_only[0]
        )

    async def capture(self, question: TaskQuestion, *, user_id: str) -> QuestionCapture:
        call = await self.capture_call(
            question.question, label=question.id, user_id=user_id
        )
        return QuestionCapture(
            inputs=call.inputs,
            facts=call.facts,
            window=call.window,
            activation_only=call.activation_only,
            question=question,
        )

    async def replay(
        self,
        capture: CallCapture,
        *,
        form: FusionForm,
        weight: float | None,
        token_budget: int = _TOKEN_BUDGET,
    ) -> _Replay:
        """think_about's steps after fusion, without its boost writes."""
        fused, concepts = fuse_cell(
            capture.inputs,
            form=form,
            weight=weight,
            activation_only=capture.activation_only,
        )
        gated = await self._gate_retired_candidates(fused)
        selected = self._select_within_budget(gated, token_budget)
        thoughts = [_trace_thought(t, s) for t, s in selected] + capture.facts
        window = CognitiveMemory._assemble_with_trace_guarantee(
            thoughts,
            int(self._config.get("retrieval.max_retrievals")),
            int(self._config.get("retrieval.min_trace_slots")),
        )
        return _Replay(gated=gated, window=window, concepts=concepts)


async def _maintenance(base_url: str, *statements: str) -> None:
    # CREATE and DROP DATABASE run from outside the databases they touch.
    conn = await asyncpg.connect(_with_database(base_url, "postgres"))
    try:
        for statement in statements:
            await conn.execute(statement)
    finally:
        await conn.close()


async def _copy_store(base_url: str, template: str) -> None:
    await _maintenance(
        base_url,
        _DROP_SCREEN,
        f"CREATE DATABASE {_quoted(SCREEN_DB)} TEMPLATE {_quoted(template)}",
    )


async def _read_screen_traces(screen_url: str, user_id: str) -> dict[str, str]:
    """Trace id -> content of the user; refuses any database but the copy."""
    conn = await asyncpg.connect(screen_url)
    try:
        name = await conn.fetchval("SELECT current_database()")
        if name != SCREEN_DB:
            raise ScreenPreconditionError(f"connected to {name!r}, not {SCREEN_DB!r}")
        async with conn.transaction(readonly=True):
            rows = await conn.fetch(_USER_TRACES_SQL, user_id)
    finally:
        await conn.close()
    if not rows:
        raise ScreenPreconditionError(f"no traces for user {user_id!r}")
    return {str(row["id"]): str(row["content"]) for row in rows}


def _screen_memory(
    plan: _ScreenPlan, pool: PoolName, owner_ids: list[str]
) -> tuple[_ScreenMemory, _ConceptChannel | None]:
    # Storage is injected and connect() gets no URL: connect(db_url) would
    # build a fresh context and discard the concept channel.
    storage = create_storage_context(plan.screen_url)
    channel: _ConceptChannel | None = None
    if pool == "pool-concept10":
        channel = _ConceptChannel(storage, owner_ids)
        storage = dataclasses.replace(storage, vectors=channel)
    return _ScreenMemory(storage=storage, config=plan.config), channel


async def _cell(
    memory: _ScreenMemory,
    captures: list[QuestionCapture],
    *,
    pool: PoolName,
    cell: tuple[FusionForm, float | None],
    id_map: dict[str, str],
) -> tuple[CellResult, list[list[Thought]]]:
    form, weight = cell
    replays = [await memory.replay(c, form=form, weight=weight) for c in captures]
    pairs = list(zip(captures, replays, strict=True))
    questions = [
        question_window(c.question, gated=r.gated, window=r.window, id_map=id_map)
        for c, r in pairs
    ]
    stretches = [r.concepts.stretch for r in replays if r.concepts.stretch is not None]
    result = CellResult(
        pool=pool,
        form=form,
        weight=captures[0].inputs.concept_weight if weight is None else weight,
        required_in_window=sum(q.in_window for q in questions),
        concepts_dropped=sum(len(r.concepts.dropped) for r in replays),
        entity_terms_dropped=sum(r.concepts.entity_dropped for r in replays),
        questions_dropped=[c.question.id for c, r in pairs if r.concepts.dropped],
        slots_at_clamp=sum(
            1 for r in replays for t in r.window if t.relevance >= _CLAMP_SHOWN
        ),
        stretch=median(stretches) if stretches else None,
        questions=questions,
    )
    return result, [r.window for r in replays]


def _validity(
    pool: PoolName,
    captures: list[QuestionCapture],
    offline: list[list[Thought]],
    *,
    id_map: dict[str, str],
    appended: int,
) -> PoolValidity:
    mismatches = [
        WindowMismatch(
            question_id=c.question.id,
            think_about=TaskArmRunner._format_window(c.window, id_map),
            offline=TaskArmRunner._format_window(window, id_map),
        )
        for c, window in zip(captures, offline, strict=True)
        if _window_key(c.window) != _window_key(window)
    ]
    return PoolValidity(
        pool=pool,
        matched=len(captures) - len(mismatches),
        checks=len(captures),
        concept_appended=appended,
        mismatches=mismatches,
    )


async def _pool_pass(plan: _ScreenPlan, pool: PoolName) -> _PoolPass:
    await _copy_store(plan.base_url, plan.template)
    contents = await _read_screen_traces(plan.screen_url, plan.user_id)
    id_map = {key: tid for tid, key in seed_keys(contents, plan.seeds).items()}
    memory, channel = _screen_memory(plan, pool, list(contents))
    try:
        await memory.connect()
        logger.info("fusion-screen pool=%s questions=%d", pool, len(plan.questions))
        captures = [
            await memory.capture(q, user_id=plan.user_id) for q in plan.questions
        ]
        cells = [
            await _cell(memory, captures, pool=pool, cell=cell, id_map=id_map)
            for cell in _CELLS
        ]
    finally:
        await memory.close()
    if channel is not None and channel.appended == 0:
        raise ScreenPreconditionError(f"{pool}: the concept channel appended no trace")
    appended = channel.appended if channel is not None else 0
    validity = _validity(pool, captures, cells[0][1], id_map=id_map, appended=appended)
    return _PoolPass(validity, [c for c, _ in cells], len(contents))


def _reference_diff(plan: _ScreenPlan, screen: dict[str, bool]) -> ReferenceDiff:
    """The seeding's own reachability windows against the screen's pool-prod/prod."""
    path = Path(plan.arm.output.dir) / plan.arm.name / "task_reachability.json"
    report = TaskVerifyReport.model_validate_json(path.read_text(encoding="utf-8"))
    required = {q.id: set(q.requires_trace_ids) for q in plan.questions}
    reference = {
        c.question_id: any(
            line.split()[1] in required[c.question_id] for line in c.window
        )
        for c in report.checks
    }
    return ReferenceDiff(
        path=str(path),
        reference_in_window=sum(reference.values()),
        screen_in_window=sum(screen.values()),
        diff=window_diff(reference, screen),
    )


def _plan(arm: Arm, template: str) -> _ScreenPlan:
    # Checked before any statement: every pass and the closing cleanup drop
    # the screen copy, which must never be the template.
    if template == SCREEN_DB:
        raise ScreenPreconditionError(f"template {template!r} is the screen copy")
    config = arm.to_memory_config()
    # verify_reachability forces the token hop on; the windows compared were taken so.
    config._set("recall_tokens.enabled", True)
    tier = arm.task.density_tier
    questions = [
        q
        for q in load_task_questions(arm.task.questions_path)
        if tier in q.density_tiers and q.tier_label != "none"
    ]
    return _ScreenPlan(
        arm=arm,
        config=config,
        template=template,
        questions=questions,
        seeds=load_task_seed_traces(arm.task.seed_traces_path),
    )


async def screen(arm: Arm, *, template: str) -> FusionScreenReport:
    """Raises ScreenPreconditionError when the copy is not one mappable seeding."""
    plan = _plan(arm, template)
    passes: list[_PoolPass] = []
    try:
        for pool in _POOLS:
            passes.append(await _pool_pass(plan, pool))
    finally:
        await _maintenance(plan.base_url, _DROP_SCREEN)
    cells = [cell for p in passes for cell in p.cells]
    prod_questions = cells[0].questions
    prod = in_window(prod_questions)
    return FusionScreenReport(
        arm_name=arm.name,
        template_db=template,
        user_id=plan.user_id,
        store_traces=passes[0].store_traces,
        density_tier=arm.task.density_tier,
        token_budget=_TOKEN_BUDGET,
        validity=[p.validity for p in passes],
        reference=_reference_diff(plan, prod),
        cells=[
            c.model_copy(
                update={
                    "versus_prod": window_diff(prod, in_window(c.questions)),
                    "slot1_changed": slot1_changes(prod_questions, c.questions),
                }
            )
            for c in cells
        ],
    )
