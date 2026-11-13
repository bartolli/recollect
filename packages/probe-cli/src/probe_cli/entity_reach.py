"""Entity reach: whether the trace-entity graph reaches each bridged answer trace.

Read-only over one seeded task-arm store. Each surface-form rule joins
entity name keys; reach under a rule is a breadth-first walk over the
bipartite graph of fixture trace ids and joined name keys.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import asyncpg
from pydantic import BaseModel, Field
from recollect.config import MemoryConfig
from recollect.config import config as sdk_config
from recollect.embeddings import FastEmbedProvider
from recollect.pool import ENTITY_NAME_KEY_SQL

from probe_cli.corpus import (
    PairLabel,
    TierLabel,
    load_entity_pair_labels,
    load_task_questions,
    load_task_seed_traces,
)

if TYPE_CHECKING:
    from probe_cli.arm import Arm
    from probe_cli.corpus import EntityPairLabel, TaskQuestion, TaskSeedTrace

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_TOKENS = frozenset(
    {"the", "a", "an", "of", "on", "at", "in", "for", "and", "s", "to"}
)
_MIN_ENTITY_CHARS = 3
_MAX_HOP = 6
_UNLABELED = "unlabeled"
_BRIDGED_PREFIX = "t3q"
_LABELS_FILE = "entity_pair_labels.jsonl"

_USERS_SQL = "SELECT DISTINCT user_id FROM memory_traces"
_TRACES_SQL = "SELECT id, COALESCE(content, '') AS content FROM memory_traces"
_MENTIONS_SQL = f"""
    SELECT trace_id, entity_name,
           {ENTITY_NAME_KEY_SQL.format("entity_name")} AS name_key
    FROM trace_entities
"""  # noqa: S608 -- the interpolated key expression is a module constant


class StorePreconditionError(ValueError):
    pass


class StoreTrace(BaseModel):
    id: str
    content: str


class StoreMention(BaseModel):
    trace_id: str
    entity_name: str
    name_key: str


class StoreRead(BaseModel):
    user_ids: list[str | None]
    traces: list[StoreTrace]
    mentions: list[StoreMention]


class EntityPair(BaseModel):
    a: str
    b: str
    similarity: float | None = Field(default=None, ge=-1.0, le=1.0)


class QuestionReach(BaseModel):
    id: str
    tier_label: TierLabel
    required: str
    seed_entities: list[str]
    hop: int | None = Field(default=None, ge=0, le=_MAX_HOP)
    candidates: int | None = Field(default=None, ge=1)


class RuleReach(BaseModel):
    rule: str
    pair_count: int = Field(ge=0)
    label_counts: dict[str, int]
    unlabeled: list[EntityPair]
    pairs: list[EntityPair]
    questions: list[QuestionReach]


class EntityReachReport(BaseModel):
    arm_name: str
    store_traces: int = Field(ge=0)
    store_user_id: str | None
    entities: int = Field(ge=0)
    single_trace_entities: int = Field(ge=0)
    thresholds: list[float]
    rules: list[RuleReach]
    # Fixture trace id -> name keys: re-analysis survives a store reset.
    mentions: dict[str, list[str]]


def _tokens(key: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(key)) - _STOP_TOKENS


def containment_pairs(keys: list[str]) -> list[EntityPair]:
    tokens = {key: _tokens(key) for key in keys}
    return [
        EntityPair(a=a, b=b)
        for a, b in combinations(sorted(tokens), 2)
        if tokens[a]
        and tokens[b]
        and (tokens[a] <= tokens[b] or tokens[b] <= tokens[a])
    ]


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(math.sumprod(vector, vector))
    return [v / norm for v in vector]


def cosine_pairs(
    keys: list[str], vectors: list[list[float]], *, threshold: float
) -> list[EntityPair]:
    units = dict(zip(keys, (_unit(v) for v in vectors), strict=True))
    pairs: list[EntityPair] = []
    for a, b in combinations(sorted(units), 2):
        similarity = max(-1.0, min(1.0, math.sumprod(units[a], units[b])))
        if similarity >= threshold:
            pairs.append(EntityPair(a=a, b=b, similarity=similarity))
    return pairs


def _root(parent: dict[str, str], key: str) -> str:
    while parent[key] != key:
        parent[key] = parent[parent[key]]
        key = parent[key]
    return key


def representatives(keys: list[str], pairs: list[EntityPair]) -> dict[str, str]:
    """Union-find over the joined names; each class is represented by its minimum."""
    parent = {key: key for key in keys}
    for pair in pairs:
        a, b = _root(parent, pair.a), _root(parent, pair.b)
        parent[max(a, b)] = min(a, b)
    return {key: _root(parent, key) for key in keys}


def question_entities(text: str, keys: list[str]) -> list[str]:
    lowered = text.lower()
    return sorted(
        key
        for key in set(keys)
        if len(key) >= _MIN_ENTITY_CHARS
        and re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered)
    )


def _first_hop(
    seeds: set[str], required: str, trace_reps: dict[str, set[str]]
) -> tuple[int | None, int | None]:
    seen = set(seeds)
    frontier = seeds
    reached: set[str] = set()
    for hop in range(_MAX_HOP + 1):
        added = {
            t for t, reps in trace_reps.items() if t not in reached and reps & frontier
        }
        reached |= added
        if required in reached:
            return hop, len(reached)
        # Traces mentioning an already-seen entity are already reached.
        frontier = {rep for t in added for rep in trace_reps[t]} - seen
        if not frontier:
            break
        seen |= frontier
    return None, None


def reach(
    question: TaskQuestion,
    *,
    mentions: dict[str, list[str]],
    representative: dict[str, str],
) -> QuestionReach:
    seeds = question_entities(question.question, list(representative))
    required = question.requires_trace_ids[0]
    trace_reps = {
        trace: {representative[key] for key in keys} for trace, keys in mentions.items()
    }
    hop, candidates = _first_hop(
        {representative[key] for key in seeds}, required, trace_reps
    )
    return QuestionReach(
        id=question.id,
        tier_label=question.tier_label,
        required=required,
        seed_entities=seeds,
        hop=hop,
        candidates=candidates,
    )


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def join_labels(
    pairs: list[EntityPair], labels: list[EntityPairLabel]
) -> tuple[dict[str, int], list[EntityPair]]:
    index = {_pair_key(row.a, row.b): row.label for row in labels}
    counts = dict.fromkeys((*get_args(PairLabel), _UNLABELED), 0)
    unlabeled: list[EntityPair] = []
    for pair in pairs:
        label = index.get(_pair_key(pair.a, pair.b), _UNLABELED)
        counts[label] += 1
        if label == _UNLABELED:
            unlabeled.append(pair)
    return counts, unlabeled


def require_one_user(user_ids: list[str | None]) -> str | None:
    if len(user_ids) > 1:
        raise StorePreconditionError(
            f"store holds traces of {len(user_ids)} users; entity-reach reads "
            "one fresh seeding per store"
        )
    return user_ids[0] if user_ids else None


def _fixture_ids(
    traces: list[StoreTrace], seeds: list[TaskSeedTrace]
) -> dict[str, str]:
    by_text = {seed.text.strip(): seed.id for seed in seeds}
    fixture_ids: dict[str, str] = {}
    for trace in traces:
        seed_id = by_text.get(trace.content.strip())
        if seed_id is None:
            raise StorePreconditionError(
                f"store trace {trace.id} matches no fixture seed text"
            )
        fixture_ids[trace.id] = seed_id
    return fixture_ids


def fixture_mentions(
    store: StoreRead, seeds: list[TaskSeedTrace]
) -> dict[str, list[str]]:
    fixture_ids = _fixture_ids(store.traces, seeds)
    keys: dict[str, set[str]] = {seed_id: set() for seed_id in fixture_ids.values()}
    for mention in store.mentions:
        keys[fixture_ids[mention.trace_id]].add(mention.name_key)
    return {seed_id: sorted(names) for seed_id, names in sorted(keys.items())}


def _trace_counts(mentions: dict[str, list[str]]) -> Counter[str]:
    """Name key -> number of fixture traces mentioning it."""
    return Counter(key for names in mentions.values() for key in names)


def rule_pairs(
    keys: list[str], vectors: list[list[float]], thresholds: list[float]
) -> dict[str, list[EntityPair]]:
    rules: dict[str, list[EntityPair]] = {
        "exact": [],
        "containment": containment_pairs(keys),
    }
    for threshold in thresholds:
        rules[f"embedding>={threshold:.2f}"] = cosine_pairs(
            keys, vectors, threshold=threshold
        )
    return rules


def rule_reach(
    rule: str,
    pairs: list[EntityPair],
    *,
    mentions: dict[str, list[str]],
    questions: list[TaskQuestion],
    labels: list[EntityPairLabel],
) -> RuleReach:
    representative = representatives(sorted(_trace_counts(mentions)), pairs)
    label_counts, unlabeled = join_labels(pairs, labels)
    return RuleReach(
        rule=rule,
        pair_count=len(pairs),
        label_counts=label_counts,
        unlabeled=unlabeled,
        pairs=pairs,
        questions=[
            reach(q, mentions=mentions, representative=representative)
            for q in questions
        ],
    )


async def read_store(db_url: str) -> StoreRead:
    conn = await asyncpg.connect(db_url)
    try:
        async with conn.transaction(readonly=True):
            users = await conn.fetch(_USERS_SQL)
            traces = await conn.fetch(_TRACES_SQL)
            mentions = await conn.fetch(_MENTIONS_SQL)
    finally:
        await conn.close()
    return StoreRead(
        user_ids=[row["user_id"] for row in users],
        traces=[StoreTrace.model_validate(dict(row)) for row in traces],
        mentions=[StoreMention.model_validate(dict(row)) for row in mentions],
    )


async def embed_names(keys: list[str]) -> list[list[float]]:
    provider = FastEmbedProvider.from_config(MemoryConfig())
    return await provider.generate_embeddings_batch(keys, task="clustering")


async def measure(arm: Arm, thresholds: list[float]) -> EntityReachReport:
    """Raises StorePreconditionError when the store is not one fixture seeding."""
    store = await read_store(arm.task.db_url or sdk_config.database_url)
    user_id = require_one_user(store.user_ids)
    mentions = fixture_mentions(store, load_task_seed_traces(arm.task.seed_traces_path))
    trace_counts = _trace_counts(mentions)
    keys = sorted(trace_counts)
    rules = rule_pairs(keys, await embed_names(keys), thresholds)
    questions = [
        q
        for q in load_task_questions(arm.task.questions_path)
        if q.id.startswith(_BRIDGED_PREFIX)
    ]
    labels = load_entity_pair_labels(
        Path(arm.task.questions_path).parent / _LABELS_FILE
    )
    return EntityReachReport(
        arm_name=arm.name,
        store_traces=len(store.traces),
        store_user_id=user_id,
        entities=len(keys),
        single_trace_entities=sum(1 for n in trace_counts.values() if n == 1),
        thresholds=thresholds,
        rules=[
            rule_reach(
                name, pairs, mentions=mentions, questions=questions, labels=labels
            )
            for name, pairs in rules.items()
        ],
        mentions=mentions,
    )
