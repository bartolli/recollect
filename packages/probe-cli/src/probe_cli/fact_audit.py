"""Story-9 slice-1 fact-channel audit: t3 tail extraction-to-promotion reliability.

Seeds each t3 tail text N rounds against a scratch DB, fresh table state per
round, and classifies the write path per tail per seeding: promoted / gated /
never_extracted. Delivery criterion is the task-arm persona line -- the answer
alias must sit inside 'subject predicate object'; content-only carriage is a
gate stage (spo_lossy), it never reaches the answer prompt. Flood axis: every
fact row the tail-shaped logistics texts write, by status.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field
from recollect.core import CognitiveMemory
from recollect.exceptions import MemorySDKError
from recollect.extraction import PatternExtractor

from probe_cli.corpus import load_task_questions, load_task_seed_traces
from probe_cli.task_scoring import contains_alias

if TYPE_CHECKING:
    from recollect.config import MemoryConfig
    from recollect.llm.protocol import LLMProvider
    from recollect.models import PersonaFact

    from probe_cli.arm import Arm
    from probe_cli.corpus import TaskQuestion, TaskSeedTrace

logger = logging.getLogger(__name__)

# The audit TRUNCATEs between rounds; a mis-set env var must not empty a
# load-bearing DB, so the database name is an invariant, not a knob.
_SCRATCH_DB_NAME = "probe_fact_audit"

# All data tables; applied_migrations and embedding_contract are bootstrap
# state and survive the reset.
_DATA_TABLES = (
    "associations",
    "concept_embeddings",
    "entity_relations",
    "memory_traces",
    "persona_facts",
    "recall_tokens",
    "sessions",
    "token_stamps",
    "trace_concepts",
    "trace_entities",
)

Classification = Literal["promoted", "gated", "never_extracted"]


class TailOutcome(BaseModel):
    tail_id: str
    question_id: str
    classification: Classification
    # spo_lossy | candidate | confidence | fact_type | swallowed
    gate_stage: str = ""
    fact_type: str = ""
    relations_total: int = 0
    answer_relations: int = 0
    facts_written: int = 0
    detail: str = ""


class AuditRound(BaseModel):
    round_index: int
    seeded: int
    ingest_failures: int = 0
    relations_total: int = 0
    facts_total: int = 0
    facts_by_status: dict[str, int] = Field(default_factory=dict)
    outcomes: list[TailOutcome] = Field(default_factory=list)


class FactAuditReport(BaseModel):
    arm_name: str
    model: str
    rounds: int
    confidence_threshold: float
    per_tail: dict[str, dict[str, int]] = Field(default_factory=dict)
    round_records: list[AuditRound] = Field(default_factory=list)


def _spo(fact: PersonaFact) -> str:
    return f"{fact.subject} {fact.predicate} {fact.object}"


def _relation_text(rel: dict[str, Any]) -> str:
    return f"{rel.get('source', '')} {rel.get('target', '')} {rel.get('context', '')}"


def classify_tail(
    *,
    tail_id: str,
    question_id: str,
    pattern: dict[str, Any],
    facts: list[PersonaFact],
    aliases: list[str],
    confidence_threshold: float,
) -> TailOutcome:
    """Three-way split with gate stage, ordered by proximity to delivery."""
    relations = pattern.get("relations", [])
    fact_type = str(pattern.get("fact_type", ""))
    answer_rels = [r for r in relations if contains_alias(_relation_text(r), aliases)]
    classification, gate_stage, detail = _classify(
        facts=facts,
        aliases=aliases,
        answer_rels=answer_rels,
        fact_type=fact_type,
        confidence_threshold=confidence_threshold,
    )
    return TailOutcome(
        tail_id=tail_id,
        question_id=question_id,
        classification=classification,
        gate_stage=gate_stage,
        fact_type=fact_type,
        relations_total=len(relations),
        answer_relations=len(answer_rels),
        facts_written=len(facts),
        detail=detail,
    )


def _classify(
    *,
    facts: list[PersonaFact],
    aliases: list[str],
    answer_rels: list[dict[str, Any]],
    fact_type: str,
    confidence_threshold: float,
) -> tuple[Classification, str, str]:
    surfacing = [f for f in facts if f.status in ("promoted", "pinned")]
    spo_hit = next((f for f in surfacing if contains_alias(_spo(f), aliases)), None)
    if spo_hit:
        return "promoted", "", _spo(spo_hit)
    lossy = next((f for f in surfacing if contains_alias(f.content, aliases)), None)
    if lossy:
        return "gated", "spo_lossy", f"answer outside SPO: {_spo(lossy)}"
    candidate_hit = next(
        (
            f
            for f in facts
            if f.status == "candidate"
            and (contains_alias(_spo(f), aliases) or contains_alias(f.content, aliases))
        ),
        None,
    )
    if candidate_hit:
        return "gated", "candidate", _spo(candidate_hit)
    if answer_rels and fact_type != "semantic":
        return "gated", "fact_type", ""
    if answer_rels and all(
        float(r.get("confidence", 0.8)) < confidence_threshold for r in answer_rels
    ):
        return "gated", "confidence", ""
    if answer_rels:
        # Semantic, confident, no surviving row: dedup/supersede swallowed it.
        return "gated", "swallowed", ""
    return "never_extracted", "", ""


def aggregate_outcomes(rounds: list[AuditRound]) -> dict[str, dict[str, int]]:
    per_tail: dict[str, Counter[str]] = {}
    for rnd in rounds:
        for o in rnd.outcomes:
            per_tail.setdefault(o.tail_id, Counter())[o.classification] += 1
    return {tail: dict(counts) for tail, counts in sorted(per_tail.items())}


def _require_scratch_db(db_url: str) -> None:
    name = urlsplit(db_url).path.lstrip("/") if db_url else ""
    if name != _SCRATCH_DB_NAME:
        raise ValueError(
            f"fact audit truncates its DB between rounds; refusing database "
            f"{name!r} -- point task.db_url at {_SCRATCH_DB_NAME!r}"
        )


class FactAuditRunner:
    def __init__(
        self,
        *,
        arm_name: str,
        provider: LLMProvider,
        config: MemoryConfig,
        seed_traces_path: str,
        questions_path: str,
        db_url: str,
        rounds: int,
    ) -> None:
        self._arm_name = arm_name
        self._provider = provider
        self._config = config
        self._seed_traces_path = seed_traces_path
        self._questions_path = questions_path
        self._db_url = db_url
        self._rounds = rounds
        self._extractor = PatternExtractor(provider, config=config)

    @classmethod
    def from_arm(cls, arm: Arm, provider: LLMProvider) -> FactAuditRunner:
        t = arm.task
        if not t.enabled:
            raise ValueError("FactAuditRunner requires arm.task.enabled=True")
        if not t.seed_traces_path or not t.questions_path:
            raise ValueError("fact audit requires seed_traces_path, questions_path")
        config = arm.to_memory_config()
        # Fact writes are token-independent; skipping T2 assessment drops cost
        # without touching the measured axis.
        config._set("recall_tokens.enabled", False)
        return cls(
            arm_name=arm.name,
            provider=provider,
            config=config,
            seed_traces_path=t.seed_traces_path,
            questions_path=t.questions_path,
            db_url=t.db_url,
            rounds=arm.runs,
        )

    @property
    def model(self) -> str:
        return getattr(self._provider, "model_name", "unknown")

    def _load_tails(self) -> list[tuple[TaskSeedTrace, TaskQuestion]]:
        seeds = {s.id: s for s in load_task_seed_traces(self._seed_traces_path)}
        pairs: list[tuple[TaskSeedTrace, TaskQuestion]] = []
        for q in load_task_questions(self._questions_path):
            if q.tier_label != "t3":
                continue
            for tid in q.requires_trace_ids:
                if tid not in seeds:
                    raise ValueError(f"question {q.id} requires unknown seed {tid}")
                pairs.append((seeds[tid], q))
        return pairs

    async def run(self) -> FactAuditReport:
        _require_scratch_db(self._db_url)
        tails = self._load_tails()
        threshold = float(self._config.get("persona.confidence_threshold", 0.6))
        rounds: list[AuditRound] = []
        for i in range(self._rounds):
            logger.info(
                "fact-audit arm=%s round=%d/%d tails=%d",
                self._arm_name,
                i + 1,
                self._rounds,
                len(tails),
            )
            rounds.append(await self._run_round(tails, i, threshold))
        return FactAuditReport(
            arm_name=self._arm_name,
            model=self.model,
            rounds=self._rounds,
            confidence_threshold=threshold,
            per_tail=aggregate_outcomes(rounds),
            round_records=rounds,
        )

    async def _run_round(
        self,
        tails: list[tuple[TaskSeedTrace, TaskQuestion]],
        index: int,
        threshold: float,
    ) -> AuditRound:
        memory = CognitiveMemory(extractor=self._extractor, config=self._config)
        try:
            await memory.connect(db_url=self._db_url)
            await self._reset_tables(memory)
            record = AuditRound(round_index=index, seeded=0)
            user_id = f"probe-{self._arm_name}"
            session_id = f"probe-{self._arm_name}-r{index}-{int(time.time() * 1000)}"
            for seed, question in tails:
                await self._audit_one(
                    memory,
                    record,
                    seed,
                    question,
                    threshold,
                    session_id=session_id,
                    user_id=user_id,
                )
            all_facts = await memory.storage.facts.get_persona_facts(
                limit=500, user_id=user_id
            )
            record.facts_total = len(all_facts)
            record.facts_by_status = dict(Counter(f.status for f in all_facts))
            return record
        finally:
            await memory.close()

    async def _audit_one(
        self,
        memory: CognitiveMemory,
        record: AuditRound,
        seed: TaskSeedTrace,
        question: TaskQuestion,
        threshold: float,
        *,
        session_id: str,
        user_id: str,
    ) -> None:
        try:
            trace = await memory.experience(
                seed.text, session_id=session_id, user_id=user_id
            )
        except MemorySDKError:
            logger.exception("audit ingest failed for %s", seed.id)
            record.ingest_failures += 1
            return
        record.seeded += 1
        facts = await memory.storage.facts.get_facts_by_source_trace_id(trace.id)
        outcome = classify_tail(
            tail_id=seed.id,
            question_id=question.id,
            pattern=trace.pattern,
            facts=facts,
            aliases=question.answers,
            confidence_threshold=threshold,
        )
        record.relations_total += outcome.relations_total
        record.outcomes.append(outcome)

    async def _reset_tables(self, memory: CognitiveMemory) -> None:
        # Fresh table state per round: the extraction path's subject read is
        # unscoped, so cross-round SPO duplicates mention-count-promote
        # candidates and corrupt the per-seeding classification.
        pool = await memory.storage.pool.get_pool()
        await pool.execute(f"TRUNCATE {', '.join(_DATA_TABLES)} CASCADE")
