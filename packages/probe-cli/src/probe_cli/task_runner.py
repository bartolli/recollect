"""Task-eval arm runner: context-injection answering over a seeded corpus.

Per question, the answering model runs twice -- with the memory context block
and without -- and both responses score against the closed-form answers. The
with/without pair on identical questions is the memory-attributable delta.
Seeding runs with recall tokens per arm config: T3 questions require the
write-time groups the chains produce.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from recollect.core import CognitiveMemory
from recollect.exceptions import MemorySDKError
from recollect.extraction import PatternExtractor
from recollect.llm.pydantic_ai import PydanticAIProvider
from recollect.llm.types import Message

from probe_cli.corpus import (
    AnswerType,
    TaskQuestion,
    TaskSeedTrace,
    TierLabel,
    load_task_questions,
    load_task_seed_traces,
)
from probe_cli.task_scoring import score_answer

if TYPE_CHECKING:
    from recollect.config import MemoryConfig
    from recollect.llm.protocol import LLMProvider
    from recollect.models import Thought

    from probe_cli.arm import Arm

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = (
    "You answer questions for a personal memory assistant. Answer with the "
    "shortest factual phrase, no explanation. If the provided context does "
    "not contain the answer, reply exactly: unknown."
)


class TaskQuestionResult(BaseModel):
    question_id: str
    tier_label: TierLabel
    answer_type: AnswerType
    correct_with: bool = False
    correct_without: bool = False
    response_with: str = ""
    response_without: str = ""
    context_thoughts: int = 0
    success: bool = True
    error: str = ""
    latency_ms: float = 0.0


class TaskRunReport(BaseModel):
    arm_name: str
    run_index: int
    model: str
    answer_model: str
    density_tier: int
    seeded_traces: int
    results: list[TaskQuestionResult] = Field(default_factory=list)


class ReachabilityCheck(BaseModel):
    question_id: str
    tier_label: TierLabel
    holds: bool
    detail: str = ""


class TaskVerifyReport(BaseModel):
    arm_name: str
    density_tier: int
    checks: list[ReachabilityCheck] = Field(default_factory=list)

    @property
    def drifted(self) -> list[ReachabilityCheck]:
        return [c for c in self.checks if not c.holds]


def _format_context(persona_lines: list[str], thoughts: list[Thought]) -> str:
    parts: list[str] = []
    if persona_lines:
        parts.append("PERSONA FACTS:")
        parts.extend(f"- {line}" for line in persona_lines)
    if thoughts:
        parts.append("MEMORY CONTEXT:")
        parts.extend(f"- {t.reconstruction}" for t in thoughts if t.reconstruction)
    return "\n".join(parts)


class TaskArmRunner:
    def __init__(
        self,
        *,
        arm_name: str,
        provider: LLMProvider,
        answer_provider: LLMProvider,
        config: MemoryConfig,
        seed_traces_path: str,
        questions_path: str,
        db_url: str = "",
        runs: int = 1,
        density_tier: int = 0,
        with_recall: bool = True,
        with_priming: bool = True,
    ) -> None:
        self._arm_name = arm_name
        self._provider = provider
        self._answer_provider = answer_provider
        self._config = config
        self._seed_traces_path = seed_traces_path
        self._questions_path = questions_path
        self._db_url = db_url
        self._runs = runs
        self._density_tier = density_tier
        self._with_recall = with_recall
        self._with_priming = with_priming
        self._extractor = PatternExtractor(provider, config=config)

    @classmethod
    def from_arm(cls, arm: Arm, provider: LLMProvider) -> TaskArmRunner:
        t = arm.task
        if not t.enabled:
            raise ValueError("TaskArmRunner requires arm.task.enabled=True")
        if not t.seed_traces_path or not t.questions_path:
            raise ValueError("task arm requires seed_traces_path, questions_path")
        if not t.answer_model:
            raise ValueError("task arm requires answer_model")
        config = arm.to_memory_config()
        config._set("recall_tokens.enabled", t.with_tokens)
        answer_provider = PydanticAIProvider(model=t.answer_model)
        return cls(
            arm_name=arm.name,
            provider=provider,
            answer_provider=answer_provider,
            config=config,
            seed_traces_path=t.seed_traces_path,
            questions_path=t.questions_path,
            db_url=t.db_url,
            runs=arm.runs,
            density_tier=t.density_tier,
            with_recall=t.with_recall,
            with_priming=t.with_priming,
        )

    @property
    def model(self) -> str:
        return getattr(self._provider, "model_name", "unknown")

    @property
    def answer_model(self) -> str:
        return getattr(self._answer_provider, "model_name", "unknown")

    def _load_fixture(self) -> tuple[list[TaskSeedTrace], list[TaskQuestion]]:
        seeds = [
            s
            for s in load_task_seed_traces(self._seed_traces_path)
            if s.density_tier <= self._density_tier
        ]
        questions = [
            q
            for q in load_task_questions(self._questions_path)
            if self._density_tier in q.density_tiers
        ]
        return seeds, questions

    async def _seed(
        self, memory: CognitiveMemory, seeds: list[TaskSeedTrace], *, user_id: str
    ) -> dict[str, str]:
        # Deterministic shuffle: block-ordered fixture seeding fabricates
        # temporal-association seams between adjacent chains (spreading
        # activation then bleeds across them); interleaved arrival is the
        # production shape. Fixed seed keeps runs reproducible.
        ordered = list(seeds)
        random.Random(1889).shuffle(ordered)  # noqa: S311 -- determinism, not crypto
        session_id = f"probe-{self._arm_name}-seed-{int(time.time() * 1000)}"
        id_map: dict[str, str] = {}
        for entry in ordered:
            try:
                trace = await memory.experience(
                    entry.text, session_id=session_id, user_id=user_id
                )
            except MemorySDKError:
                logger.exception("seed ingest failed for %s", entry.id)
                continue
            id_map[entry.id] = trace.id
        return id_map

    async def _persona_lines(
        self, memory: CognitiveMemory, *, user_id: str
    ) -> list[str]:
        facts = await memory.storage.facts.get_persona_facts(limit=100, user_id=user_id)
        return [
            f"{f.subject} {f.predicate} {f.object}"
            for f in facts
            if f.status in ("promoted", "pinned")
        ]

    async def _ask(self, question: str, context: str) -> str:
        content = (
            f"{context}\n\nQuestion: {question}"
            if context
            else (f"Question: {question}")
        )
        return await self._answer_provider.complete(
            [
                Message(role="system", content=_ANSWER_SYSTEM),
                Message(role="user", content=content),
            ],
            max_tokens=256,
        )

    async def _answer_one(
        self,
        memory: CognitiveMemory,
        question: TaskQuestion,
        persona_lines: list[str],
        *,
        user_id: str,
    ) -> TaskQuestionResult:
        start = time.perf_counter()
        thoughts: list[Thought] = []
        try:
            if self._with_recall:
                thoughts = await memory.think_about(question.question, user_id=user_id)
            context = _format_context(
                persona_lines if self._with_priming else [], thoughts
            )
            response_with = await self._ask(question.question, context)
            response_without = await self._ask(question.question, "")
        except MemorySDKError as exc:
            return TaskQuestionResult(
                question_id=question.id,
                tier_label=question.tier_label,
                answer_type=question.answer_type,
                success=False,
                error=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        return TaskQuestionResult(
            question_id=question.id,
            tier_label=question.tier_label,
            answer_type=question.answer_type,
            correct_with=score_answer(response_with, question),
            correct_without=score_answer(response_without, question),
            response_with=response_with,
            response_without=response_without,
            context_thoughts=len(thoughts),
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def run_all(self) -> list[TaskRunReport]:
        seeds, questions = self._load_fixture()
        user_id = f"probe-{self._arm_name}"
        memory = CognitiveMemory(extractor=self._extractor, config=self._config)
        out: list[TaskRunReport] = []
        try:
            await memory.connect(db_url=self._db_url or None)
            id_map = await self._seed(memory, seeds, user_id=user_id)
            persona_lines = (
                await self._persona_lines(memory, user_id=user_id)
                if self._with_priming
                else []
            )
            for i in range(self._runs):
                logger.info(
                    "task-arm=%s run=%d/%d density=%d",
                    self._arm_name,
                    i + 1,
                    self._runs,
                    self._density_tier,
                )
                results = [
                    await self._answer_one(memory, q, persona_lines, user_id=user_id)
                    for q in questions
                ]
                out.append(
                    TaskRunReport(
                        arm_name=self._arm_name,
                        run_index=i,
                        model=self.model,
                        answer_model=self.answer_model,
                        density_tier=self._density_tier,
                        seeded_traces=len(id_map),
                        results=results,
                    )
                )
        finally:
            await memory.close()
        return out

    async def verify_reachability(self) -> TaskVerifyReport:
        """Confirm each intended tier_label empirically; report drift.

        t2/any: required trace present in think_about results. t3: present
        with tokens on AND absent with tokens off. t1: absent from retrieval
        AND carried by a promoted/pinned fact. none: skipped by construction.
        Runs with tokens forced on -- labels are properties of the full stack.
        """
        seeds, questions = self._load_fixture()
        user_id = f"probe-{self._arm_name}"
        self._config._set("recall_tokens.enabled", True)
        memory = CognitiveMemory(extractor=self._extractor, config=self._config)
        checks: list[ReachabilityCheck] = []
        try:
            await memory.connect(db_url=self._db_url or None)
            id_map = await self._seed(memory, seeds, user_id=user_id)
            facts = await memory.storage.facts.get_persona_facts(
                limit=200, user_id=user_id
            )
            fact_sources = {
                f.source_trace_id
                for f in facts
                if f.status in ("promoted", "pinned") and f.source_trace_id
            }
            for q in questions:
                if q.tier_label == "none":
                    continue
                checks.append(
                    await self._check_one(
                        memory, q, id_map, fact_sources, user_id=user_id
                    )
                )
        finally:
            await memory.close()
        return TaskVerifyReport(
            arm_name=self._arm_name,
            density_tier=self._density_tier,
            checks=checks,
        )

    async def _check_one(
        self,
        memory: CognitiveMemory,
        question: TaskQuestion,
        id_map: dict[str, str],
        fact_sources: set[str],
        *,
        user_id: str,
    ) -> ReachabilityCheck:
        required = {id_map[t] for t in question.requires_trace_ids if t in id_map}
        if not required:
            return ReachabilityCheck(
                question_id=question.id,
                tier_label=question.tier_label,
                holds=False,
                detail="required traces failed to seed",
            )
        retrieved = await self._retrieved_ids(memory, question, user_id=user_id)
        present = bool(required & retrieved)
        if question.tier_label in ("t2", "any"):
            return ReachabilityCheck(
                question_id=question.id,
                tier_label=question.tier_label,
                holds=present,
                detail="" if present else "required trace not retrieved",
            )
        if question.tier_label == "t3":
            self._config._set("recall_tokens.enabled", False)
            try:
                without = await self._retrieved_ids(memory, question, user_id=user_id)
            finally:
                self._config._set("recall_tokens.enabled", True)
            token_only = present and not (required & without)
            detail = (
                ""
                if token_only
                else (
                    "not retrieved at all"
                    if not present
                    else "also reachable without tokens"
                )
            )
            return ReachabilityCheck(
                question_id=question.id,
                tier_label="t3",
                holds=token_only,
                detail=detail,
            )
        # t1: the embedding gap holds AND a promoted/pinned fact carries it.
        fact_backed = bool(required & fact_sources)
        gap_holds = not present
        detail = (
            ""
            if (gap_holds and fact_backed)
            else (
                "trace retrievable -- gap does not hold"
                if present
                else "no promoted/pinned fact from required trace"
            )
        )
        return ReachabilityCheck(
            question_id=question.id,
            tier_label="t1",
            holds=gap_holds and fact_backed,
            detail=detail,
        )

    async def _retrieved_ids(
        self, memory: CognitiveMemory, question: TaskQuestion, *, user_id: str
    ) -> set[str]:
        thoughts = await memory.think_about(question.question, user_id=user_id)
        return {t.trace.id for t in thoughts}
