"""Stateful arm runner for P6 situational measurement."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from recollect.core import CognitiveMemory
from recollect.exceptions import MemorySDKError
from recollect.extraction import PatternExtractor
from recollect.models import RecallToken

from probe_cli.corpus import (
    EvalEntry,
    ExpectedAction,
    SeedGroup,
    SeedTrace,
    load_eval_corpus,
    load_seed_groups,
    load_seed_traces,
)

if TYPE_CHECKING:
    from recollect.config import MemoryConfig
    from recollect.llm.protocol import LLMProvider

    from probe_cli.arm import Arm

logger = logging.getLogger(__name__)


class EvalResult(BaseModel):
    entry_id: str
    expected_action: ExpectedAction
    expected_group_id: str | None = None
    expected_linked_trace_ids: list[str] = Field(default_factory=list)
    actual_action: ExpectedAction | None = None
    actual_group_id: str | None = None
    actual_linked_trace_ids: list[str] = Field(default_factory=list)
    actual_implication: str = ""
    actual_significance: float | None = None
    related_trace_ids: list[str] = Field(default_factory=list)
    candidate_token_ids: list[str] = Field(default_factory=list)
    success: bool = True
    error: str = ""
    latency_ms: float = 0.0


class SituationalRunReport(BaseModel):
    arm_name: str
    run_index: int
    model: str
    seed_groups_restored: int
    seed_traces_ingested: int
    eval_results: list[EvalResult] = Field(default_factory=list)


async def ingest_seed_traces(
    memory: CognitiveMemory,
    traces: list[SeedTrace],
    *,
    session_id: str,
    user_id: str,
) -> dict[str, str]:
    # Caller must override recall_tokens.enabled=false on the memory's config
    # before calling — seeded traces must not auto-create tokens via experience().
    id_map: dict[str, str] = {}
    for entry in traces:
        try:
            trace = await memory.experience(
                entry.text,
                session_id=session_id,
                user_id=user_id,
            )
        except MemorySDKError:
            logger.exception("seed ingest failed for %s", entry.id)
            continue
        id_map[entry.id] = trace.id
    return id_map


async def restore_seed_groups(
    memory: CognitiveMemory,
    groups: list[SeedGroup],
    seed_id_map: dict[str, str],
) -> dict[str, str]:
    # Returns group_id -> token_id map; harness uses it to score extend/revise targets.
    group_token_map: dict[str, str] = {}
    for g in groups:
        token = RecallToken(
            label=g.label,
            strength=g.strength,
            significance=g.significance,
            status=g.status,
        )
        await memory.storage.recall_tokens.create_token(token)
        member_trace_ids = [
            seed_id_map[m] for m in g.member_trace_ids if m in seed_id_map
        ]
        if not member_trace_ids:
            logger.warning(
                "group %s has no resolvable members; skipped stamping",
                g.group_id,
            )
            continue
        await memory.storage.recall_tokens.stamp_traces(token.id, member_trace_ids)
        group_token_map[g.group_id] = token.id
    return group_token_map


def _resolve_actual_links(
    actual_action: ExpectedAction,
    linked_indices: list[int],
    related_trace_ids: list[str],
    inverse_seed_map: dict[str, str],
) -> list[str]:
    if actual_action != "create":
        return []
    out: list[str] = []
    for i in linked_indices:
        if 1 <= i <= len(related_trace_ids):
            uuid = related_trace_ids[i - 1]
            out.append(inverse_seed_map.get(uuid, uuid))
    return out


def _resolve_actual_group(
    actual_action: ExpectedAction,
    group_number: int,
    candidate_token_ids: list[str],
    inverse_token_map: dict[str, str],
) -> str | None:
    if actual_action not in ("extend", "revise"):
        return None
    idx = group_number - 1
    if idx < 0 or idx >= len(candidate_token_ids):
        return None
    return inverse_token_map.get(candidate_token_ids[idx])


class SituationalArmRunner:
    def __init__(
        self,
        *,
        arm_name: str,
        provider: LLMProvider,
        config: MemoryConfig,
        seed_traces_path: str,
        seed_groups_path: str,
        eval_corpus_path: str,
        db_url: str = "",
        runs: int = 1,
    ) -> None:
        self._arm_name = arm_name
        self._provider = provider
        self._config = config
        self._seed_traces_path = seed_traces_path
        self._seed_groups_path = seed_groups_path
        self._eval_corpus_path = eval_corpus_path
        self._db_url = db_url
        self._runs = runs
        self._extractor = PatternExtractor(provider, config=config)

    @classmethod
    def from_arm(cls, arm: Arm, provider: LLMProvider) -> SituationalArmRunner:
        s = arm.situational
        if not s.enabled:
            raise ValueError(
                "SituationalArmRunner requires arm.situational.enabled=True"
            )
        if not s.seed_traces_path or not s.seed_groups_path or not s.eval_corpus_path:
            raise ValueError(
                "situational arm requires seed_traces_path, seed_groups_path, "
                "eval_corpus_path"
            )
        config = arm.to_memory_config()
        # recall_tokens.enabled=false suppresses experience() side effects on both
        # seed and eval traces; assess_situational bypasses the gate by design.
        config._set("recall_tokens.enabled", False)
        return cls(
            arm_name=arm.name,
            provider=provider,
            config=config,
            seed_traces_path=s.seed_traces_path,
            seed_groups_path=s.seed_groups_path,
            eval_corpus_path=s.eval_corpus_path,
            db_url=s.db_url,
            runs=arm.runs,
        )

    @property
    def model(self) -> str:
        return getattr(self._provider, "model_name", "unknown")

    async def run_all(self) -> list[SituationalRunReport]:
        # Seed once, eval N times: re-seeding per run pollutes the DB with stale
        # group instances ⇒ find_groups_for_traces surfaces tokens absent from
        # the current run's inverse_token_map ⇒ actual_group_id resolves to None.
        seed_traces = load_seed_traces(self._seed_traces_path)
        seed_groups = load_seed_groups(self._seed_groups_path)
        eval_corpus = load_eval_corpus(self._eval_corpus_path)
        user_id = f"probe-{self._arm_name}"
        seed_session_id = f"probe-{self._arm_name}-seed-{int(time.time() * 1000)}"

        memory = CognitiveMemory(extractor=self._extractor, config=self._config)
        out: list[SituationalRunReport] = []
        try:
            await memory.connect(db_url=self._db_url or None)
            seed_id_map = await ingest_seed_traces(
                memory, seed_traces,
                session_id=seed_session_id, user_id=user_id,
            )
            group_token_map = await restore_seed_groups(
                memory, seed_groups, seed_id_map
            )
            inverse_token_map = {tid: gid for gid, tid in group_token_map.items()}
            inverse_seed_map = {tid: cid for cid, tid in seed_id_map.items()}

            for i in range(self._runs):
                logger.info(
                    "situational-arm=%s run=%d/%d",
                    self._arm_name, i + 1, self._runs,
                )
                eval_session_id = (
                    f"probe-{self._arm_name}-eval-r{i}-{int(time.time() * 1000)}"
                )
                results = await self._run_eval(
                    memory, eval_corpus,
                    inverse_token_map, inverse_seed_map,
                    session_id=eval_session_id, user_id=user_id,
                )
                out.append(SituationalRunReport(
                    arm_name=self._arm_name,
                    run_index=i,
                    model=self.model,
                    seed_groups_restored=len(group_token_map),
                    seed_traces_ingested=len(seed_id_map),
                    eval_results=results,
                ))
        finally:
            await memory.close()
        return out

    async def _run_eval(
        self,
        memory: CognitiveMemory,
        eval_corpus: list[EvalEntry],
        inverse_token_map: dict[str, str],
        inverse_seed_map: dict[str, str] | None = None,
        *,
        session_id: str,
        user_id: str,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        for entry in eval_corpus:
            results.append(
                await self._assess_one(
                    memory,
                    entry,
                    inverse_token_map,
                    inverse_seed_map or {},
                    session_id=session_id,
                    user_id=user_id,
                )
            )
        return results

    async def _assess_one(
        self,
        memory: CognitiveMemory,
        entry: EvalEntry,
        inverse_token_map: dict[str, str],
        inverse_seed_map: dict[str, str] | None = None,
        *,
        session_id: str,
        user_id: str,
    ) -> EvalResult:
        inverse_seed_map = inverse_seed_map or {}
        start = time.perf_counter()
        try:
            trace = await memory.experience(
                entry.text,
                session_id=session_id,
                user_id=user_id,
            )
        except MemorySDKError as exc:
            return EvalResult(
                entry_id=entry.id,
                expected_action=entry.expected_action,
                expected_group_id=entry.expected_group_id,
                success=False,
                error=f"ingest: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        try:
            outcome = await memory.assess_situational(trace)
        except MemorySDKError as exc:
            await self._cleanup_eval_trace(memory, trace.id)
            return EvalResult(
                entry_id=entry.id,
                expected_action=entry.expected_action,
                expected_group_id=entry.expected_group_id,
                success=False,
                error=f"assess: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        await self._cleanup_eval_trace(memory, trace.id)
        if outcome is None:
            return EvalResult(
                entry_id=entry.id,
                expected_action=entry.expected_action,
                expected_group_id=entry.expected_group_id,
                actual_action="none",  # No related ⇒ implicit none
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        actual_group_id = _resolve_actual_group(
            outcome.assessment.action,
            outcome.assessment.group_number,
            outcome.candidate_token_ids,
            inverse_token_map,
        )
        actual_linked = _resolve_actual_links(
            outcome.assessment.action,
            outcome.assessment.linked_indices,
            outcome.related_trace_ids,
            inverse_seed_map,
        )
        return EvalResult(
            entry_id=entry.id,
            expected_action=entry.expected_action,
            expected_group_id=entry.expected_group_id,
            expected_linked_trace_ids=entry.expected_linked_trace_ids,
            actual_action=outcome.assessment.action,
            actual_group_id=actual_group_id,
            actual_linked_trace_ids=actual_linked,
            actual_implication=outcome.assessment.implication,
            actual_significance=outcome.assessment.significance,
            related_trace_ids=outcome.related_trace_ids,
            candidate_token_ids=outcome.candidate_token_ids,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    @staticmethod
    async def _cleanup_eval_trace(memory: CognitiveMemory, trace_id: str) -> None:
        # Hard erase, not forget(): forget archives, and archived eval
        # traces accumulate across seed-once-eval-N runs as stale-state
        # retrieval pollution. erase() also evicts the buffer slot.
        try:
            await memory.erase(trace_id)
        except MemorySDKError:
            logger.exception("cleanup erase failed for %s", trace_id)
