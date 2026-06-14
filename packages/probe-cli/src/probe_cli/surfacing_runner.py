"""Read-path persona-fact surfacing arm: per-query surfacing records.

Companion metrics in surfacing_metrics. The arm seeds the eval fixture,
force-promotes candidate facts, then probes _find_relevant_persona_facts
per query and labels each surfaced fact by its source trace.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from recollect.core import CognitiveMemory
from recollect.exceptions import MemorySDKError
from recollect.extraction import PatternExtractor

from probe_cli.corpus import (
    CorpusEntry,
    QueryEntry,
    load_corpus,
    load_query_corpus,
)

if TYPE_CHECKING:
    from recollect.config import MemoryConfig
    from recollect.llm.protocol import LLMProvider
    from recollect.models import PersonaFact

    from probe_cli.arm import Arm

logger = logging.getLogger(__name__)


class SurfacedFact(BaseModel):
    # score = blended S from _find_relevant_persona_facts (max(bi, 0.7*csim+0.3*bi)),
    # NOT _compute_fact_relevance -- the variable the recall floor gates on.
    # source_trace_activated: the fact's source_trace is recall-token-activated
    # for this query (situationally live); propagated_sim is its activation weight.
    fact_id: str
    source_trace_id: str | None = None
    # source_eval_id: the source trace's corpus id (via db_to_eval), so a
    # ground-truth-aware metric can match a surfaced fact to surface/forbid/
    # grouped sets authored in corpus ids.
    source_eval_id: str = ""
    score: float = 0.0
    relevant: bool = False
    source_trace_activated: bool = False
    source_trace_propagated_sim: float = 0.0
    # Recall safety-bypass categories surface below the floor regardless of
    # activation, so the situational-lift metric excludes them from the
    # bridge-recoverable cohort.
    category: str = ""


class QuerySurfacing(BaseModel):
    # is_distractor: query has no relevant_trace_ids -> every surfaced fact is noise.
    query_id: str
    phrasing_style: str = ""
    is_distractor: bool = False
    surfaced: list[SurfacedFact] = Field(default_factory=list)


class SurfacingRunReport(BaseModel):
    arm_name: str
    model: str
    seeded_traces: int
    promoted_facts: int
    queries: list[QuerySurfacing] = Field(default_factory=list)


def _label_relevant(
    source_trace_id: str | None,
    db_to_eval: dict[str, str],
    relevant_eval_ids: set[str],
) -> bool:
    if source_trace_id is None:
        return False
    eval_id = db_to_eval.get(source_trace_id)
    return eval_id is not None and eval_id in relevant_eval_ids


def _build_query_surfacing(
    query: QueryEntry,
    facts: list[PersonaFact],
    scores: dict[str, float],
    db_to_eval: dict[str, str],
    token_activated: dict[str, float] | None = None,
) -> QuerySurfacing:
    relevant_eval_ids = set(query.relevant_trace_ids)
    activated = token_activated or {}
    surfaced = [
        SurfacedFact(
            fact_id=f.id,
            source_trace_id=f.source_trace_id,
            source_eval_id=db_to_eval.get(f.source_trace_id or "", ""),
            score=scores.get(f.id, 0.0),
            relevant=_label_relevant(
                f.source_trace_id, db_to_eval, relevant_eval_ids
            ),
            source_trace_activated=f.source_trace_id in activated,
            source_trace_propagated_sim=activated.get(f.source_trace_id or "", 0.0),
            category=f.category,
        )
        for f in facts
    ]
    return QuerySurfacing(
        query_id=query.id,
        phrasing_style=query.phrasing_style,
        is_distractor=not query.relevant_trace_ids,
        surfaced=surfaced,
    )


class SurfacingArmRunner:
    def __init__(
        self,
        *,
        arm_name: str,
        provider: LLMProvider,
        config: MemoryConfig,
        traces_path: str,
        queries_path: str,
        db_url: str = "",
        user_id: str = "",
    ) -> None:
        self._arm_name = arm_name
        self._provider = provider
        self._config = config
        self._traces_path = traces_path
        self._queries_path = queries_path
        self._db_url = db_url
        self._user_id = user_id or f"probe-{arm_name}"
        self._extractor = PatternExtractor(provider, config=config)

    @classmethod
    def from_arm(cls, arm: Arm, provider: LLMProvider) -> SurfacingArmRunner:
        s = arm.surfacing
        if not s.enabled:
            raise ValueError("SurfacingArmRunner requires arm.surfacing.enabled=True")
        if not s.traces_corpus_path or not s.query_corpus_path:
            raise ValueError(
                "surfacing arm requires traces_corpus_path, query_corpus_path"
            )
        return cls(
            arm_name=arm.name,
            provider=provider,
            config=arm.to_memory_config(),
            traces_path=s.traces_corpus_path,
            queries_path=s.query_corpus_path,
            db_url=s.db_url,
        )

    @property
    def model(self) -> str:
        return getattr(self._provider, "model_name", "unknown")

    async def run(self) -> SurfacingRunReport:
        traces = load_corpus(self._traces_path).entries
        queries = load_query_corpus(self._queries_path).entries
        # user_id time-suffixed: with no per-run probe_eval reset (slice 2), the
        # _find_relevant_persona_facts user filter walls each run off from the last.
        user_id = f"{self._user_id}-{int(time.time() * 1000)}"
        memory = CognitiveMemory(extractor=self._extractor, config=self._config)
        try:
            await memory.connect(db_url=self._db_url or None)
            db_to_eval = await self._seed(memory, traces, user_id=user_id)
            promoted = await self._promote_candidates(memory, user_id=user_id)
            scored = [
                await self._score_query(memory, q, db_to_eval, user_id=user_id)
                for q in queries
            ]
            results = [qs for qs in scored if qs is not None]
        finally:
            await memory.close()
        return SurfacingRunReport(
            arm_name=self._arm_name,
            model=self.model,
            seeded_traces=len(db_to_eval),
            promoted_facts=promoted,
            queries=results,
        )

    async def _seed(
        self,
        memory: CognitiveMemory,
        traces: list[CorpusEntry],
        *,
        user_id: str,
    ) -> dict[str, str]:
        session_id = f"{user_id}-seed"
        db_to_eval: dict[str, str] = {}
        for t in traces:
            try:
                trace = await memory.experience(
                    t.text, session_id=session_id, user_id=user_id
                )
            except MemorySDKError:
                logger.exception("seed ingest failed for %s", t.id)
                continue
            db_to_eval[trace.id] = t.id
        return db_to_eval

    async def _promote_candidates(
        self, memory: CognitiveMemory, *, user_id: str
    ) -> int:
        # Force-promote so all candidate facts enter the surfacing pool; the
        # read-path gate's promoted/pinned bypass then exposes the full
        # distribution for the threshold sweep.
        facts = await memory.storage.facts.get_persona_facts(
            limit=10_000, user_id=user_id
        )
        promoted = 0
        for f in facts:
            if f.status != "candidate":
                continue
            if await memory.storage.facts.update_fact_status(f.id, "promoted"):
                promoted += 1
        return promoted

    async def _score_query(
        self,
        memory: CognitiveMemory,
        query: QueryEntry,
        db_to_eval: dict[str, str],
        *,
        user_id: str,
    ) -> QuerySurfacing | None:
        # Skip on failure rather than abort the run -- the seed cost is already
        # paid; one bad query should not discard the whole measurement.
        try:
            embedding = await memory._embeddings.generate_embedding(
                query.text, task="search_query"
            )
            token_activated = await self._token_activated_traces(
                memory, embedding, user_id=user_id
            )
            facts, scores = await memory._find_relevant_persona_facts(
                query.text, embedding, user_id=user_id
            )
        except MemorySDKError:
            logger.exception("score failed for query %s", query.id)
            return None
        return _build_query_surfacing(
            query, facts, scores, db_to_eval, token_activated
        )

    @staticmethod
    async def _token_activated_traces(
        memory: CognitiveMemory,
        embedding: list[float],
        *,
        user_id: str,
    ) -> dict[str, float]:
        # Seed token activation from the trace vector search exactly as
        # think_about does, then run the production _activate_recall_tokens.
        # Approximation: omits think_about's entity-matched seed supplement, so
        # the spike undercounts activation -- a positive signal is conservative.
        search_limit = int(memory._config.get("retrieval.search_limit", 10))
        storage_scored = await memory.storage.vectors.search_semantic(
            embedding, limit=search_limit, user_id=user_id
        )
        return await memory._activate_recall_tokens(
            embedding, storage_scored, user_id=user_id
        )
