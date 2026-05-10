"""Arm runners: extraction-only and retrieval variants."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from recollect.core import CognitiveMemory
from recollect.exceptions import ExtractionError, MemorySDKError
from recollect.extraction import PatternExtractor

from probe_cli.arm import Arm
from probe_cli.corpus import Corpus, QueryCorpus
from probe_cli.metrics import (
    FOREIGN_PREFIX,
    EntryResult,
    QueryResult,
    RetrievalRunReport,
    RunReport,
)

if TYPE_CHECKING:
    from recollect.llm.protocol import LLMProvider
    from recollect.models import Thought

logger = logging.getLogger(__name__)


def resolve_thoughts(
    thoughts: list[Thought], inverse_id_map: dict[str, str]
) -> tuple[list[str], int]:
    # Persona-fact Thoughts carry a synthetic UUID with no link to source trace
    # (core.py:_persona_facts_to_thoughts) — filter from rank set, count separately.
    ranked: list[str] = []
    persona_facts = 0
    for t in thoughts:
        if t.trace.pattern.get("persona_fact"):
            persona_facts += 1
            continue
        ranked.append(inverse_id_map.get(t.trace.id, f"{FOREIGN_PREFIX}{t.trace.id}"))
    return ranked, persona_facts


class ArmRunner:
    # Extraction-only: PatternExtractor.extract per entry. No DB, no retrieval.
    def __init__(self, arm: Arm, provider: LLMProvider) -> None:
        self._arm = arm
        self._provider = provider
        self._config = arm.to_memory_config()
        self._extractor = PatternExtractor(provider, config=self._config)

    @property
    def template_version(self) -> str:
        return self._extractor.template_version

    @property
    def model(self) -> str:
        return self._arm.extraction.pydantic_ai_model

    async def run_once(self, corpus: Corpus, run_index: int) -> RunReport:
        entries: list[EntryResult] = []
        for entry in corpus.entries:
            entries.append(await self._extract_one(entry.id, entry.text))
        return RunReport(
            arm_name=self._arm.name,
            run_index=run_index,
            model=self.model,
            prompt_version=self.template_version,
            entries=entries,
        )

    async def run_all(self, corpus: Corpus) -> list[RunReport]:
        reports: list[RunReport] = []
        for i in range(self._arm.runs):
            logger.info(
                "arm=%s run=%d/%d entries=%d",
                self._arm.name,
                i + 1,
                self._arm.runs,
                len(corpus),
            )
            reports.append(await self.run_once(corpus, run_index=i))
        return reports

    async def _extract_one(self, entry_id: str, text: str) -> EntryResult:
        start = time.perf_counter()
        try:
            extraction = await self._extractor.extract(text)
        except ExtractionError as exc:
            return EntryResult(
                entry_id=entry_id,
                success=False,
                error=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        return EntryResult(
            entry_id=entry_id,
            success=True,
            extraction=extraction,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class RetrievalArmRunner:
    # Per run: fresh CognitiveMemory ⇒ ingest via experience ⇒ query via think_about
    # ⇒ resolve trace UUIDs back to corpus_id. session_id partitions retrieval.
    def __init__(self, arm: Arm, provider: LLMProvider) -> None:
        if not arm.retrieval.enabled:
            raise ValueError("RetrievalArmRunner requires arm.retrieval.enabled=True")
        if not arm.retrieval.traces_corpus_path:
            raise ValueError("retrieval.traces_corpus_path is required")
        if not arm.retrieval.query_corpus_path:
            raise ValueError("retrieval.query_corpus_path is required")
        self._arm = arm
        self._provider = provider
        self._config = arm.to_memory_config()
        self._extractor = PatternExtractor(provider, config=self._config)
        self._user_id = f"probe-{arm.name}"

    @property
    def template_version(self) -> str:
        return self._extractor.template_version

    @property
    def model(self) -> str:
        return self._arm.extraction.pydantic_ai_model

    async def run_once(
        self,
        traces_corpus: Corpus,
        query_corpus: QueryCorpus,
        run_index: int,
    ) -> RetrievalRunReport:
        session_id = f"probe-{self._arm.name}-r{run_index}-{int(time.time() * 1000)}"
        memory = CognitiveMemory(extractor=self._extractor, config=self._config)
        # PoolManager reads global MemoryConfig — connect(db_url=...) is the only
        # path that rebuilds storage against the arm's URL.
        db_url = self._arm.retrieval.db_url or None
        try:
            await memory.connect(db_url=db_url)
            id_map = await self._ingest(memory, traces_corpus, session_id)
            ingest_failures = len(traces_corpus) - len(id_map)
            inverse = {trace_id: corpus_id for corpus_id, trace_id in id_map.items()}
            queries = await self._run_queries(
                memory, query_corpus, inverse, session_id
            )
        finally:
            await memory.close()

        return RetrievalRunReport(
            arm_name=self._arm.name,
            run_index=run_index,
            model=self.model,
            prompt_version=self.template_version,
            top_k=self._arm.retrieval.top_k,
            ingested=len(id_map),
            ingest_failures=ingest_failures,
            queries=queries,
        )

    async def run_all(
        self, traces_corpus: Corpus, query_corpus: QueryCorpus
    ) -> list[RetrievalRunReport]:
        reports: list[RetrievalRunReport] = []
        for i in range(self._arm.runs):
            logger.info(
                "retrieval-arm=%s run=%d/%d traces=%d queries=%d",
                self._arm.name,
                i + 1,
                self._arm.runs,
                len(traces_corpus),
                len(query_corpus),
            )
            reports.append(await self.run_once(traces_corpus, query_corpus, i))
        return reports

    async def _ingest(
        self,
        memory: CognitiveMemory,
        corpus: Corpus,
        session_id: str,
    ) -> dict[str, str]:
        id_map: dict[str, str] = {}
        for entry in corpus.entries:
            try:
                trace = await memory.experience(
                    entry.text,
                    session_id=session_id,
                    user_id=self._user_id,
                )
            except MemorySDKError:
                logger.exception("ingest failed for %s", entry.id)
                continue
            id_map[entry.id] = trace.id
        return id_map

    async def _run_queries(
        self,
        memory: CognitiveMemory,
        query_corpus: QueryCorpus,
        inverse_id_map: dict[str, str],
        session_id: str,
    ) -> list[QueryResult]:
        out: list[QueryResult] = []
        for q in query_corpus.entries:
            start = time.perf_counter()
            try:
                thoughts = await memory.think_about(
                    q.text,
                    token_budget=self._arm.retrieval.token_budget,
                    session_id=session_id,
                    user_id=self._user_id,
                )
            except MemorySDKError as exc:
                out.append(
                    QueryResult(
                        query_id=q.id,
                        success=False,
                        error=str(exc),
                        relevant_corpus_ids=q.relevant_trace_ids,
                        expected_category=q.expected_category,
                        is_distractor=not q.relevant_trace_ids,
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
                )
                continue
            ranked, persona_facts = resolve_thoughts(thoughts, inverse_id_map)
            out.append(
                QueryResult(
                    query_id=q.id,
                    success=True,
                    ranked_corpus_ids=ranked,
                    relevant_corpus_ids=q.relevant_trace_ids,
                    expected_category=q.expected_category,
                    is_distractor=not q.relevant_trace_ids,
                    persona_fact_count=persona_facts,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            )
        return out
