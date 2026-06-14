"""Surfacing arm over the token-rich situational substrate (slice-1b).

Slice 1 was a substrate null: probe_eval stamps zero recall tokens, so
source_trace activation was empty and ruling 5 went unmeasured. This runner
seeds the situational fixture's seed traces clean (no organic tokens), restores
the canonical token groups (deterministic stamping), and reports persona-fact
coverage on grouped traces -- the substrate gate that must pass before the
bridge-query measurement is worth authoring.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from recollect.core import CognitiveMemory

from probe_cli.corpus import load_corpus, load_query_corpus, load_seed_groups
from probe_cli.situational_runner import restore_seed_groups
from probe_cli.surfacing_runner import QuerySurfacing, SurfacingArmRunner
from probe_cli.surfacing_situational import (
    GroupedFactCoverage,
    compute_grouped_fact_coverage,
)

if TYPE_CHECKING:
    from recollect.llm.protocol import LLMProvider

    from probe_cli.arm import Arm
    from probe_cli.corpus import CorpusEntry

logger = logging.getLogger(__name__)


class SituationalSurfacingReport(BaseModel):
    arm_name: str
    model: str
    seeded_traces: int
    promoted_facts: int
    coverage: GroupedFactCoverage
    queries: list[QuerySurfacing] = Field(default_factory=list)


class SituationalSurfacingArmRunner(SurfacingArmRunner):
    _groups_path: str

    @classmethod
    def from_arm(
        cls, arm: Arm, provider: LLMProvider
    ) -> SituationalSurfacingArmRunner:
        s = arm.surfacing
        if not s.enabled:
            raise ValueError(
                "SituationalSurfacingArmRunner requires arm.surfacing.enabled=True"
            )
        if not s.traces_corpus_path or not s.seed_groups_path:
            raise ValueError(
                "situational-surfacing arm requires traces_corpus_path, "
                "seed_groups_path"
            )
        runner = cls(
            arm_name=arm.name,
            provider=provider,
            config=arm.to_memory_config(),
            traces_path=s.traces_corpus_path,
            queries_path=s.query_corpus_path,
            db_url=s.db_url,
        )
        runner._groups_path = s.seed_groups_path
        return runner

    async def _seed_clean_and_promote(
        self,
        memory: CognitiveMemory,
        traces: list[CorpusEntry],
        *,
        user_id: str,
    ) -> tuple[dict[str, str], dict[str, str], int]:
        # Seed clean: recall_tokens off so only the restored groups define token
        # structure (mirrors the situational arm's seed discipline).
        self._config._set("recall_tokens.enabled", False)
        db_to_eval = await self._seed(memory, traces, user_id=user_id)
        promoted = await self._promote_candidates(memory, user_id=user_id)
        seed_id_map = {ev: db for db, ev in db_to_eval.items()}
        return db_to_eval, seed_id_map, promoted

    async def run_measurement(self) -> SituationalSurfacingReport:
        # Distinct from the parent run() (organic corpus, SurfacingRunReport):
        # seed clean -> restore the token groups -> enable tokens for query-time
        # activation -> score the bridge queries. Coverage rides along as the
        # substrate signal.
        traces = load_corpus(self._traces_path).entries
        groups = load_seed_groups(self._groups_path)
        queries = load_query_corpus(self._queries_path).entries
        user_id = f"{self._user_id}-{int(time.time() * 1000)}"
        memory = CognitiveMemory(extractor=self._extractor, config=self._config)
        try:
            await memory.connect(db_url=self._db_url or None)
            db_to_eval, seed_id_map, promoted = await self._seed_clean_and_promote(
                memory, traces, user_id=user_id
            )
            await restore_seed_groups(memory, groups, seed_id_map)
            # Seeding stayed clean; enable activation for the query loop only.
            self._config._set("recall_tokens.enabled", True)
            facts = await memory.storage.facts.get_persona_facts(
                limit=10_000, user_id=user_id
            )
            coverage = compute_grouped_fact_coverage(facts, groups, seed_id_map)
            scored = [
                await self._score_query(memory, q, db_to_eval, user_id=user_id)
                for q in queries
            ]
            surfaced = [qs for qs in scored if qs is not None]
        finally:
            await memory.close()
        return SituationalSurfacingReport(
            arm_name=self._arm_name,
            model=self.model,
            seeded_traces=len(db_to_eval),
            promoted_facts=promoted,
            coverage=coverage,
            queries=surfaced,
        )
