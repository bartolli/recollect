"""Core cognitive memory system.

Orchestrates storage, embeddings, extraction, and working memory
into a coherent API for human-like memory operations.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import timedelta
from typing import Any, cast

from pydantic import ValidationError

from recollect.buffer import WorkingMemory
from recollect.config import MemoryConfig
from recollect.config import config as default_config
from recollect.datetime_utils import memory_timestamp_for_comparison, now_utc
from recollect.embeddings import FastEmbedProvider
from recollect.exceptions import (
    EmbeddingError,
    ExtractionError,
    PromptValidationError,
    SessionNotFoundError,
    StorageError,
    TraceNotFoundError,
)
from recollect.extraction import PatternExtractor
from recollect.llm.types import (
    ExtractionResult,
    Message,
    Relation,
    SituationalAssessment,
    TokenAssessment,
)
from recollect.models import (
    Association,
    ConceptEmbedding,
    ConsolidationResult,
    EntityRelation,
    FactCategory,
    FactStatus,
    HealthStatus,
    MemoryStats,
    MemoryTrace,
    PersonaFact,
    RecallToken,
    Session,
    Thought,
    TraceConcept,
    TraceEntity,
    _clamp_strength,
    apply_activation_boost,
    apply_displacement_decay,
    apply_retrieval_boost,
    apply_time_decay,
)
from recollect.prompts import LoadedPrompt, load_packaged_default, load_prompt_file
from recollect.storage_context import StorageContext, create_storage_context

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (chars / 4)."""
    return max(1, len(text) // 4)


def _compute_decay_rate(
    base_rate: float,
    significance: float,
    emotional_valence: float,
    *,
    sig_reduction: float = 0.7,
    val_reduction: float = 0.5,
) -> float:
    """Compute decay rate reduced by significance and emotional intensity.

    High significance and strong emotion make memories last longer.
    """
    sig_factor = 1.0 - (significance * sig_reduction)
    emo_factor = 1.0 - (abs(emotional_valence) * val_reduction)
    return base_rate * sig_factor * emo_factor


def _rrf_fuse(
    ranked_lists: dict[str, list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple signal sources.

    Each source provides a ranked list of trace IDs (best first).
    Returns {trace_id: fused_score} ordered by score descending.
    """
    scores: dict[str, float] = {}
    for source_ids in ranked_lists.values():
        for rank, trace_id in enumerate(source_ids, start=1):
            scores[trace_id] = scores.get(trace_id, 0.0) + 1.0 / (k + rank)
    return scores


_PREDICATE_ALIASES: dict[str, str] = {
    "started_at": "works_at",
    "employed_at": "works_at",
    "joined": "works_at",
    "has_allergy": "is_allergic_to",
    "allergic_to": "is_allergic_to",
    "likes": "prefers",
    "enjoys": "prefers",
    "lives_in": "located_in",
    "moved_to": "located_in",
}


def _canonicalize_predicate(predicate: str) -> str:
    """Normalize predicate to canonical form via alias lookup."""
    return _PREDICATE_ALIASES.get(predicate, predicate)


_SURFACING_RANK: dict[str, int] = {
    "archived": 0,
    "candidate": 0,
    "promoted": 1,
    "pinned": 2,
}


def _inherit_surfacing_status(old: PersonaFact, new: PersonaFact) -> PersonaFact:
    # A correction must not demote out of recall: pinned stays pinned,
    # promoted stays promoted. Archived/candidate old facts inherit nothing.
    if _SURFACING_RANK[old.status] > _SURFACING_RANK[new.status]:
        return new.model_copy(update={"status": old.status})
    return new


def _relations_from_pattern(pattern: dict[str, Any]) -> list[Relation]:
    # trace.pattern persists the full ExtractionResult dump; rows predating
    # that contract or carrying junk shapes fall back to no relations.
    relations: list[Relation] = []
    for item in pattern.get("relations") or []:
        try:
            relations.append(Relation.model_validate(item))
        except ValidationError:
            continue
    return relations


_CATEGORY_SCOPE_MAP: dict[str, str] = {
    "health": "health/safety",
    "dietary": "health/safety",
    "constraint": "health/safety",
    "relationship": "social",
    "identity": "identity",
    "preference": "preference",
    "schedule": "routine",
    "general": "general",
}


def _category_to_scope(category: FactCategory) -> str:
    """Map a fact category to a retrieval scope."""
    return _CATEGORY_SCOPE_MAP.get(category, "general")


_FAST_TRACK_CATEGORIES: frozenset[str] = frozenset(
    {
        "health",
        "dietary",
        "constraint",
    }
)

# Write-time safety surfacing is recall-maximal by the design asymmetry: an extra
# ignored line of context costs nothing, a silent safety miss costs the feature.
# Every safety-relevant domain surfaces ALL safety categories; the runtime LLM
# does the per-write discard. Domains absent here (social/finance/general) surface
# no safety fact. The map keeps per-domain shape so narrowing a domain to a subset
# is a measured follow-up, not a refactor.
_DOMAIN_SAFETY_MAP: dict[str, frozenset[str]] = {
    "food": _FAST_TRACK_CATEGORIES,
    "travel": _FAST_TRACK_CATEGORIES,
    "medication": _FAST_TRACK_CATEGORIES,
    "medical": _FAST_TRACK_CATEGORIES,
    "exercise": _FAST_TRACK_CATEGORIES,
    "environment": _FAST_TRACK_CATEGORIES,
}


def _should_fast_track(category: str, confidence: float) -> bool:
    """Check if a fact should skip candidate stage.

    Only safety-critical categories fast-track. The confidence
    parameter is accepted but not used for fast-track decisions.
    """
    return category in _FAST_TRACK_CATEGORIES


def _find_contradicting_fact(
    existing: list[PersonaFact], new_fact: PersonaFact
) -> PersonaFact | None:
    """Find existing fact with same subject+predicate but different object."""
    canonical_new = _canonicalize_predicate(new_fact.predicate)
    for fact in existing:
        canonical_existing = _canonicalize_predicate(fact.predicate)
        if canonical_existing == canonical_new and fact.object != new_fact.object:
            return fact
    return None


def _find_exact_duplicate(
    existing: list[PersonaFact], new_fact: PersonaFact
) -> PersonaFact | None:
    """Find existing fact with same subject+predicate+object."""
    canonical_new = _canonicalize_predicate(new_fact.predicate)
    for fact in existing:
        canonical_existing = _canonicalize_predicate(fact.predicate)
        if canonical_existing == canonical_new and fact.object == new_fact.object:
            return fact
    return None


def _collect_concept_phrases(
    result: ExtractionResult, *, include_relation_tags: bool
) -> list[str]:
    # Order-preserving case-insensitive dedupe across concepts and (optionally)
    # per-relation context_tags. Tags broaden the trace's MaxSim surface symmetric
    # with how persona facts already do via _embed_fact_tags.
    seen: set[str] = set()
    out: list[str] = []
    sources: list[list[str]] = [list(result.concepts)]
    if include_relation_tags:
        for rel in result.relations:
            sources.append(list(rel.context_tags))
    for src in sources:
        for phrase in src:
            cleaned = phrase.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
    return out


_QUERY_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "what",
        "how",
        "when",
        "where",
        "why",
        "who",
        "which",
        "my",
        "our",
        "his",
        "her",
        "their",
        "its",
        "i",
        "we",
        "it",
        "he",
        "she",
        "they",
        "you",
        "do",
        "does",
        "did",
        "is",
        "are",
        "was",
        "were",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "have",
        "has",
        "had",
        "not",
        "no",
        "yes",
        "if",
        "but",
        "and",
        "or",
        "so",
        "yet",
        "for",
        "got",
        "get",
        "go",
        "plan",
        "book",
        "find",
        "need",
        "want",
        "let",
        "tell",
        "show",
        "give",
        "take",
        "make",
        "help",
        "remember",
    }
)


class CognitiveMemory:
    """Human-like memory system with cognitive model.

    Combines working memory buffer, long-term storage with semantic
    search, spreading activation, and strength-based consolidation.
    """

    def __init__(
        self,
        *,
        storage: StorageContext | None = None,
        embeddings: FastEmbedProvider | None = None,
        extractor: PatternExtractor | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        self._config = config or default_config
        self._storage = storage or create_storage_context()
        self._embeddings = embeddings or FastEmbedProvider.from_config(self._config)
        self._extractor = extractor
        self._buffer = WorkingMemory(self._config.working_memory_capacity)
        self._connected = False
        self._situational_default: LoadedPrompt | None = None

    def _situational_defaults(self) -> tuple[str, str]:
        if self._situational_default is None:
            path = str(
                self._config.get("recall_tokens.assessment_template_path", "") or ""
            ).strip()
            if path:
                loaded = load_prompt_file(path)
                if loaded.applies_to != "situational":
                    raise PromptValidationError(
                        f"{path}: applies-to must be 'situational', "
                        f"got '{loaded.applies_to}'"
                    )
            else:
                loaded = load_packaged_default("situational.default.md")
            self._situational_default = loaded
            logger.info(
                "Situational template loaded: version=%s source=%s",
                loaded.version,
                loaded.source,
            )
        return self._situational_default.system, self._situational_default.user

    # -- Lifecycle --

    @property
    def storage(self) -> StorageContext:
        return self._storage

    async def connect(self, db_url: str | None = None) -> None:
        """Open storage, apply bootstrap, verify embedding contract."""
        if db_url:
            self._storage = create_storage_context(db_url)
        await self._storage.initialize()
        from recollect.storage_ops import (
            get_embedding_contract,
            verify_embedding_contract,
        )

        pool = await self._storage.pool.get_pool()
        stored = await get_embedding_contract(pool)
        verify_embedding_contract(stored=stored, current=self._embeddings.contract())
        self._connected = True
        logger.info("CognitiveMemory connected")

    async def close(self) -> None:
        """Close storage connection."""
        await self._storage.close()
        self._connected = False
        logger.info("CognitiveMemory closed")

    async def __aenter__(self) -> CognitiveMemory:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()

    # -- Core operations --

    async def experience(
        self,
        content: str,
        *,
        context: dict[str, Any] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> MemoryTrace:
        """Encode a new experience as a memory trace.

        1. Extract patterns (concepts, entities, emotion)
        2. Generate embedding vector
        3. Create trace and add to working memory
        4. Handle displacement decay if buffer was full
        5. Store in long-term storage with temporal association
        """
        if not content or not content.strip():
            raise ValueError("Content must be a non-empty string")
        # No extractor => no persona facts => nothing to lose; gate only the
        # lossy path. persona_facts.user_id is NOT NULL, so a user-less write
        # would silently drop every promoted relation.
        if (
            user_id is None
            and self._extractor is not None
            and self._config.get("persona.auto_extract", True)
        ):
            raise ValueError(
                "user_id is required when persona.auto_extract is enabled: "
                "persona facts are user-owned. Pass user_id or set "
                "persona.auto_extract=false."
            )

        result = await self._extract_pattern(content)
        embedding = await self._embeddings.generate_embedding(
            content, task="search_document"
        )

        decay_rate = _compute_decay_rate(
            base_rate=float(self._config.get("memory.decay_rate", 0.1)),
            significance=result.significance,
            emotional_valence=result.emotional_valence,
            sig_reduction=float(self._config.get("decay.significance_reduction", 0.7)),
            val_reduction=float(self._config.get("decay.valence_reduction", 0.5)),
        )
        trace = MemoryTrace(
            content=content,
            pattern=result.model_dump(),
            context=context or {},
            embedding=embedding,
            emotional_valence=result.emotional_valence,
            significance=result.significance,
            decay_rate=decay_rate,
            session_id=session_id,
            user_id=user_id,
        )

        if session_id is not None:
            await self._ensure_session(session_id, user_id)

        # Buffer ⊆ memory_traces invariant: the INSERT precedes the buffer
        # add, so a storage failure cannot strand an unpersisted trace in
        # working memory or persist displacement decay for a write that
        # never happened.
        await self._storage.traces.store_trace(trace)

        displaced = self._buffer.add(trace)
        if displaced is not None:
            decayed = apply_displacement_decay(displaced)
            await self._storage.traces.update_trace_strength(
                displaced.id, decayed.strength
            )

        # Independent operations: temporal link + extraction links + concept embeddings
        await asyncio.gather(
            self._create_temporal_association(trace),
            self._store_extraction_links(trace, result),
            self._embed_trace_concepts(trace, result),
        )

        # Depend on extraction links, but independent of each other
        await asyncio.gather(
            self._create_entity_associations(trace, result),
            self._create_concept_associations(trace, result),
        )

        if self._config.get("persona.auto_extract", True):
            await self._extract_persona_facts(trace, result)

        await self._extract_entity_relations(trace, result)

        if self._config.get("recall_tokens.enabled", True):
            await self._assess_recall_tokens(trace)

        logger.debug("Experienced: %s (strength=%.2f)", trace.id[:8], trace.strength)
        return trace

    async def think_about(
        self,
        query: str,
        *,
        token_budget: int = 2000,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[Thought]:
        """Retrieve relevant memories within a token budget.

        1. Generate query embedding
        2. Check working memory and long-term storage
        3. Spread activation from top candidates
        4. Merge, rank, select within budget
        5. Apply retrieval/activation boosts
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if token_budget <= 0:
            raise ValueError("Token budget must be positive")

        query_embedding = await self._embeddings.generate_embedding(
            query, task="search_query"
        )

        search_limit = int(self._config.get("retrieval.search_limit", 10))
        wm_candidates = self._search_working_memory(query_embedding, user_id=user_id)
        storage_scored = await self._storage.vectors.search_semantic(
            query_embedding,
            limit=search_limit,
            session_id=session_id,
            user_id=user_id,
        )
        storage_candidates = [trace for trace, _ in storage_scored]

        entity_matches = await self._match_query_entities(query, user_id=user_id)
        if entity_matches:
            entity_trace_ids = [tid for tid, _ in entity_matches]
            entity_candidates = await self._storage.traces.get_traces_bulk(
                entity_trace_ids
            )
            seen_ids = {t.id for t, _ in storage_scored}
            for t in entity_candidates:
                if t.id not in seen_ids:
                    sim = (
                        _cosine_similarity(query_embedding, t.embedding)
                        if t.embedding is not None
                        else 0.0
                    )
                    storage_scored.append((t, sim))
                    storage_candidates.append(t)
                    seen_ids.add(t.id)

        seed_count = int(self._config.get("retrieval.spread_seed_count", 3))
        activated = await self._spread_from_candidates(
            storage_candidates[:seed_count], user_id=user_id
        )

        token_activated = await self._activate_recall_tokens(
            query_embedding, storage_scored, user_id=user_id
        )

        all_candidates = await self._merge_candidates(
            query_embedding,
            wm_candidates,
            storage_scored,
            activated,
            entity_matches=entity_matches if entity_matches else None,
            token_activated=token_activated,
            user_id=user_id,
        )

        if session_id is not None:
            all_candidates = [
                (t, s) for t, s in all_candidates if t.session_id == session_id
            ]

        selected = self._select_within_budget(all_candidates, token_budget)

        thoughts = await self._boost_and_build_thoughts(selected)
        await self._boost_unselected(all_candidates, thoughts)

        persona_facts, semantic_scores = await self._find_relevant_persona_facts(
            query,
            query_embedding,
            user_id=user_id,
        )
        if persona_facts:
            fact_thoughts = self._persona_facts_to_thoughts(
                persona_facts,
                semantic_scores,
            )
            thoughts = thoughts + fact_thoughts

        max_total = int(self._config.get("retrieval.max_retrievals", 7))
        min_traces = int(self._config.get("retrieval.min_trace_slots", 2))
        thoughts = self._assemble_with_trace_guarantee(
            thoughts,
            max_total,
            min_traces,
        )

        logger.debug(
            "think_about: %d thoughts (budget=%d)",
            len(thoughts),
            token_budget,
        )
        return thoughts

    @staticmethod
    def _assemble_with_trace_guarantee(
        thoughts: list[Thought],
        max_total: int,
        min_traces: int,
    ) -> list[Thought]:
        """Assemble final thoughts guaranteeing minimum trace slots.

        Pinned facts get priority but cannot consume more than
        (max_total - min_traces) slots. The reserved slots hold actual
        memory traces -- persona-fact thoughts are facts regardless of
        pin flag, so they never satisfy the trace guarantee.
        Remaining slots filled by relevance regardless of kind.
        """
        pinned = sorted(
            [t for t in thoughts if t.pinned],
            key=lambda t: -t.relevance,
        )
        traces_only = sorted(
            [
                t
                for t in thoughts
                if not t.pinned and not t.trace.pattern.get("persona_fact")
            ],
            key=lambda t: -t.relevance,
        )
        max_pinned = max(0, max_total - min_traces)
        selected = pinned[:max_pinned] + traces_only[:min_traces]
        chosen = {id(t) for t in selected}
        remaining = sorted(
            [t for t in thoughts if id(t) not in chosen],
            key=lambda t: -t.relevance,
        )
        selected += remaining[: max_total - len(selected)]
        selected.sort(key=lambda t: -t.relevance)
        return selected

    async def consolidate(self, threshold: float | None = None) -> ConsolidationResult:
        """Consolidate or forget pending traces based on strength.

        Traces above threshold are consolidated to long-term memory.
        Traces past their grace period that remain weak are forgotten.
        Young weak traces are kept pending with updated decay.
        """
        threshold = threshold or self._config.consolidation_threshold
        grace_hours = float(self._config.get("forgetting.grace_period_hours", 24))
        batch_size = int(self._config.get("consolidation.batch_size", 50))
        traces = await self._storage.traces.get_unconsolidated_traces(limit=batch_size)

        consolidated = 0
        forgotten = 0
        still_pending = 0

        for trace in traces:
            decayed = apply_time_decay(trace)
            result = await self._consolidate_one(trace, decayed, threshold, grace_hours)
            if result == "consolidated":
                consolidated += 1
            elif result == "forgotten":
                forgotten += 1
            else:
                still_pending += 1

        if self._config.get("recall_tokens.enabled", True):
            decay_factor = float(
                self._config.get("recall_tokens.decay_factor", 0.9)
            )
            inactivity = float(
                self._config.get("recall_tokens.decay_inactivity_seconds", 1800)
            )
            cutoff = now_utc() - timedelta(seconds=inactivity)
            try:
                tokens_decayed = await self._storage.recall_tokens.decay_inactive(
                    decay_factor, inactive_before=cutoff
                )
                if tokens_decayed > 0:
                    logger.debug("Decayed %d recall tokens", tokens_decayed)
            except (StorageError, OSError):
                logger.exception("Recall token decay failed during consolidation")

        logger.info(
            "Consolidation: %d consolidated, %d forgotten, %d pending",
            consolidated,
            forgotten,
            still_pending,
        )
        return ConsolidationResult(
            consolidated=consolidated,
            forgotten=forgotten,
            still_pending=still_pending,
        )

    async def forget(self, trace_id: str) -> bool:
        """Explicitly forget a memory trace and its derived facts."""
        if not trace_id:
            raise ValueError("Trace ID must be a non-empty string")
        deleted = await self._delete_trace_with_buffer_evict(trace_id)
        if not deleted:
            raise TraceNotFoundError(f"Trace {trace_id} not found")
        await self._cleanup_trace_facts(trace_id)
        logger.debug("Forgot trace: %s", trace_id[:8])
        return True

    async def _delete_trace_with_buffer_evict(self, trace_id: str) -> bool:
        # Buffer ⊆ memory_traces invariant: dangling refs FK-violate temporal assoc.
        deleted = await self._storage.traces.delete_trace(trace_id)
        if deleted:
            self._buffer.evict(trace_id)
        return deleted

    async def _cleanup_trace_facts(self, trace_id: str) -> None:
        """Remove persona facts and entity relations linked to a trace."""
        try:
            facts = await self._storage.facts.get_persona_facts()
            for fact in facts:
                if fact.source_trace_id == trace_id:
                    await self._storage.concept_embeddings.delete_by_owner(
                        "fact", fact.id
                    )
                    await self._storage.facts.delete_persona_fact(fact.id)
            await self._storage.concept_embeddings.delete_by_owner("trace", trace_id)
            await self._storage.entity_relations.delete_by_trace(trace_id)
            await self._storage.recall_tokens.delete_by_trace(trace_id)
        except (StorageError, OSError):
            logger.exception(
                "Cleanup of derived data failed for trace %s",
                trace_id[:8],
            )

    async def reinforce(self, trace_id: str, *, factor: float = 1.1) -> MemoryTrace:
        """Manually strengthen a memory trace."""
        if not trace_id:
            raise ValueError("Trace ID must be a non-empty string")
        if factor <= 0.0:
            raise ValueError("Reinforcement factor must be positive")
        trace = await self._storage.traces.get_trace(trace_id)
        if trace is None:
            raise TraceNotFoundError(f"Trace {trace_id} not found")
        new_strength = _clamp_strength(trace.strength * factor)
        await self._storage.traces.update_trace_strength(trace_id, new_strength)
        return trace.model_copy(update={"strength": new_strength})

    # -- Introspection --

    def active_traces(self) -> list[MemoryTrace]:
        """Get traces currently in working memory."""
        return self._buffer.get_active()

    async def associations(self, trace_id: str) -> list[Association]:
        """Get all associations for a trace."""
        return await self._storage.associations.get_associations(trace_id)

    async def timeline(self, limit: int = 20) -> list[MemoryTrace]:
        """Get most recent memory traces."""
        if limit <= 0:
            raise ValueError("Limit must be positive")
        return await self._storage.traces.get_recent_traces(limit)

    def stats(self) -> MemoryStats:
        """Get memory system statistics."""
        wm_stats = self._buffer.get_stats()
        return MemoryStats(
            working_memory_items=wm_stats["current_items"],
            working_memory_capacity=wm_stats["capacity"],
            working_memory_utilization=wm_stats["utilization"],
            total_seen=wm_stats["total_seen"],
            total_displaced=wm_stats["total_displaced"],
            displacement_rate=wm_stats["displacement_rate"],
            connected=self._connected,
        )

    def health(self) -> HealthStatus:
        """Get system health status."""
        return HealthStatus(
            status="ok" if self._connected else "disconnected",
        )

    async def pin(self, trace_id: str) -> list[PersonaFact]:
        """Pin a trace's extracted relations as permanent persona facts.

        Promotes what extraction persisted in trace.pattern, bypassing
        the confidence gate -- pin is the user saying "this matters".
        The generic `user noted` SPO is only the empty-extraction
        fallback.
        """
        if not trace_id:
            raise ValueError("Trace ID must be a non-empty string")
        trace = await self._storage.traces.get_trace(trace_id)
        if trace is None:
            raise TraceNotFoundError(f"Trace {trace_id} not found")
        relations = _relations_from_pattern(trace.pattern)
        if not relations:
            fallback = PersonaFact(
                subject="user",
                predicate="noted",
                object=trace.content or "",
                content=trace.content or "",
                source_trace_id=trace_id,
                user_id=trace.user_id,
                confidence=1.0,
                status="pinned",
            )
            await self._storage.facts.store_persona_fact(fallback)
            return [fallback]
        facts: list[PersonaFact] = []
        for rel in relations:
            fact = await self._relation_to_fact(rel, trace, status="pinned")
            await self._storage.facts.store_persona_fact(fact)
            await self._embed_fact_tags(fact)
            facts.append(fact)
        return facts

    async def unpin(self, fact_id: str) -> bool:
        """Archive a persona fact: it leaves every surfacing path.

        The row is retained for audit. Returns False when the fact
        does not exist.
        """
        if not fact_id:
            raise ValueError("Fact ID must be a non-empty string")
        return await self._storage.facts.update_fact_status(fact_id, "archived")

    async def facts(
        self,
        subject: str | None = None,
        *,
        user_id: str | None = None,
    ) -> list[PersonaFact]:
        """List persona facts, optionally filtered by subject and user."""
        return await self._storage.facts.get_persona_facts(subject, user_id=user_id)

    async def start_session(
        self,
        *,
        user_id: str,
        title: str = "",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Start a new conversation session."""
        import uuid as _uuid

        session = Session(
            id=session_id or str(_uuid.uuid4()),
            user_id=user_id,
            title=title,
            metadata=metadata or {},
        )
        await self._storage.sessions.create_session(session)
        return session

    async def end_session(self, session_id: str) -> Session:
        """Mark a session as ended."""
        session = await self._storage.sessions.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} not found")
        await self._storage.sessions.end_session(session_id)
        return session.model_copy(update={"status": "ended", "ended_at": now_utc()})

    async def summarize_session(self, session_id: str) -> MemoryTrace:
        """Generate a summary trace from all memories in a session."""
        session = await self._storage.sessions.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} not found")
        traces = await self._storage.traces.get_traces_by_session(session_id)
        if not traces:
            raise ValueError(f"Session {session_id} has no traces")

        summary_text = await self._generate_session_summary(traces, session)
        embedding = await self._embeddings.generate_embedding(
            summary_text, task="search_document"
        )
        result = await self._extract_pattern(summary_text)

        summary_trace = MemoryTrace(
            content=summary_text,
            pattern={
                **result.model_dump(),
                "session_summary": True,
                "source_session_id": session_id,
                "trace_count": len(traces),
            },
            embedding=embedding,
            strength=float(self._config.get("session.summary_strength", 0.8)),
            significance=float(self._config.get("session.summary_significance", 0.7)),
            decay_rate=float(self._config.get("session.summary_decay_rate", 0.02)),
            emotional_valence=result.emotional_valence,
            session_id=session_id,
            user_id=session.user_id,
        )

        await self._storage.traces.store_trace(summary_trace)
        await self._storage.sessions.update_session(
            session_id,
            status="summarized",
            summary_trace_id=summary_trace.id,
        )

        await asyncio.gather(
            self._store_extraction_links(summary_trace, result),
            self._embed_trace_concepts(summary_trace, result),
        )

        return summary_trace

    async def get_sessions(
        self,
        user_id: str,
        *,
        limit: int = 50,
    ) -> list[Session]:
        """List sessions for a user."""
        return await self._storage.sessions.get_sessions(user_id, limit=limit)

    # -- Private: experience helpers --

    async def _extract_pattern(self, content: str) -> ExtractionResult:
        """Extract patterns via LLM, retry once, then fallback.

        Strategy: try -> retry -> store with empty extraction.
        A degraded trace (embedding only, no concept tags) is better
        than losing the memory entirely. Structured output eliminates
        JSON parse failures, but network and API errors still warrant
        a retry.
        """
        if self._extractor is None:
            return ExtractionResult()
        for attempt in range(2):
            try:
                return await self._extractor.extract(content)
            except (ExtractionError, ValueError, KeyError):
                if attempt == 0:
                    logger.exception("LLM extraction failed, retrying")
                    continue
                logger.exception(
                    "LLM extraction failed on retry, storing with empty extraction"
                )
                return ExtractionResult()
        return ExtractionResult()  # unreachable, satisfies type checker

    async def _ensure_session(
        self,
        session_id: str,
        user_id: str | None,
    ) -> None:
        """Create session if it does not exist (lazy creation)."""
        existing = await self._storage.sessions.get_session(session_id)
        if existing is None:
            if user_id is None:
                raise ValueError("user_id required when creating a new session")
            session = Session(id=session_id, user_id=user_id)
            await self._storage.sessions.create_session(session)

    async def _generate_session_summary(
        self,
        traces: list[MemoryTrace],
        session: Session,
    ) -> str:
        """Generate session summary via LLM or fallback concatenation."""
        if self._extractor is None:
            contents = [t.content for t in traces if t.content]
            return f"Session ({len(traces)} memories): " + " | ".join(contents[:10])

        trace_texts = [
            f"{i}. {t.content}" for i, t in enumerate(traces, 1) if t.content
        ]
        prompt = (
            "Summarize this conversation session into a single cohesive paragraph. "
            "Capture key topics, decisions, facts learned, and unresolved questions. "
            "Include all proper nouns, numbers, and specific details. "
            "This summary will be embedded for semantic retrieval.\n\n"
            f"Session: {session.title or '(untitled)'}\n" + "\n".join(trace_texts)
        )

        max_tokens = int(self._config.get("session.summary_max_tokens", 1024))
        messages = [
            Message(
                role="system",
                content=(
                    "You are a memory consolidation system. Produce a dense, "
                    "factual summary paragraph. No filler. Include all proper nouns, "
                    "numbers, and specific details."
                ),
            ),
            Message(role="user", content=prompt),
        ]

        try:
            return (
                await self._extractor._provider.complete(
                    messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            ).strip()
        except (ExtractionError, ValueError, RuntimeError, OSError):
            logger.exception("Session summary LLM failed")
            raise

    async def _create_temporal_association(self, trace: MemoryTrace) -> None:
        """Create temporal association to the previous trace in buffer."""
        active = self._buffer.get_active()
        if len(active) < 2:
            return
        previous = active[-2]
        weight = float(self._config.get("associations.temporal_weight", 0.5))
        association = Association(
            source_trace_id=previous.id,
            target_trace_id=trace.id,
            association_type="temporal",
            weight=weight,
        )
        try:
            await self._storage.associations.store_association(association)
        except (StorageError, OSError):
            logger.exception(
                "Failed to create temporal association %s -> %s",
                previous.id[:8],
                trace.id[:8],
            )

    async def _store_extraction_links(
        self, trace: MemoryTrace, result: ExtractionResult
    ) -> None:
        """Store entity and concept links from extraction result."""
        entities = [
            TraceEntity(
                entity_name=e.name,
                entity_type=e.entity_type,
                trace_id=trace.id,
            )
            for e in result.entities
        ]
        concepts = [TraceConcept(concept=c, trace_id=trace.id) for c in result.concepts]
        try:
            await self._storage.entities.store_trace_entities(trace.id, entities)
            await self._storage.entities.store_trace_concepts(trace.id, concepts)
        except (StorageError, OSError):
            logger.exception("Failed to store extraction links for %s", trace.id[:8])

    async def _embed_trace_concepts(
        self,
        trace: MemoryTrace,
        result: ExtractionResult,
    ) -> None:
        # Persona facts already MaxSim their context_tags (_embed_fact_tags).
        # Traces that don't promote lose that surface unless we embed here too.
        # Flag-gated: opt-in via extraction.embed_relation_tags.
        phrases = _collect_concept_phrases(
            result,
            include_relation_tags=bool(
                self._config.get("extraction.embed_relation_tags", False)
            ),
        )
        if not phrases:
            return
        try:
            vectors = await self._embeddings.generate_embeddings_batch(
                phrases, task="search_document"
            )
            embeddings = [
                ConceptEmbedding(
                    concept=c,
                    owner_type="trace",
                    owner_id=trace.id,
                    embedding=v,
                )
                for c, v in zip(phrases, vectors, strict=True)
            ]
            await self._storage.concept_embeddings.store_concept_embeddings(embeddings)
        except (StorageError, OSError):
            logger.exception("Failed to embed concepts for trace %s", trace.id[:8])

    async def _embed_fact_tags(self, fact: PersonaFact) -> None:
        """Embed each context tag individually for concept attention."""
        if not fact.context_tags:
            return
        try:
            vectors = await self._embeddings.generate_embeddings_batch(
                fact.context_tags, task="search_document"
            )
            embeddings = [
                ConceptEmbedding(
                    concept=tag,
                    owner_type="fact",
                    owner_id=fact.id,
                    embedding=v,
                )
                for tag, v in zip(fact.context_tags, vectors, strict=True)
            ]
            await self._storage.concept_embeddings.store_concept_embeddings(embeddings)
        except (StorageError, OSError):
            logger.exception("Failed to embed tags for fact %s", fact.id[:8])

    async def _create_entity_associations(
        self, trace: MemoryTrace, result: ExtractionResult
    ) -> None:
        """Create associations to traces sharing entities."""
        weight = float(self._config.get("associations.entity_weight", 0.7))
        max_links = int(self._config.get("associations.max_links_per_entity", 5))
        for entity in result.entities:
            await self._link_traces_by_key(
                trace, entity.name, "entity", weight, max_links
            )

    async def _create_concept_associations(
        self, trace: MemoryTrace, result: ExtractionResult
    ) -> None:
        """Create associations to traces sharing concepts."""
        weight = float(self._config.get("associations.concept_weight", 0.4))
        max_links = int(self._config.get("associations.max_links_per_entity", 5))
        for concept in result.concepts:
            await self._link_traces_by_key(trace, concept, "concept", weight, max_links)

    async def _link_traces_by_key(
        self,
        trace: MemoryTrace,
        key: str,
        assoc_type: str,
        weight: float,
        max_links: int,
    ) -> None:
        """Create association for a shared key.

        DB pair_key column handles bidirectionality.
        """
        try:
            if assoc_type == "entity":
                existing_ids = await self._storage.entities.get_traces_by_entity(
                    key, limit=max_links
                )
            else:
                existing_ids = await self._storage.entities.get_traces_by_concept(
                    key, limit=max_links
                )
        except (StorageError, OSError):
            logger.exception("Failed to find traces for %s=%s", assoc_type, key)
            return

        for other_id in existing_ids:
            if other_id == trace.id:
                continue
            association = Association(
                source_trace_id=trace.id,
                target_trace_id=other_id,
                association_type=assoc_type,
                weight=weight,
                forward_strength=weight,
                backward_strength=weight,
            )
            try:
                await self._storage.associations.store_association(association)
            except (StorageError, OSError):
                logger.exception(
                    "Failed to create %s association %s <-> %s",
                    assoc_type,
                    trace.id[:8],
                    other_id[:8],
                )

    async def _extract_persona_facts(
        self, trace: MemoryTrace, result: ExtractionResult
    ) -> None:
        """Extract persona facts from semantic-type traces with promotion gate."""
        if result.fact_type != "semantic":
            return
        threshold = float(self._config.get("persona.confidence_threshold", 0.6))
        for rel in result.relations:
            if rel.confidence < threshold:
                continue
            status: FactStatus = (
                "promoted"
                if _should_fast_track(rel.category, rel.confidence)
                else "candidate"
            )
            fact = await self._relation_to_fact(rel, trace, status=status)
            stored = await self._store_or_promote_fact(fact)
            if stored is not None:
                await self._embed_fact_tags(stored)

    async def _relation_to_fact(
        self, rel: Relation, trace: MemoryTrace, *, status: FactStatus
    ) -> PersonaFact:
        # Single construction path: extraction-time promotion and pin()
        # must produce identical facts apart from status.
        content = rel.context or f"{rel.source} {rel.relation} {rel.target}"
        try:
            fact_embedding = await self._embeddings.generate_embedding(
                content, task="search_document"
            )
        except (ValueError, RuntimeError, OSError):
            fact_embedding = None
        return PersonaFact(
            subject=rel.source,
            predicate=_canonicalize_predicate(rel.relation),
            object=rel.target,
            category=rel.category,
            content=content,
            source_trace_id=trace.id,
            user_id=trace.user_id,
            confidence=rel.confidence,
            status=status,
            scope=_category_to_scope(rel.category),
            context_tags=rel.context_tags,
            embedding=fact_embedding,
        )

    async def _store_or_promote_fact(self, fact: PersonaFact) -> PersonaFact | None:
        """Store a new fact or promote an existing candidate.

        Returns the fact iff a new row was written; None on the duplicate
        and failure paths. Derived writes (tag embeddings) key off this --
        embedding tags for a never-stored fact id orphans them.
        """
        try:
            existing = await self._storage.facts.get_persona_facts(subject=fact.subject)
            duplicate = _find_exact_duplicate(existing, fact)
            if duplicate:
                new_count = await self._storage.facts.increment_mention_count(
                    duplicate.id
                )
                if new_count >= 2 and duplicate.status == "candidate":
                    await self._storage.facts.update_fact_status(
                        duplicate.id, "promoted"
                    )
                    logger.debug(
                        "Promoted fact: %s %s %s (mentions=%d)",
                        duplicate.subject,
                        duplicate.predicate,
                        duplicate.object,
                        new_count,
                    )
                return None
            contradicting = _find_contradicting_fact(existing, fact)
            if contradicting:
                return await self._check_predicate_similarity(contradicting, fact)
            await self._storage.facts.store_persona_fact(fact)
            return fact
        except (StorageError, OSError):
            logger.exception(
                "Failed to store/promote fact: %s %s %s",
                fact.subject,
                fact.predicate,
                fact.object,
            )
            return None

    async def _check_predicate_similarity(
        self, old_fact: PersonaFact, new_fact: PersonaFact
    ) -> PersonaFact:
        """Supersede if predicates match exactly or by embedding similarity.

        Every branch writes new_fact; returns it for derived-write gating.
        """
        if old_fact.predicate == new_fact.predicate:
            replacement = _inherit_surfacing_status(old_fact, new_fact)
            await self._storage.facts.supersede_persona_fact(old_fact.id, replacement)
            return replacement
        threshold = float(
            self._config.get("persona.predicate_similarity_threshold", 0.8)
        )
        emb_a = await self._embeddings.generate_embedding(
            old_fact.predicate, task="clustering"
        )
        emb_b = await self._embeddings.generate_embedding(
            new_fact.predicate, task="clustering"
        )
        if _cosine_similarity(emb_a, emb_b) >= threshold:
            replacement = _inherit_surfacing_status(old_fact, new_fact)
            await self._storage.facts.supersede_persona_fact(old_fact.id, replacement)
            return replacement
        await self._storage.facts.store_persona_fact(new_fact)
        return new_fact

    async def _extract_entity_relations(
        self, trace: MemoryTrace, result: ExtractionResult
    ) -> None:
        """Create entity-to-entity edges from extracted relations."""
        for rel in result.relations:
            if not rel.source or not rel.target:
                continue
            entity_rel = EntityRelation(
                source_entity=rel.source,
                relation=rel.relation,
                target_entity=rel.target,
                context=f"{rel.source} {rel.relation} {rel.target}",
                weight=rel.confidence,
                source_trace_id=trace.id,
            )
            try:
                await self._storage.entity_relations.store_relation(entity_rel)
            except (StorageError, OSError):
                logger.exception(
                    "Failed to store entity relation: %s -> %s",
                    rel.source,
                    rel.target,
                )

    # -- Private: recall token helpers --

    async def assess_situational(
        self, trace: MemoryTrace
    ) -> SituationalAssessment | None:
        """Assessment-only surface for measurement; bypasses recall_tokens.enabled."""
        if self._extractor is None or trace.embedding is None:
            return None
        related, existing_groups = await self._collect_assessment_inputs(trace)
        if not related:
            return None
        assessment = await self._call_token_assessment(
            trace, related, existing_groups
        )
        return SituationalAssessment(
            assessment=assessment,
            related_trace_ids=[t.id for t, _ in related],
            candidate_token_ids=[str(g["token_id"]) for g in existing_groups],
        )

    async def _collect_assessment_inputs(
        self, trace: MemoryTrace
    ) -> tuple[list[tuple[MemoryTrace, float]], list[dict[str, object]]]:
        assert trace.embedding is not None  # Guarded by callers
        top_k = int(self._config.get("recall_tokens.write_time_top_k", 5))
        threshold = float(self._config.get("recall_tokens.write_time_threshold", 0.42))
        # Scope to the trace's owner: a situational group must never span
        # users, and a cross-user trace must never anchor group discovery.
        related = await self._find_related_for_tokens(
            trace.embedding, trace.id, top_k, threshold, user_id=trace.user_id
        )
        if not related:
            return [], []
        related_ids = [t.id for t, _ in related]
        existing_groups = await self._storage.recall_tokens.find_groups_for_traces(
            related_ids, include_archived=True, user_id=trace.user_id
        )
        return related, existing_groups

    async def _assess_recall_tokens(self, trace: MemoryTrace) -> None:
        """Write-time assessment: find related, find groups, ask LLM, apply."""
        if self._extractor is None or trace.embedding is None:
            return
        try:
            related, existing_groups = await self._collect_assessment_inputs(trace)
            if not related:
                return
            assessment = await self._call_token_assessment(
                trace, related, existing_groups
            )
            await self._apply_token_action(trace, assessment, related, existing_groups)
        except (StorageError, ExtractionError, ValueError, RuntimeError, OSError):
            logger.exception(
                "Recall token assessment failed for trace %s",
                trace.id[:8],
            )

    async def _find_related_for_tokens(
        self,
        embedding: list[float],
        exclude_id: str,
        top_k: int,
        threshold: float,
        *,
        user_id: str | None = None,
    ) -> list[tuple[MemoryTrace, float]]:
        """Find traces related to the new trace above similarity threshold."""
        candidates = await self._storage.vectors.search_semantic(
            embedding, limit=top_k, user_id=user_id
        )
        return [
            (trace, sim)
            for trace, sim in candidates
            if trace.id != exclude_id and sim >= threshold
        ]

    async def _call_token_assessment(
        self,
        trace: MemoryTrace,
        related: list[tuple[MemoryTrace, float]],
        existing_groups: list[dict[str, object]],
    ) -> TokenAssessment:
        """Call LLM to assess situational group action for a new memory."""
        assert self._extractor is not None  # Guarded by caller
        # expose [in Gn] / [lone] to prompt: structural signal vs semantic inference.
        id_to_group_labels: dict[str, list[str]] = {}
        for gi, g in enumerate(existing_groups, 1):
            for tid in cast(list[str], g.get("stamped_trace_ids", [])):
                id_to_group_labels.setdefault(tid, []).append(f"G{gi}")
        numbered_lines = []
        for i, (t, _) in enumerate(related):
            bindings = id_to_group_labels.get(t.id)
            tag = f"[in {', '.join(bindings)}]" if bindings else "[lone]"
            numbered_lines.append(f"{i + 1}. {tag} {t.content}")
        numbered_list = "\n".join(numbered_lines)
        groups_str = self._format_existing_groups(related, existing_groups)
        default_system, default_user = self._situational_defaults()
        system_prompt = self._config.recall_token_system_prompt or default_system
        user_template = self._config.recall_token_user_prompt or default_user
        user_prompt = user_template.format(
            new_content=trace.content or "",
            numbered_list=numbered_list,
            existing_groups=groups_str,
        )
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        return await self._extractor._provider.complete_structured(
            messages,
            output_type=TokenAssessment,
            max_tokens=int(
                self._config.get("recall_tokens.assessment_max_tokens", 8192)
            ),
            temperature=0.0,
        )

    @staticmethod
    def _format_existing_groups(
        related: list[tuple[MemoryTrace, float]],
        groups: list[dict[str, object]],
    ) -> str:
        """Format existing groups for the LLM prompt."""
        if not groups:
            return "None"
        id_to_index = {t.id: i + 1 for i, (t, _) in enumerate(related)}
        lines = []
        for i, g in enumerate(groups, 1):
            stamped = cast(list[str], g["stamped_trace_ids"])
            indices = sorted(id_to_index[tid] for tid in stamped if tid in id_to_index)
            indices_str = ", ".join(str(idx) for idx in indices)
            sig = float(cast(float, g.get("significance", 0.5)))
            lines.append(
                f"G{i}: {g['label']} (memories: {indices_str}, significance: {sig:.1f})"
            )
        return "\n".join(lines)

    async def _apply_token_action(
        self,
        trace: MemoryTrace,
        assessment: TokenAssessment,
        related: list[tuple[MemoryTrace, float]],
        existing_groups: list[dict[str, object]],
    ) -> None:
        """Apply the assessed token action (create/extend/revise/none)."""
        if assessment.action == "create":
            await self._apply_token_create(trace, assessment, related)
        elif assessment.action == "extend":
            await self._apply_token_extend(trace, assessment, existing_groups)
        elif assessment.action == "revise":
            await self._apply_token_revise(trace, assessment, existing_groups)

    async def _apply_token_create(
        self,
        trace: MemoryTrace,
        assessment: TokenAssessment,
        related: list[tuple[MemoryTrace, float]],
    ) -> None:
        """Create a new situational group with the trace and linked memories."""
        if not assessment.linked_indices:
            return
        existing_ids = []
        for idx in assessment.linked_indices:
            if 1 <= idx <= len(related):
                existing_ids.append(related[idx - 1][0].id)
        if not existing_ids:
            return
        label = (
            f"{assessment.person_ref} | "
            f"{assessment.situation} | "
            f"{assessment.implication}"
        )
        all_ids = [trace.id, *existing_ids]
        token = RecallToken(label=label, significance=assessment.significance)
        await self._storage.recall_tokens.create_token(token)
        await self._storage.recall_tokens.stamp_traces(token.id, all_ids)
        logger.debug(
            "Created situational token %s (%s) on %d traces",
            token.id[:8],
            label,
            len(all_ids),
        )

    async def _apply_token_extend(
        self,
        trace: MemoryTrace,
        assessment: TokenAssessment,
        existing_groups: list[dict[str, object]],
    ) -> None:
        """Extend an existing group: stamp trace and append implication."""
        group_idx = assessment.group_number - 1
        if group_idx < 0 or group_idx >= len(existing_groups):
            return
        group = existing_groups[group_idx]
        token_id = str(group["token_id"])
        await self._storage.recall_tokens.stamp_traces(token_id, [trace.id])
        if assessment.implication:
            old_label = str(group["label"])
            new_label = self._append_implication(old_label, assessment.implication)
            await self._storage.recall_tokens.update_token_label(token_id, new_label)
        # Reactivate archived tokens -- reinforce_tokens handles the
        # strength=significance reset and status='active' transition
        if str(group.get("status", "active")) == "archived":
            boost = float(self._config.get("recall_tokens.reinforce_boost", 0.1))
            await self._storage.recall_tokens.reinforce_tokens(
                [token_id], boost
            )
            logger.debug(
                "Reactivated archived token %s with strength=significance",
                token_id[:8],
            )
        logger.debug(
            "Extended token %s with trace %s (implication: %s)",
            token_id[:8],
            trace.id[:8],
            assessment.implication,
        )

    async def _apply_token_revise(
        self,
        trace: MemoryTrace,
        assessment: TokenAssessment,
        existing_groups: list[dict[str, object]],
    ) -> None:
        """Revise an existing group: rewrite label and update significance."""
        group_idx = assessment.group_number - 1
        if group_idx < 0 or group_idx >= len(existing_groups):
            return
        group = existing_groups[group_idx]
        token_id = str(group["token_id"])
        old_label = str(group["label"])
        person_ref = old_label.split(" | ", 1)[0] if " | " in old_label else ""
        situation = assessment.situation or person_ref
        implication = assessment.implication
        new_label = f"{person_ref} | {situation} | {implication}"
        await self._storage.recall_tokens.update_token(
            token_id, new_label, assessment.significance
        )
        await self._storage.recall_tokens.stamp_traces(token_id, [trace.id])
        logger.debug(
            "Revised token %s: %s -> %s (significance=%.2f)",
            token_id[:8],
            old_label[:40],
            new_label[:40],
            assessment.significance,
        )

    @staticmethod
    def _append_implication(label: str, implication: str) -> str:
        """Append a new implication to a token label's implication section."""
        if " | " in label:
            parts = label.split(" | ")
            if len(parts) >= 3:
                parts[2] = parts[2] + ", " + implication
            else:
                parts.append(implication)
            return " | ".join(parts)
        return label + " | " + implication

    # -- Private: think_about helpers --

    def _search_working_memory(
        self, query_embedding: list[float], *, user_id: str | None = None
    ) -> list[tuple[MemoryTrace, float]]:
        """Find relevant traces in working memory by embedding similarity.

        The buffer is shared process state, not per-user; user_id filters
        candidates the same way the storage channels do. None: no filter.
        """
        wm_threshold = float(self._config.get("retrieval.wm_similarity_threshold", 0.3))
        results: list[tuple[MemoryTrace, float]] = []
        for trace in self._buffer.get_active():
            if user_id is not None and trace.user_id != user_id:
                continue
            if trace.embedding is not None:
                sim = _cosine_similarity(query_embedding, trace.embedding)
                if sim > wm_threshold:
                    results.append((trace, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    async def _spread_from_candidates(
        self, candidates: list[MemoryTrace], *, user_id: str | None = None
    ) -> list[tuple[MemoryTrace, float]]:
        """Spread activation from top candidates."""
        activated: list[tuple[MemoryTrace, float]] = []
        seen_ids: set[str] = set()
        for candidate in candidates:
            spreads = await self._storage.vectors.spread_activation(
                candidate.id, user_id=user_id
            )
            for trace, level in spreads:
                if trace.id not in seen_ids:
                    activated.append((trace, level))
                    seen_ids.add(trace.id)
        return activated

    async def _activate_recall_tokens(
        self,
        query_embedding: list[float],
        storage_scored: list[tuple[MemoryTrace, float]],
        *,
        user_id: str | None = None,
    ) -> dict[str, float]:
        """Query-time token activation with iterative re-seeding.

        Returns {trace_id: propagated_sim} for token-activated traces.
        Propagated similarity uses the formula:
            propagated_sim = anchor_cosine * hop_decay
                * token_strength * token_significance
        Multiple rounds re-seed from top discovered traces until ranking
        stabilizes or max_rounds is reached.
        """
        if not self._config.get("recall_tokens.enabled", True):
            return {}
        if not storage_scored:
            return {}
        hop_decay = float(self._config.get("recall_tokens.hop_decay", 0.85))
        strength_threshold = float(
            self._config.get("recall_tokens.strength_threshold", 0.1)
        )
        reinforce_boost = float(
            self._config.get("recall_tokens.reinforce_boost", 0.1)
        )
        max_rounds = int(self._config.get("recall_tokens.iter_max_rounds", 3))
        stability_threshold = float(
            self._config.get("recall_tokens.iter_stability_threshold", 0.95)
        )
        top_seeds = int(self._config.get("recall_tokens.iter_top_seeds", 3))

        # Build cosine lookup for seed traces
        seed_cosines: dict[str, float] = {t.id: sim for t, sim in storage_scored}
        seed_ids = list(seed_cosines.keys())

        # Track best propagated_sim per trace across all rounds
        propagated_sims: dict[str, float] = {}
        all_token_ids: set[str] = set()
        used_seeds: set[str] = set(seed_ids)

        # Round 1: one-hop from vector results
        round1_props, round1_tokens = await self._token_hop(
            seed_ids, seed_cosines, hop_decay, strength_threshold, user_id=user_id
        )
        propagated_sims.update(round1_props)
        all_token_ids.update(round1_tokens)

        if not propagated_sims:
            return {}

        prev_top_k = self._top_k_ids(propagated_sims, top_seeds)

        # Rounds 2..max_rounds: iterative re-seeding
        for _round_num in range(2, max_rounds + 1):
            new_seed_ids = [
                tid for tid in prev_top_k if tid not in used_seeds
            ]
            if not new_seed_ids:
                break
            used_seeds.update(new_seed_ids)
            # Seeds for round N use their own propagated_sim as anchor
            seed_props = {tid: propagated_sims[tid] for tid in new_seed_ids}
            round_props, round_tokens = await self._token_hop(
                new_seed_ids,
                seed_props,
                hop_decay,
                strength_threshold,
                exclude_ids=list(propagated_sims.keys()) + seed_ids,
                user_id=user_id,
            )
            all_token_ids.update(round_tokens)
            for tid, prop in round_props.items():
                propagated_sims[tid] = max(propagated_sims.get(tid, 0.0), prop)

            curr_top_k = self._top_k_ids(propagated_sims, top_seeds)
            overlap = len(set(prev_top_k) & set(curr_top_k))
            stability = overlap / len(prev_top_k) if prev_top_k else 0.0
            if stability >= stability_threshold:
                break
            prev_top_k = curr_top_k

        # Reinforce activated tokens
        await self._reinforce_activated_tokens(
            all_token_ids, reinforce_boost
        )

        return propagated_sims

    async def _token_hop(
        self,
        seed_ids: list[str],
        anchor_sims: dict[str, float],
        hop_decay: float,
        strength_threshold: float,
        *,
        exclude_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> tuple[dict[str, float], set[str]]:
        """Single hop of token activation from seed traces.

        Returns (propagated_sims, activated_token_ids).
        """
        try:
            rows = await self._storage.recall_tokens.get_activated_trace_ids(
                seed_ids,
                strength_threshold=strength_threshold,
                user_id=user_id,
            )
        except (StorageError, OSError):
            logger.exception("Token hop activation failed")
            return {}, set()
        exclude = set(exclude_ids) if exclude_ids else set()
        exclude.update(seed_ids)
        propagated: dict[str, float] = {}
        for trace_id, _label, tok_strength, tok_significance, anchor_id in rows:
            if trace_id in exclude:
                continue
            anchor_sim = anchor_sims.get(anchor_id, 0.0)
            prop = anchor_sim * hop_decay * tok_strength * tok_significance
            if prop > propagated.get(trace_id, 0.0):
                propagated[trace_id] = prop
        # Collect actual token IDs for reinforcement
        actual_token_ids: set[str] = set()
        try:
            token_data = await self._storage.recall_tokens.get_tokens_for_traces(
                seed_ids, strength_threshold=strength_threshold
            )
            for token, _stamped in token_data:
                actual_token_ids.add(token.id)
        except (StorageError, OSError):
            logger.exception("Failed to collect token IDs for reinforcement")
        return propagated, actual_token_ids

    @staticmethod
    def _top_k_ids(sims: dict[str, float], k: int) -> list[str]:
        """Return top-k trace IDs by propagated similarity."""
        return sorted(sims, key=lambda tid: sims[tid], reverse=True)[:k]

    async def _reinforce_activated_tokens(
        self, token_ids: set[str], boost: float
    ) -> None:
        """Reinforce activated tokens (Hebbian learning)."""
        if not token_ids:
            return
        try:
            await self._storage.recall_tokens.reinforce_tokens(
                list(token_ids), boost
            )
        except (StorageError, OSError):
            logger.exception("Token reinforcement failed")

    async def _merge_candidates(
        self,
        query_embedding: list[float],
        wm: list[tuple[MemoryTrace, float]],
        storage: list[tuple[MemoryTrace, float]],
        activated: list[tuple[MemoryTrace, float]],
        entity_matches: list[tuple[str, float]] | None = None,
        token_activated: dict[str, float] | None = None,
        *,
        user_id: str | None = None,
    ) -> list[tuple[MemoryTrace, float]]:
        """Merge candidates from all sources using weighted max-score fusion.

        Takes the maximum similarity score a trace receives from any source,
        then adds small bonuses for entity matches and spreading activation.
        Entity bonus is gated by concept similarity.
        """
        entity_bonus = float(self._config.get("retrieval.entity_match_bonus", 0.1))
        spread_bonus = float(self._config.get("activation.spread_weight_factor", 0.1))
        significance_weight = float(
            self._config.get("retrieval.significance_weight", 0.15)
        )
        valence_weight = float(self._config.get("retrieval.valence_weight", 0.05))

        traces_by_id: dict[str, MemoryTrace] = {}
        scores_by_id: dict[str, float] = {}

        for trace, sim in wm:
            traces_by_id[trace.id] = trace
            scores_by_id[trace.id] = max(scores_by_id.get(trace.id, 0.0), sim)

        for trace, sim in storage:
            traces_by_id[trace.id] = trace
            scores_by_id[trace.id] = max(scores_by_id.get(trace.id, 0.0), sim)

        activation_levels = self._collect_activation_levels(
            activated, traces_by_id, scores_by_id, spread_bonus
        )

        entity_sims: dict[str, float] = {}
        if entity_matches:
            for tid, sim in entity_matches:
                entity_sims[tid] = sim

        concept_weight = float(
            self._config.get("retrieval.concept_attention_weight", 0.70)
        )
        concept_sims: dict[str, float] = {}
        if concept_weight > 0.0 and traces_by_id:
            try:
                concept_sims = (
                    await self._storage.concept_embeddings.get_max_sim_per_owner(
                        query_embedding,
                        owner_type="trace",
                        owner_ids=list(traces_by_id.keys()),
                    )
                )
            except (StorageError, OSError):
                logger.exception("Concept attention lookup failed for traces")

        token_bonuses: dict[str, float] = token_activated or {}
        await self._fetch_token_traces(
            query_embedding,
            token_bonuses,
            traces_by_id,
            scores_by_id,
            user_id=user_id,
        )

        propagation_blend = float(
            self._config.get("recall_tokens.propagation_blend", 0.50)
        )

        return self._compute_fused_scores(
            traces_by_id,
            scores_by_id,
            activation_levels,
            entity_sims,
            spread_bonus,
            entity_bonus,
            significance_weight,
            valence_weight,
            concept_sims=concept_sims,
            concept_weight=concept_weight,
            token_bonuses=token_bonuses,
            propagation_blend=propagation_blend,
        )

    @staticmethod
    def _collect_activation_levels(
        activated: list[tuple[MemoryTrace, float]],
        traces_by_id: dict[str, MemoryTrace],
        scores_by_id: dict[str, float],
        spread_bonus: float,
    ) -> dict[str, float]:
        """Collect activation levels, using fraction as base for unseen traces."""
        activation_levels: dict[str, float] = {}
        for trace, level in activated:
            traces_by_id[trace.id] = trace
            activation_levels[trace.id] = max(
                activation_levels.get(trace.id, 0.0), level
            )
            # Traces found only via activation get a fraction as base score
            if trace.id not in scores_by_id:
                scores_by_id[trace.id] = level * spread_bonus
        return activation_levels

    async def _fetch_token_traces(
        self,
        query_embedding: list[float],
        token_bonuses: dict[str, float],
        traces_by_id: dict[str, MemoryTrace],
        scores_by_id: dict[str, float],
        *,
        user_id: str | None = None,
    ) -> None:
        """Fetch token-activated traces not yet in the candidate pool.

        user_id post-filter: get_traces_bulk fetches by ID with no user
        predicate; guard the pool even if a caller feeds unscoped IDs.
        """
        if not token_bonuses:
            return
        missing = [tid for tid in token_bonuses if tid not in traces_by_id]
        if not missing:
            return
        try:
            token_traces = await self._storage.traces.get_traces_bulk(missing)
            for t in token_traces:
                if user_id is not None and t.user_id != user_id:
                    continue
                traces_by_id[t.id] = t
                if t.embedding is not None:
                    scores_by_id[t.id] = _cosine_similarity(
                        query_embedding, t.embedding
                    )
        except (StorageError, OSError):
            logger.exception("Failed to fetch token-activated traces")

    @staticmethod
    def _compute_fused_scores(
        traces_by_id: dict[str, MemoryTrace],
        scores_by_id: dict[str, float],
        activation_levels: dict[str, float],
        entity_sims: dict[str, float],
        spread_bonus: float,
        entity_bonus: float,
        significance_weight: float = 0.15,
        valence_weight: float = 0.05,
        *,
        concept_sims: dict[str, float] | None = None,
        concept_weight: float = 0.0,
        token_bonuses: dict[str, float] | None = None,
        propagation_blend: float = 0.50,
    ) -> list[tuple[MemoryTrace, float]]:
        """Compute fused scores with concept-primary blend, clamped to [0, 1].

        When concept embeddings exist for a trace, base cosine similarity
        is blended with concept max-sim (weight*concept + (1-weight)*base).
        This mirrors attention: concept scores modulate rather than add.
        Entity bonus remains gated by concept similarity.
        """
        if concept_sims is None:
            concept_sims = {}
        result: list[tuple[MemoryTrace, float]] = []
        for trace_id, base in scores_by_id.items():
            trace = traces_by_id.get(trace_id)
            if trace is None:
                continue
            # Concept-primary blend: concept attention modulates base
            # similarity instead of adding to it. When concept embeddings
            # exist, effective_sim = weight*concept + (1-weight)*base.
            concept_sim = concept_sims.get(trace_id, 0.0)
            if concept_sim > 0.0 and concept_weight > 0.0:
                effective_sim = (
                    concept_weight * concept_sim + (1.0 - concept_weight) * base
                )
            else:
                effective_sim = base
            significance_boost = trace.significance * significance_weight
            valence_boost = abs(trace.emotional_valence) * valence_weight
            score = effective_sim + significance_boost + valence_boost
            if trace_id in activation_levels:
                score += activation_levels[trace_id] * spread_bonus
            if trace_id in entity_sims:
                score += (
                    entity_sims[trace_id]
                    * entity_bonus
                    * trace.significance
                    * concept_sim
                )
            if token_bonuses and trace_id in token_bonuses:
                score += token_bonuses[trace_id] * propagation_blend
            # Lower clamp is load-bearing: _fetch_token_traces seeds raw
            # cosine (negative for anti-correlated), and Thought.relevance
            # declares ge=0 -- unclamped, one bad candidate fails the recall.
            score = min(max(score, 0.0), 1.0)
            result.append((trace, score))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _select_within_budget(
        self,
        candidates: list[tuple[MemoryTrace, float]],
        token_budget: int,
    ) -> list[tuple[MemoryTrace, float]]:
        """Select top candidates that fit within token budget."""
        selected: list[tuple[MemoryTrace, float]] = []
        tokens_used = 0
        max_retrievals = int(self._config.get("retrieval.max_retrievals", 3))

        for trace, score in candidates:
            if len(selected) >= max_retrievals:
                break
            tokens = _estimate_tokens(trace.content or "")
            if tokens_used + tokens > token_budget:
                break
            selected.append((trace, score))
            tokens_used += tokens

        return selected

    async def _boost_and_build_thoughts(
        self, selected: list[tuple[MemoryTrace, float]]
    ) -> list[Thought]:
        """Apply retrieval boost to selected traces and build Thoughts."""
        thoughts: list[Thought] = []
        for trace, relevance in selected:
            in_wm = self._buffer.find(lambda t, _id=trace.id: t.id == _id) is not None
            boosted = apply_retrieval_boost(trace, from_working_memory=in_wm)
            await self._storage.traces.update_trace_strength(trace.id, boosted.strength)
            await self._storage.traces.mark_retrieved(trace.id)

            content = trace.content or str(trace.pattern)
            thoughts.append(
                Thought(
                    trace=trace,
                    relevance=relevance,
                    token_count=_estimate_tokens(content),
                    reconstructed=trace.content is None,
                    reconstruction=content,
                )
            )
        return thoughts

    async def _boost_unselected(
        self,
        all_candidates: list[tuple[MemoryTrace, float]],
        thoughts: list[Thought],
    ) -> None:
        """Apply small activation boost to candidates not selected."""
        selected_ids = {t.trace.id for t in thoughts}
        for trace, _ in all_candidates:
            if trace.id not in selected_ids:
                boosted = apply_activation_boost(trace)
                await self._storage.traces.update_trace_strength(
                    trace.id, boosted.strength
                )
                await self._storage.traces.mark_activated(trace.id)

    async def _match_query_entities(
        self, query: str, *, user_id: str | None = None
    ) -> list[tuple[str, float]]:
        """Find trace IDs for entities mentioned in the query.

        Extracts candidate entity names from query text, then uses
        trigram similarity matching in storage.
        """
        names = self._extract_entity_names_from_query(query)
        if not names:
            return []
        try:
            return await self._storage.entities.match_entities(
                names, user_id=user_id
            )
        except (StorageError, OSError):
            logger.exception("Entity matching failed for query")
            return []

    async def surface_relevant_facts(
        self,
        trace: MemoryTrace,
        *,
        user_id: str | None = None,
    ) -> list[PersonaFact]:
        """Persona facts relevant to a just-written trace, for write-time return.

        Read-only: surfacing is not mentioning, so no mention_count or recency
        write. Safety facts (health/dietary/constraint) surface by domain-gating
        the trace's extracted domains against `_DOMAIN_SAFETY_MAP` -- no cosine.
        Non-safety facts surface by semantic/entity relevance above
        `persona.surface_relevance_floor`, capped at `persona.max_surfaced_facts`.
        Safety facts are listed first.
        """
        safety = await self._surface_safety_facts(trace, user_id=user_id)
        non_safety = await self._surface_non_safety_facts(trace, user_id=user_id)
        return self._deduplicate_facts(safety + non_safety)

    async def _surface_safety_facts(
        self, trace: MemoryTrace, *, user_id: str | None
    ) -> list[PersonaFact]:
        safety_categories: set[str] = set()
        for domain in trace.pattern.get("domains", []):
            safety_categories |= _DOMAIN_SAFETY_MAP.get(domain, frozenset())
        if not safety_categories:
            return []
        facts = await self._storage.facts.get_persona_facts(user_id=user_id)
        return [
            f
            for f in facts
            if f.status in ("promoted", "pinned")
            and f.category in _FAST_TRACK_CATEGORIES
            and f.category in safety_categories
        ]

    async def _surface_non_safety_facts(
        self, trace: MemoryTrace, *, user_id: str | None
    ) -> list[PersonaFact]:
        if not trace.content:
            return []
        floor = float(self._config.get("persona.surface_relevance_floor", 0.5))
        cap = int(self._config.get("persona.max_surfaced_facts", 3))
        # nomic retrieval is asymmetric: the trace is the QUERY here, so embed it
        # search_query. Reusing trace.embedding (search_document, write-side) is
        # the wrong mode and inflates the baseline against search_document facts.
        try:
            query_embedding = await self._embeddings.generate_embedding(
                trace.content, task="search_query"
            )
        except EmbeddingError:
            return []
        facts, scores = await self._find_relevant_persona_facts(
            trace.content, query_embedding, user_id=user_id
        )
        relevant = [
            f
            for f in facts
            if f.category not in _FAST_TRACK_CATEGORIES
            and scores.get(f.id, 0.0) >= floor
        ]
        return relevant[:cap]

    async def _find_relevant_persona_facts(
        self,
        query: str,
        query_embedding: list[float],
        *,
        user_id: str | None = None,
    ) -> tuple[list[PersonaFact], dict[str, float]]:
        """Find persona facts via semantic search + entity supplement.

        Promoted/pinned facts bypass the similarity gate and always enter
        the candidate pool. Ranking and slot limits handle relevance --
        irrelevant facts score low and get cut by rank, not by threshold.

        Scoring uses concept attention (ColBERT-style max-sim on LLM-extracted
        tags, 0.7 weight) blended with bi-encoder similarity (0.3 weight).

        Primary: embed(query) vs embed(fact.context) via pgvector.
        Supplement: entity name match adds structurally-related facts.
        Supplement: all promoted/pinned facts (no similarity gate).
        """
        try:
            entity_names = self._extract_entity_names_from_query(query)
            max_facts = int(self._config.get("persona.max_facts_per_query", 5))

            # Primary: semantic search on fact content embeddings
            semantic_results = await self._storage.facts.search_facts_semantic(
                query_embedding,
                limit=max_facts * 2,
                user_id=user_id,
            )
            # All promoted/pinned semantic results enter the pool
            semantic_facts = [
                f for f, _ in semantic_results
                if f.status in ("promoted", "pinned")
            ]
            semantic_scores: dict[str, float] = {
                f.id: sim for f, sim in semantic_results
                if f.status in ("promoted", "pinned")
            }

            # Supplement: entity name match (adds facts for named entities)
            entity_facts = await self._storage.facts.get_persona_facts_by_entities(
                entity_names,
                user_id=user_id,
            )

            # Supplement: all promoted/pinned facts bypass the similarity gate
            all_user_facts = await self._storage.facts.get_persona_facts(
                user_id=user_id,
            )
            promoted_facts = [
                f for f in all_user_facts
                if f.status in ("promoted", "pinned")
            ]

            all_facts = self._deduplicate_facts(
                semantic_facts + entity_facts + promoted_facts
            )

            # Score facts not yet scored via direct cosine similarity
            for fact in all_facts:
                if fact.id not in semantic_scores and fact.embedding:
                    semantic_scores[fact.id] = _cosine_similarity(
                        query_embedding, fact.embedding
                    )
                elif fact.id not in semantic_scores:
                    semantic_scores[fact.id] = 0.0

            # Concept attention: blend concept and bi-encoder scores
            # Concept is primary (0.7) because LLM wrote tags as query predictions.
            # Bi-encoder is fallback (0.3) for general semantic similarity.
            try:
                fact_ids = [f.id for f in all_facts]
                if fact_ids:
                    fact_concept_sims = (
                        await self._storage.concept_embeddings.get_max_sim_per_owner(
                            query_embedding,
                            owner_type="fact",
                            owner_ids=fact_ids,
                        )
                    )
                    for fid, csim in fact_concept_sims.items():
                        bi_sim = semantic_scores.get(fid, 0.0)
                        semantic_scores[fid] = 0.7 * csim + 0.3 * bi_sim
            except (StorageError, OSError):
                logger.exception("Fact concept attention lookup failed")

            return (
                self._rank_and_limit_facts(all_facts, max_facts, semantic_scores),
                semantic_scores,
            )
        except (StorageError, OSError):
            logger.exception("Persona fact lookup failed")
            return [], {}

    @staticmethod
    def _deduplicate_facts(facts: list[PersonaFact]) -> list[PersonaFact]:
        """Remove duplicate facts by ID, preserving order."""
        seen: set[str] = set()
        unique: list[PersonaFact] = []
        for fact in facts:
            if fact.id not in seen:
                seen.add(fact.id)
                unique.append(fact)
        return unique

    @staticmethod
    def _rank_and_limit_facts(
        facts: list[PersonaFact],
        limit: int,
        semantic_scores: dict[str, float] | None = None,
    ) -> list[PersonaFact]:
        """Rank facts by semantic similarity to the query, then limit."""

        def sort_key(f: PersonaFact) -> tuple[float, float]:
            sim = (semantic_scores or {}).get(f.id, 0.0)
            return (-sim, -f.confidence)

        return sorted(facts, key=sort_key)[:limit]


    def _extract_entity_names_from_query(self, query: str) -> list[str]:
        """Extract potential entity names from query text.

        Capitalized words not in common stop words are treated as
        potential entity names (proper nouns).
        """
        words = query.split()
        entities: list[str] = []
        for word in words:
            cleaned = word.strip(".,!?;:'\"")
            if (
                cleaned
                and cleaned[0].isupper()
                and cleaned.lower() not in _QUERY_STOP_WORDS
            ):
                entities.append(cleaned)
        return entities

    def _persona_facts_to_thoughts(
        self,
        facts: list[PersonaFact],
        semantic_scores: dict[str, float] | None = None,
    ) -> list[Thought]:
        """Convert persona facts to Thought objects using ranking strategy."""
        strategy = str(self._config.get("persona.ranking_strategy", "hybrid"))
        threshold = float(self._config.get("persona.confidence_threshold", 0.6))
        thoughts: list[Thought] = []
        for fact in facts:
            relevance = self._compute_fact_relevance(
                fact,
                (semantic_scores or {}).get(fact.id, 0.0),
            )
            triple = f"{fact.subject} {fact.predicate} {fact.object}"
            if fact.content and fact.content != triple:
                content = f"[IMPORTANT CONTEXT] {triple} -- {fact.content}"
            else:
                content = f"[IMPORTANT CONTEXT] {triple}"
            trace = MemoryTrace(
                content=content,
                pattern={"persona_fact": True, "category": fact.category},
                strength=1.0,
                significance=1.0,
            )
            if strategy == "pinned":
                pinned = True
            elif strategy == "relevance":
                pinned = False
            else:  # hybrid
                pinned = relevance >= threshold
            thoughts.append(
                Thought(
                    trace=trace,
                    relevance=relevance,
                    token_count=_estimate_tokens(content),
                    reconstructed=True,
                    reconstruction=content,
                    pinned=pinned,
                )
            )
        return thoughts

    @staticmethod
    def _compute_fact_relevance(
        fact: PersonaFact,
        semantic_similarity: float = 0.0,
    ) -> float:
        """Compute fact relevance blending confidence with semantic similarity."""
        if semantic_similarity > 0.0:
            return 0.3 * fact.confidence + 0.7 * semantic_similarity
        return fact.confidence

    # -- Private: consolidation helpers --

    async def _consolidate_one(
        self,
        trace: MemoryTrace,
        decayed: MemoryTrace,
        threshold: float,
        grace_hours: float,
    ) -> str:
        """Consolidate a single trace.

        Returns 'consolidated', 'forgotten', or 'pending'.
        """
        if decayed.strength >= threshold:
            await self._storage.traces.update_trace_strength(trace.id, decayed.strength)
            await self._storage.traces.mark_consolidated(trace.id)
            return "consolidated"

        now = memory_timestamp_for_comparison(now_utc())
        created = memory_timestamp_for_comparison(trace.created_at)
        age_hours = (now - created).total_seconds() / 3600.0

        if age_hours >= grace_hours:
            await self._delete_trace_with_buffer_evict(trace.id)
            return "forgotten"

        await self._storage.traces.update_trace_strength(trace.id, decayed.strength)
        return "pending"
