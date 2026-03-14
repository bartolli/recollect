"""Data models and cognitive strength functions for Memory SDK."""

from __future__ import annotations

import math
import uuid
from datetime import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field

from recollect.config import config
from recollect.datetime_utils import (
    memory_timestamp_for_comparison,
    memory_timestamp_for_storage,
    now_utc,
)


class MemoryTrace(BaseModel):
    """A memory trace -- patterns extracted from experience, not raw content."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str | None = None
    pattern: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    strength: float = Field(default=0.3, ge=0.0, le=1.0)
    activation_count: int = Field(default=0, ge=0)
    retrieval_count: int = Field(default=0, ge=0)
    last_activation: dt | None = None
    last_retrieval: dt | None = None
    consolidated: bool = False
    created_at: dt = Field(default_factory=memory_timestamp_for_storage)
    decay_rate: float = Field(default=0.1, ge=0.0)
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    significance: float = Field(default=0.1, ge=0.0, le=1.0)
    session_id: str | None = None
    user_id: str | None = None

    @property
    def confidence(self) -> str:
        """Human-readable confidence label derived from strength."""
        if self.strength >= 0.8:
            return "vivid"
        if self.strength >= 0.5:
            return "clear"
        if self.strength >= 0.2:
            return "fuzzy"
        return "fading"

    model_config = {"arbitrary_types_allowed": True}


class Thought(BaseModel):
    """A reconstructed thought from a memory trace."""

    trace: MemoryTrace
    relevance: float = Field(default=0.0, ge=0.0)
    token_count: int = Field(default=0, ge=0)
    reconstructed: bool = False
    reconstruction: str = ""
    pinned: bool = False


class Association(BaseModel):
    """A connection between two memory traces."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_trace_id: str
    target_trace_id: str
    association_type: str = "semantic"
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    forward_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    backward_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    activation_count: int = Field(default=0, ge=0)
    last_activation: dt | None = None
    created_at: dt = Field(default_factory=memory_timestamp_for_storage)


FactStatus = Literal["candidate", "promoted", "pinned"]

FactCategory = Literal[
    "health",
    "dietary",
    "identity",
    "relationship",
    "preference",
    "schedule",
    "constraint",
    "general",
]

SessionStatus = Literal["active", "ended", "summarized"]


class PersonaFact(BaseModel):
    """A permanent fact stored as an SPO (subject-predicate-object) triple."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str
    predicate: str
    object: str
    category: FactCategory = "general"
    content: str
    source_trace_id: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    created_at: dt = Field(default_factory=memory_timestamp_for_storage)
    updated_at: dt = Field(default_factory=memory_timestamp_for_storage)
    superseded_by: str | None = None
    status: FactStatus = "candidate"
    mention_count: int = Field(default=1, ge=1)
    scope: str = "general"
    context_tags: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    user_id: str | None = None


class TraceEntity(BaseModel):
    """Links an entity occurrence to a memory trace."""

    entity_name: str
    entity_type: str = "unknown"
    trace_id: str


class TraceConcept(BaseModel):
    """Links a concept to a memory trace."""

    concept: str
    trace_id: str


class ConceptEmbedding(BaseModel):
    """A concept's embedding vector, linked to a trace or fact."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    concept: str
    owner_type: Literal["trace", "fact"]
    owner_id: str
    embedding: list[float]
    created_at: dt = Field(default_factory=memory_timestamp_for_storage)


class RecallToken(BaseModel):
    """A situational recall token linking causally related memory traces.

    Tokens are created at write-time when the LLM detects situational
    dependencies between memories. Four lifecycle actions manage tokens:
    create, extend, revise, none. Strength decays without reinforcement
    and increases on recall (Hebbian learning). Significance reflects
    real-world importance (health/safety=high, trivia=low).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    significance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: dt = Field(default_factory=memory_timestamp_for_storage)
    last_activated_at: dt = Field(default_factory=memory_timestamp_for_storage)
    status: Literal["active", "archived"] = "active"


class TokenStamp(BaseModel):
    """Links a recall token to a memory trace (many-to-many)."""

    token_id: str
    trace_id: str
    stamped_at: dt = Field(default_factory=memory_timestamp_for_storage)


class EntityRelation(BaseModel):
    """A typed relationship between two entities with disambiguation context."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_entity: str
    relation: str
    target_entity: str
    context: str = ""
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    source_trace_id: str | None = None
    created_at: dt = Field(default_factory=memory_timestamp_for_storage)


class ConsolidationResult(BaseModel):
    """Result of a consolidation pass."""

    consolidated: int = 0
    forgotten: int = 0
    still_pending: int = 0


class Session(BaseModel):
    """A conversation session grouping memory traces."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str = ""
    status: SessionStatus = "active"
    summary_trace_id: str | None = None
    created_at: dt = Field(default_factory=memory_timestamp_for_storage)
    ended_at: dt | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryStats(BaseModel):
    """Memory system statistics."""

    working_memory_items: int = Field(default=0, ge=0)
    working_memory_capacity: int = Field(default=7, ge=1)
    working_memory_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    total_seen: int = Field(default=0, ge=0)
    total_displaced: int = Field(default=0, ge=0)
    displacement_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    connected: bool = False


class HealthStatus(BaseModel):
    """System health status."""

    status: str = "disconnected"


# -- Strength functions --
# These are pure functions that return new MemoryTrace instances.
# All multipliers come from the cognitive model parameters.


def _clamp_strength(value: float) -> float:
    max_strength = float(config.get("strengthening.max_strength", 1.0))
    return max(0.0, min(max_strength, value))


def apply_activation_boost(trace: MemoryTrace) -> MemoryTrace:
    """Apply small boost for spreading activation (x1.01 default)."""
    factor = 1.0 + float(config.get("retrieval.activation_boost", 0.01))
    return trace.model_copy(
        update={"strength": _clamp_strength(trace.strength * factor)}
    )


def apply_retrieval_boost(
    trace: MemoryTrace, *, from_working_memory: bool = False
) -> MemoryTrace:
    """Apply boost for conscious retrieval.

    Working memory retrieval gets a stronger boost (x1.2) than
    long-term retrieval (x1.1).
    """
    if from_working_memory:
        factor = 1.0 + float(config.get("retrieval.wm_retrieval_boost", 0.2))
    else:
        factor = 1.0 + float(config.get("retrieval.retrieval_boost", 0.1))
    return trace.model_copy(
        update={"strength": _clamp_strength(trace.strength * factor)}
    )


def apply_displacement_decay(trace: MemoryTrace) -> MemoryTrace:
    """Apply decay when pushed out of working memory (x0.8 default)."""
    factor = float(config.get("forgetting.displacement_decay", 0.8))
    return trace.model_copy(
        update={"strength": _clamp_strength(trace.strength * factor)}
    )


def apply_time_decay(trace: MemoryTrace) -> MemoryTrace:
    """Apply exponential time decay based on age.

    Formula: strength * exp(-decay_rate * hours_since_creation)
    """
    now = memory_timestamp_for_comparison(now_utc())
    created = memory_timestamp_for_comparison(trace.created_at)
    hours_passed = (now - created).total_seconds() / 3600.0

    decay_rate = trace.decay_rate or float(config.get("memory.decay_rate", 0.1))
    decay_factor = math.exp(-decay_rate * hours_passed)

    return trace.model_copy(
        update={"strength": _clamp_strength(trace.strength * decay_factor)}
    )
