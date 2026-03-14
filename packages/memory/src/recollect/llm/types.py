"""Core types for LLM provider abstraction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    """A message in a conversation with an LLM."""

    role: Literal["system", "user", "assistant"]
    content: str


class CompletionParams(BaseModel):
    """LLM completion parameters with passthrough support.

    Base parameters (max_tokens, temperature) are defined explicitly.
    Additional provider-specific parameters can be passed via extra fields
    (e.g., top_p, presence_penalty) and will be forwarded to the API.
    """

    model_config = ConfigDict(extra="allow")

    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)


class Entity(BaseModel):
    """A named entity extracted from text."""

    name: str
    entity_type: str = "unknown"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class Relation(BaseModel):
    """A relationship between entities or concepts."""

    source: str
    relation: str
    target: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    category: str = "general"
    context: str = ""
    context_tags: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Structured patterns extracted from text by an LLM."""

    concepts: list[str] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    significance: float = Field(default=0.1, ge=0.0, le=1.0)
    fact_type: Literal["episodic", "semantic"] = "episodic"

    @field_validator("fact_type", mode="before")
    @classmethod
    def _normalize_fact_type(cls, v: object) -> str:
        """Strip whitespace and trailing punctuation from fact_type.

        Anthropic's structured output moves enum constraints to description
        hints, so the model occasionally produces values like 'semantic, '
        instead of 'semantic'.
        """
        if isinstance(v, str):
            return v.strip().rstrip(",")
        return str(v)


class TokenAssessment(BaseModel):
    """LLM assessment of situational group action for a new memory.

    Four actions manage the token lifecycle:
    - create: new situational group linking causally related memories
    - extend: new memory joins an existing group with a new implication
    - revise: new memory changes the factual basis of an existing group
    - none: no situational connection (default, expected majority)
    """

    action: Literal["create", "extend", "revise", "none"] = "none"
    group_number: int = Field(default=0, ge=0)
    person_ref: str = ""
    situation: str = ""
    implication: str = ""
    significance: float = Field(default=0.5, ge=0.0, le=1.0)
    linked_indices: list[int] = Field(default_factory=list)
