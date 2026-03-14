"""Conversion utilities for storage backends.

Pure functions that convert between Pydantic models and database row dicts.
Used by all PostgreSQL sub-store implementations.
"""

from __future__ import annotations

import json
from typing import Any

from recollect.models import (
    Association,
    ConceptEmbedding,
    MemoryTrace,
    PersonaFact,
    Session,
)


def embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert a float list to pgvector string format."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


def trace_to_params(trace: MemoryTrace) -> dict[str, Any]:
    """Convert MemoryTrace to a dict of SQL parameters."""
    embedding: str | None = None
    if trace.embedding is not None:
        embedding = embedding_to_pgvector(trace.embedding)

    return {
        "id": trace.id,
        "content": trace.content,
        "pattern": json.dumps(trace.pattern),
        "context": json.dumps(trace.context),
        "embedding": embedding,
        "strength": trace.strength,
        "activation_count": trace.activation_count,
        "retrieval_count": trace.retrieval_count,
        "last_activation": trace.last_activation,
        "last_retrieval": trace.last_retrieval,
        "consolidated": trace.consolidated,
        "created_at": trace.created_at,
        "decay_rate": trace.decay_rate,
        "emotional_valence": trace.emotional_valence,
        "significance": trace.significance,
        "session_id": trace.session_id,
        "user_id": trace.user_id,
    }


def row_to_trace(row: dict[str, Any]) -> MemoryTrace:
    """Convert a database row dict to a MemoryTrace."""
    pattern = row.get("pattern", "{}")
    if isinstance(pattern, str):
        pattern = json.loads(pattern)

    context = row.get("context", "{}")
    if isinstance(context, str):
        context = json.loads(context)

    # pgvector returns embeddings as strings like "[0.1,0.2,...]"
    raw_embedding = row.get("embedding")
    embedding: list[float] | None = None
    if isinstance(raw_embedding, str):
        embedding = json.loads(raw_embedding)
    elif isinstance(raw_embedding, list):
        embedding = raw_embedding

    return MemoryTrace(
        id=row["id"],
        content=row.get("content"),
        pattern=pattern,
        context=context,
        embedding=embedding,
        strength=row.get("strength", 0.3),
        activation_count=row.get("activation_count", 0),
        retrieval_count=row.get("retrieval_count", 0),
        last_activation=row.get("last_activation"),
        last_retrieval=row.get("last_retrieval"),
        consolidated=row.get("consolidated", False),
        created_at=row["created_at"],
        decay_rate=row.get("decay_rate", 0.1),
        emotional_valence=row.get("emotional_valence", 0.0),
        significance=row.get("significance", 0.1),
        session_id=row.get("session_id"),
        user_id=row.get("user_id"),
    )


def association_to_params(assoc: Association) -> dict[str, Any]:
    """Convert Association to a dict of SQL parameters."""
    return {
        "id": assoc.id,
        "source_trace_id": assoc.source_trace_id,
        "target_trace_id": assoc.target_trace_id,
        "association_type": assoc.association_type,
        "weight": assoc.weight,
        "forward_strength": assoc.forward_strength,
        "backward_strength": assoc.backward_strength,
        "activation_count": assoc.activation_count,
        "last_activation": assoc.last_activation,
        "created_at": assoc.created_at,
    }


def row_to_association(row: dict[str, Any]) -> Association:
    """Convert a database row dict to an Association."""
    return Association(
        id=row["id"],
        source_trace_id=row["source_trace_id"],
        target_trace_id=row["target_trace_id"],
        association_type=row.get("association_type", "semantic"),
        weight=row.get("weight", 0.5),
        forward_strength=row.get("forward_strength", 0.5),
        backward_strength=row.get("backward_strength", 0.5),
        activation_count=row.get("activation_count", 0),
        last_activation=row.get("last_activation"),
        created_at=row["created_at"],
    )


def persona_fact_to_params(fact: PersonaFact) -> dict[str, Any]:
    """Convert PersonaFact to a dict of SQL parameters."""
    return {
        "id": fact.id,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "category": fact.category,
        "content": fact.content,
        "source_trace_id": fact.source_trace_id,
        "confidence": fact.confidence,
        "created_at": fact.created_at,
        "updated_at": fact.updated_at,
        "superseded_by": fact.superseded_by,
        "status": fact.status,
        "mention_count": fact.mention_count,
        "scope": fact.scope,
        "context_tags": fact.context_tags,
        "embedding": (
            embedding_to_pgvector(fact.embedding)
            if fact.embedding is not None
            else None
        ),
        "user_id": fact.user_id,
    }


def row_to_persona_fact(row: dict[str, Any]) -> PersonaFact:
    """Convert a database row dict to a PersonaFact."""
    # pgvector returns embeddings as strings like "[0.1,0.2,...]"
    raw_embedding = row.get("embedding")
    embedding: list[float] | None = None
    if isinstance(raw_embedding, str):
        embedding = json.loads(raw_embedding)
    elif isinstance(raw_embedding, list):
        embedding = raw_embedding

    return PersonaFact(
        id=row["id"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        category=row.get("category", "general"),
        content=row["content"],
        source_trace_id=row.get("source_trace_id"),
        confidence=row.get("confidence", 0.8),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        superseded_by=row.get("superseded_by"),
        status=row.get("status", "candidate"),
        mention_count=row.get("mention_count", 1),
        scope=row.get("scope", "general"),
        context_tags=row.get("context_tags") or [],
        embedding=embedding,
        user_id=row.get("user_id"),
    )


def concept_embedding_to_params(ce: ConceptEmbedding) -> dict[str, Any]:
    """Convert ConceptEmbedding to a dict of SQL parameters."""
    return {
        "id": ce.id,
        "concept": ce.concept,
        "owner_type": ce.owner_type,
        "owner_id": ce.owner_id,
        "embedding": embedding_to_pgvector(ce.embedding),
        "created_at": ce.created_at,
    }


def row_to_concept_embedding(row: dict[str, Any]) -> ConceptEmbedding:
    """Convert a database row dict to a ConceptEmbedding."""
    raw_embedding = row.get("embedding")
    embedding: list[float] = []
    if isinstance(raw_embedding, str):
        embedding = json.loads(raw_embedding)
    elif isinstance(raw_embedding, list):
        embedding = raw_embedding

    return ConceptEmbedding(
        id=row["id"],
        concept=row["concept"],
        owner_type=row["owner_type"],
        owner_id=row["owner_id"],
        embedding=embedding,
        created_at=row["created_at"],
    )


def session_to_params(session: Session) -> dict[str, Any]:
    """Convert Session to a dict of SQL parameters."""
    return {
        "id": session.id,
        "user_id": session.user_id,
        "title": session.title,
        "status": session.status,
        "summary_trace_id": session.summary_trace_id,
        "created_at": session.created_at,
        "ended_at": session.ended_at,
        "metadata": json.dumps(session.metadata),
    }


def row_to_session(row: dict[str, Any]) -> Session:
    """Convert a database row dict to a Session."""
    metadata = row.get("metadata", "{}")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return Session(
        id=row["id"],
        user_id=row["user_id"],
        title=row.get("title", ""),
        status=row.get("status", "active"),
        summary_trace_id=row.get("summary_trace_id"),
        created_at=row["created_at"],
        ended_at=row.get("ended_at"),
        metadata=metadata,
    )
