"""Protocol definitions for Memory SDK components.

These protocols define the contracts that storage backends, embedding
providers, and extractors must implement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from recollect.models import (
        Association,
        ConceptEmbedding,
        EntityRelation,
        MemoryTrace,
        PersonaFact,
        RecallToken,
        Session,
        TraceConcept,
        TraceEntity,
    )


class TraceStore(Protocol):
    """Store and retrieve memory traces."""

    async def store_trace(self, trace: MemoryTrace) -> str: ...
    async def get_trace(self, trace_id: str) -> MemoryTrace | None: ...
    async def get_traces_bulk(self, trace_ids: list[str]) -> list[MemoryTrace]: ...
    async def delete_trace(self, trace_id: str) -> bool: ...
    async def update_trace_strength(
        self, trace_id: str, new_strength: float
    ) -> None: ...
    async def mark_activated(self, trace_id: str) -> None: ...
    async def mark_retrieved(self, trace_id: str) -> None: ...
    async def get_recent_traces(self, limit: int = 20) -> list[MemoryTrace]: ...
    async def get_unconsolidated_traces(self, limit: int = 50) -> list[MemoryTrace]: ...
    async def mark_consolidated(self, trace_id: str) -> None: ...
    async def get_traces_by_session(
        self,
        session_id: str,
        *,
        limit: int = 100,
    ) -> list[MemoryTrace]: ...


class VectorIndex(Protocol):
    """Semantic search and spreading activation."""

    async def search_semantic(
        self,
        query_embedding: list[float],
        limit: int = 10,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[tuple[MemoryTrace, float]]: ...

    async def spread_activation(
        self, seed_id: str, max_depth: int = 2
    ) -> list[tuple[MemoryTrace, float]]:
        """Return traces with their activation levels."""
        ...


class EntityIndex(Protocol):
    """Entity and concept indexing for traces."""

    async def store_trace_entities(
        self, trace_id: str, entities: list[TraceEntity]
    ) -> None: ...

    async def store_trace_concepts(
        self, trace_id: str, concepts: list[TraceConcept]
    ) -> None: ...

    async def get_traces_by_entity(
        self, entity_name: str, *, limit: int = 20
    ) -> list[str]: ...

    async def get_traces_by_concept(
        self, concept: str, *, limit: int = 20
    ) -> list[str]: ...

    async def match_entities(
        self, names: list[str], *, limit: int = 20
    ) -> list[tuple[str, float]]:
        """Match entity names by trigram similarity."""
        ...


class AssociationStore(Protocol):
    """Store and retrieve trace associations."""

    async def store_association(self, association: Association) -> None: ...
    async def get_associations(self, trace_id: str) -> list[Association]: ...


class EntityRelationStore(Protocol):
    """Store and traverse entity-to-entity relationships."""

    async def store_relation(self, rel: EntityRelation) -> str: ...
    async def get_relations(
        self, entity_name: str, *, limit: int = 20
    ) -> list[EntityRelation]: ...
    async def get_related_entities(
        self, entity_name: str, *, max_depth: int = 3, limit: int = 50
    ) -> list[str]: ...
    async def delete_by_trace(self, trace_id: str) -> int: ...


class ConceptEmbeddingStore(Protocol):
    """Store and retrieve concept-level embedding vectors."""

    async def store_concept_embeddings(
        self,
        embeddings: list[ConceptEmbedding],
    ) -> None: ...

    async def get_max_sim_per_owner(
        self,
        query_embedding: list[float],
        *,
        owner_type: str,
        owner_ids: list[str],
    ) -> dict[str, float]: ...

    async def delete_by_owner(
        self,
        owner_type: str,
        owner_id: str,
    ) -> int: ...


class RecallTokenStore(Protocol):
    """Store and retrieve recall tokens and their trace stamps."""

    async def create_token(self, token: RecallToken) -> str:
        """Insert a recall token, returning its ID."""
        ...

    async def stamp_traces(self, token_id: str, trace_ids: list[str]) -> None:
        """Stamp a token onto one or more traces."""
        ...

    async def update_token_label(self, token_id: str, new_label: str) -> None:
        """Update an existing token's label."""
        ...

    async def update_token(
        self, token_id: str, new_label: str, significance: float
    ) -> None:
        """Update an existing token's label and significance."""
        ...

    async def find_groups_for_traces(
        self,
        trace_ids: list[str],
        *,
        strength_threshold: float = 0.1,
        include_archived: bool = False,
    ) -> list[dict[str, object]]:
        """Find existing token groups linked to any of the given traces.

        When include_archived is True, also returns archived token groups
        so that extend/revise actions can reference and reactivate them.
        """
        ...

    async def get_tokens_for_traces(
        self, trace_ids: list[str], *, strength_threshold: float = 0.1
    ) -> list[tuple[RecallToken, list[str]]]:
        """Find active tokens linked to any of the given traces.

        Returns (token, list_of_stamped_trace_ids) pairs for tokens
        above the strength threshold.
        """
        ...

    async def get_activated_trace_ids(
        self,
        seed_trace_ids: list[str],
        *,
        strength_threshold: float = 0.1,
    ) -> list[tuple[str, str, float, float, str]]:
        """One-hop token activation from seed traces.

        Returns (trace_id, token_label, token_strength, token_significance,
        anchor_trace_id) for traces linked via shared tokens but NOT in
        the seed set.
        """
        ...

    async def find_token_by_traces(
        self, trace_ids: list[str]
    ) -> RecallToken | None:
        """Find an existing token that stamps ALL of the given trace IDs.

        Returns the strongest matching token, or None.
        """
        ...

    async def reinforce_tokens(
        self, token_ids: list[str], boost: float = 0.1
    ) -> None:
        """Increment strength of activated tokens (Hebbian reinforcement)."""
        ...

    async def delete_by_trace(self, trace_id: str) -> int:
        """Remove all token stamps for a trace. Returns count removed."""
        ...

    async def decay_inactive(
        self, decay_factor: float, *, min_strength: float = 0.01
    ) -> int:
        """Decay tokens not activated since last consolidation.

        Multiplies strength by decay_factor for all tokens. Archives tokens
        that fall below min_strength (sets status='archived'). Returns count
        decayed.
        """
        ...


class SessionStore(Protocol):
    """Store and manage conversation sessions."""

    async def create_session(self, session: Session) -> str: ...

    async def get_session(self, session_id: str) -> Session | None: ...

    async def end_session(
        self,
        session_id: str,
        *,
        summary_trace_id: str | None = None,
    ) -> None: ...

    async def get_sessions(
        self,
        user_id: str,
        *,
        limit: int = 50,
    ) -> list[Session]: ...

    async def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
        summary_trace_id: str | None = None,
    ) -> None: ...


class FactStore(Protocol):
    """Store and retrieve persona facts."""

    async def store_persona_fact(self, fact: PersonaFact) -> str: ...

    async def get_persona_facts(
        self,
        subject: str | None = None,
        *,
        limit: int = 50,
        user_id: str | None = None,
    ) -> list[PersonaFact]: ...

    async def get_persona_facts_by_entities(
        self,
        entity_names: list[str],
        *,
        user_id: str | None = None,
    ) -> list[PersonaFact]: ...

    async def get_facts_by_entities_and_scopes(
        self,
        entity_names: list[str],
        scopes: list[str],
        *,
        limit: int = 20,
    ) -> list[PersonaFact]: ...

    async def supersede_persona_fact(
        self, old_id: str, new_fact: PersonaFact
    ) -> str: ...

    async def delete_persona_fact(self, fact_id: str) -> bool: ...

    async def increment_mention_count(self, fact_id: str) -> int: ...

    async def update_fact_status(self, fact_id: str, status: str) -> None: ...

    async def get_facts_by_context(
        self,
        concepts: list[str],
        *,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list[PersonaFact]: ...

    async def search_facts_semantic(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list[tuple[PersonaFact, float]]: ...


class EmbeddingProtocol(Protocol):
    """Interface for embedding generation."""

    async def generate_embedding(self, text: str) -> list[float]: ...

    async def generate_embeddings_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...


class ExtractorProtocol(Protocol):
    """Interface for pattern extraction from text."""

    async def extract(self, text: str) -> dict[str, object]:
        """Extract patterns from text.

        Returns dict with keys: concepts, relations, entities,
        emotional_valence, significance.
        """
        ...

