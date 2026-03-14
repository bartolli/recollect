"""Backward-compatible storage facade.

PostgresStorage delegates to StorageContext sub-stores. New code should
use StorageContext directly via create_storage_context().
"""

from __future__ import annotations

from recollect.models import (
    Association,
    EntityRelation,
    MemoryTrace,
    PersonaFact,
    TraceConcept,
    TraceEntity,
)
from recollect.pool import PoolManager
from recollect.storage_context import create_storage_context


class PostgresStorage:
    """Backward-compatible facade over StorageContext sub-stores.

    Prefer using StorageContext directly for new code.
    """

    def __init__(self, database_url: str | None = None) -> None:
        self._ctx = create_storage_context(database_url)

    async def initialize(self) -> None:
        """Create schema tables and indexes."""
        await self._ctx.initialize()

    def get_schema_sql(self) -> str:
        """Return schema DDL for inspection."""
        return PoolManager.get_schema_sql()

    # -- Trace operations --

    async def store_trace(self, trace: MemoryTrace) -> str:
        """Store a memory trace, returning its ID."""
        return await self._ctx.traces.store_trace(trace)

    async def get_trace(self, trace_id: str) -> MemoryTrace | None:
        """Fetch a single trace by ID."""
        return await self._ctx.traces.get_trace(trace_id)

    async def get_traces_bulk(self, trace_ids: list[str]) -> list[MemoryTrace]:
        """Fetch multiple traces by ID."""
        return await self._ctx.traces.get_traces_bulk(trace_ids)

    async def update_trace_strength(self, trace_id: str, new_strength: float) -> None:
        """Set a trace's strength to a specific value."""
        await self._ctx.traces.update_trace_strength(trace_id, new_strength)

    async def mark_activated(self, trace_id: str) -> None:
        """Increment activation counter and update timestamp."""
        await self._ctx.traces.mark_activated(trace_id)

    async def mark_retrieved(self, trace_id: str) -> None:
        """Increment retrieval counter and update timestamp."""
        await self._ctx.traces.mark_retrieved(trace_id)

    async def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace by ID. Returns True if deleted."""
        return await self._ctx.traces.delete_trace(trace_id)

    async def get_recent_traces(self, limit: int = 20) -> list[MemoryTrace]:
        """Get most recently created traces."""
        return await self._ctx.traces.get_recent_traces(limit)

    async def get_unconsolidated_traces(self, limit: int = 50) -> list[MemoryTrace]:
        """Get traces not yet consolidated, ordered by creation time."""
        return await self._ctx.traces.get_unconsolidated_traces(limit)

    async def mark_consolidated(self, trace_id: str) -> None:
        """Mark a trace as consolidated into long-term memory."""
        await self._ctx.traces.mark_consolidated(trace_id)

    # -- Vector operations --

    async def search_semantic(
        self, query_embedding: list[float], limit: int = 10
    ) -> list[tuple[MemoryTrace, float]]:
        """Find traces by vector similarity with cosine similarity scores."""
        return await self._ctx.vectors.search_semantic(query_embedding, limit)

    async def spread_activation(
        self, seed_id: str, max_depth: int = 2
    ) -> list[tuple[MemoryTrace, float]]:
        """Spread activation from seed trace through associations."""
        return await self._ctx.vectors.spread_activation(seed_id, max_depth)

    # -- Entity operations --

    async def store_trace_entities(
        self, trace_id: str, entities: list[TraceEntity]
    ) -> None:
        """Batch insert entity-trace links."""
        await self._ctx.entities.store_trace_entities(trace_id, entities)

    async def store_trace_concepts(
        self, trace_id: str, concepts: list[TraceConcept]
    ) -> None:
        """Batch insert concept-trace links."""
        await self._ctx.entities.store_trace_concepts(trace_id, concepts)

    async def get_traces_by_entity(
        self, entity_name: str, *, limit: int = 20
    ) -> list[str]:
        """Return trace IDs linked to an entity name."""
        return await self._ctx.entities.get_traces_by_entity(entity_name, limit=limit)

    async def get_traces_by_concept(
        self, concept: str, *, limit: int = 20
    ) -> list[str]:
        """Return trace IDs linked to a concept."""
        return await self._ctx.entities.get_traces_by_concept(concept, limit=limit)

    async def match_entities(
        self, names: list[str], *, limit: int = 20
    ) -> list[tuple[str, float]]:
        """Match entity names using trigram similarity."""
        return await self._ctx.entities.match_entities(names, limit=limit)

    # -- Association operations --

    async def store_association(self, association: Association) -> None:
        """Store or update an association between traces."""
        await self._ctx.associations.store_association(association)

    async def get_associations(self, trace_id: str) -> list[Association]:
        """Get all associations involving a trace."""
        return await self._ctx.associations.get_associations(trace_id)

    # -- Fact operations --

    async def store_persona_fact(self, fact: PersonaFact) -> str:
        """Insert a persona fact, returning its ID."""
        return await self._ctx.facts.store_persona_fact(fact)

    async def get_persona_facts(
        self, subject: str | None = None, *, limit: int = 50
    ) -> list[PersonaFact]:
        """Get persona facts, optionally filtered by subject."""
        return await self._ctx.facts.get_persona_facts(subject, limit=limit)

    async def get_persona_facts_by_entities(
        self, entity_names: list[str]
    ) -> list[PersonaFact]:
        """Get persona facts where subject matches any entity name."""
        return await self._ctx.facts.get_persona_facts_by_entities(entity_names)

    async def get_facts_by_entities_and_scopes(
        self,
        entity_names: list[str],
        scopes: list[str],
        *,
        limit: int = 20,
    ) -> list[PersonaFact]:
        """Get facts filtered by entity names and scopes."""
        return await self._ctx.facts.get_facts_by_entities_and_scopes(
            entity_names,
            scopes,
            limit=limit,
        )

    async def supersede_persona_fact(self, old_id: str, new_fact: PersonaFact) -> str:
        """Mark old fact as superseded and insert replacement."""
        return await self._ctx.facts.supersede_persona_fact(old_id, new_fact)

    async def delete_persona_fact(self, fact_id: str) -> bool:
        """Delete a persona fact by ID. Returns True if deleted."""
        return await self._ctx.facts.delete_persona_fact(fact_id)

    async def increment_mention_count(self, fact_id: str) -> int:
        """Increment mention count for a fact."""
        return await self._ctx.facts.increment_mention_count(fact_id)

    async def update_fact_status(self, fact_id: str, status: str) -> None:
        """Update fact status."""
        await self._ctx.facts.update_fact_status(fact_id, status)

    async def get_facts_by_context(
        self, concepts: list[str], *, limit: int = 10
    ) -> list[PersonaFact]:
        """Find facts by context tag overlap."""
        return await self._ctx.facts.get_facts_by_context(concepts, limit=limit)

    # -- Entity relation operations --

    async def store_entity_relation(self, rel: EntityRelation) -> str:
        """Store an entity-to-entity relationship."""
        return await self._ctx.entity_relations.store_relation(rel)

    async def get_entity_relations(
        self, entity_name: str, *, limit: int = 20
    ) -> list[EntityRelation]:
        """Get direct relations for an entity."""
        return await self._ctx.entity_relations.get_relations(entity_name, limit=limit)

    async def get_related_entities(
        self, entity_name: str, *, max_depth: int = 3
    ) -> list[str]:
        """Get entities reachable via relationship graph."""
        return await self._ctx.entity_relations.get_related_entities(
            entity_name, max_depth=max_depth
        )

    # -- Lifecycle --

    async def close(self) -> None:
        """Close the connection pool."""
        await self._ctx.close()
