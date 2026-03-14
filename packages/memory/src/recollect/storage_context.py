"""Storage context for composable sub-store wiring.

StorageContext is a frozen dataclass that composes independently-typed
sub-stores. Use create_storage_context() to construct the default
PostgreSQL-backed context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from recollect.pool import PoolManager
from recollect.stores import (
    PgAssociationStore,
    PgConceptEmbeddingStore,
    PgEntityIndex,
    PgEntityRelationStore,
    PgFactStore,
    PgRecallTokenStore,
    PgSessionStore,
    PgTraceStore,
    PgVectorIndex,
)

if TYPE_CHECKING:
    from recollect.protocols import (
        AssociationStore,
        ConceptEmbeddingStore,
        EntityIndex,
        EntityRelationStore,
        FactStore,
        RecallTokenStore,
        SessionStore,
        TraceStore,
        VectorIndex,
    )


@dataclass(frozen=True)
class StorageContext:
    """Passive container for composable storage sub-stores.

    Each sub-store owns its domain (traces, vectors, entities,
    associations, facts) and shares a connection pool via PoolManager.
    """

    pool: PoolManager
    traces: TraceStore
    vectors: VectorIndex
    entities: EntityIndex
    associations: AssociationStore
    facts: FactStore
    entity_relations: EntityRelationStore
    concept_embeddings: ConceptEmbeddingStore
    sessions: SessionStore
    recall_tokens: RecallTokenStore

    async def initialize(self) -> None:
        """Initialize the connection pool and create schema."""
        await self.pool.initialize()

    async def close(self) -> None:
        """Close the connection pool."""
        await self.pool.close()


def create_storage_context(
    database_url: str | None = None,
) -> StorageContext:
    """Create a StorageContext with PostgreSQL sub-stores.

    All sub-stores share the same PoolManager for connection reuse.
    """
    pool = PoolManager(database_url)
    return StorageContext(
        pool=pool,
        traces=PgTraceStore(pool),
        vectors=PgVectorIndex(pool),
        entities=PgEntityIndex(pool),
        associations=PgAssociationStore(pool),
        facts=PgFactStore(pool),
        entity_relations=PgEntityRelationStore(pool),
        concept_embeddings=PgConceptEmbeddingStore(pool),
        sessions=PgSessionStore(pool),
        recall_tokens=PgRecallTokenStore(pool),
    )
