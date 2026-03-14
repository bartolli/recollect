"""PostgreSQL fact store implementation."""

from __future__ import annotations

import logging

import asyncpg

from recollect.datetime_utils import now_utc
from recollect.exceptions import StorageError
from recollect.models import PersonaFact
from recollect.pool import PoolManager
from recollect.storage_utils import persona_fact_to_params, row_to_persona_fact

logger = logging.getLogger(__name__)


class PgFactStore:
    """PostgreSQL implementation of FactStore protocol."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def store_persona_fact(self, fact: PersonaFact) -> str:
        """Insert a persona fact, returning its ID."""
        try:
            params = persona_fact_to_params(fact)
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO persona_facts (
                        id, subject, predicate, object, category,
                        content, source_trace_id, confidence,
                        created_at, updated_at, superseded_by,
                        status, mention_count, scope, context_tags,
                        embedding, user_id
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                        $12, $13, $14, $15, $16, $17
                    )
                    """,
                    params["id"],
                    params["subject"],
                    params["predicate"],
                    params["object"],
                    params["category"],
                    params["content"],
                    params["source_trace_id"],
                    params["confidence"],
                    params["created_at"],
                    params["updated_at"],
                    params["superseded_by"],
                    params["status"],
                    params["mention_count"],
                    params["scope"],
                    params["context_tags"],
                    params["embedding"],
                    params["user_id"],
                )
            return fact.id
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to store persona fact")
            raise StorageError(f"Failed to store persona fact: {exc}") from exc

    async def get_persona_facts(
        self,
        subject: str | None = None,
        *,
        limit: int = 50,
        user_id: str | None = None,
    ) -> list[PersonaFact]:
        """Get persona facts, optionally filtered by subject and user."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                if subject:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM persona_facts
                        WHERE subject = $1 AND superseded_by IS NULL
                          AND ($3::text IS NULL OR user_id = $3)
                        ORDER BY created_at DESC
                        LIMIT $2
                        """,
                        subject,
                        limit,
                        user_id,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM persona_facts
                        WHERE superseded_by IS NULL
                          AND ($2::text IS NULL OR user_id = $2)
                        ORDER BY created_at DESC
                        LIMIT $1
                        """,
                        limit,
                        user_id,
                    )
            return [row_to_persona_fact(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to get persona facts: {exc}") from exc

    async def get_persona_facts_by_entities(
        self,
        entity_names: list[str],
        *,
        user_id: str | None = None,
    ) -> list[PersonaFact]:
        """Get persona facts where subject matches any entity name."""
        if not entity_names:
            return []
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM persona_facts
                    WHERE LOWER(subject) = ANY($1)
                      AND superseded_by IS NULL
                      AND status IN ('promoted', 'pinned')
                      AND ($2::text IS NULL OR user_id = $2)
                    ORDER BY confidence DESC
                    """,
                    [n.lower() for n in entity_names],
                    user_id,
                )
            return [row_to_persona_fact(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(
                f"Failed to get persona facts by entities: {exc}"
            ) from exc

    async def get_facts_by_entities_and_scopes(
        self,
        entity_names: list[str],
        scopes: list[str],
        *,
        limit: int = 20,
    ) -> list[PersonaFact]:
        """Get promoted/pinned facts matching entity names and scopes."""
        if not entity_names:
            return []
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM persona_facts
                    WHERE LOWER(subject) = ANY($1)
                      AND superseded_by IS NULL
                      AND status IN ('promoted', 'pinned')
                      AND ($2::text[] IS NULL OR scope = ANY($2))
                    ORDER BY confidence DESC
                    LIMIT $3
                    """,
                    [n.lower() for n in entity_names],
                    scopes if scopes else None,
                    limit,
                )
            return [row_to_persona_fact(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(
                f"Failed to get facts by entities and scopes: {exc}"
            ) from exc

    async def supersede_persona_fact(self, old_id: str, new_fact: PersonaFact) -> str:
        """Mark old fact as superseded and insert replacement."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn, conn.transaction():
                params = persona_fact_to_params(new_fact)
                await conn.execute(
                    """
                    INSERT INTO persona_facts (
                        id, subject, predicate, object, category,
                        content, source_trace_id, confidence,
                        created_at, updated_at, superseded_by,
                        status, mention_count, scope, context_tags,
                        embedding, user_id
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                        $12, $13, $14, $15, $16, $17
                    )
                    """,
                    params["id"],
                    params["subject"],
                    params["predicate"],
                    params["object"],
                    params["category"],
                    params["content"],
                    params["source_trace_id"],
                    params["confidence"],
                    params["created_at"],
                    params["updated_at"],
                    params["superseded_by"],
                    params["status"],
                    params["mention_count"],
                    params["scope"],
                    params["context_tags"],
                    params["embedding"],
                    params["user_id"],
                )
                await conn.execute(
                    """
                    UPDATE persona_facts
                    SET superseded_by = $1, updated_at = $2
                    WHERE id = $3
                    """,
                    new_fact.id,
                    now_utc(),
                    old_id,
                )
            return new_fact.id
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to supersede persona fact")
            raise StorageError(f"Failed to supersede persona fact: {exc}") from exc

    async def delete_persona_fact(self, fact_id: str) -> bool:
        """Delete a persona fact by ID. Returns True if deleted."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM persona_facts WHERE id = $1", fact_id
                )
            return result == "DELETE 1"
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to delete persona fact")
            raise StorageError(
                f"Failed to delete persona fact {fact_id}: {exc}"
            ) from exc

    async def increment_mention_count(self, fact_id: str) -> int:
        """Increment mention count and return new value."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    UPDATE persona_facts
                    SET mention_count = mention_count + 1,
                        updated_at = $2
                    WHERE id = $1
                    RETURNING mention_count
                    """,
                    fact_id,
                    now_utc(),
                )
            if row is None:
                raise StorageError(f"Fact {fact_id} not found")
            return int(row["mention_count"])
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to increment mention count: {exc}") from exc

    async def update_fact_status(self, fact_id: str, status: str) -> None:
        """Update the status of a persona fact."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE persona_facts
                    SET status = $2, updated_at = $3
                    WHERE id = $1
                    """,
                    fact_id,
                    status,
                    now_utc(),
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to update fact status: {exc}") from exc

    async def get_facts_by_context(
        self,
        concepts: list[str],
        *,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list[PersonaFact]:
        """Find promoted/pinned facts whose context_tags overlap with concepts."""
        if not concepts:
            return []
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM persona_facts
                    WHERE context_tags && $1::text[]
                      AND superseded_by IS NULL
                      AND status IN ('promoted', 'pinned')
                      AND ($3::text IS NULL OR user_id = $3)
                    ORDER BY confidence DESC
                    LIMIT $2
                    """,
                    concepts,
                    limit,
                    user_id,
                )
            return [row_to_persona_fact(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to get facts by context: {exc}") from exc

    async def search_facts_semantic(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list[tuple[PersonaFact, float]]:
        """Find promoted/pinned facts by embedding similarity.

        Returns (fact, similarity) tuples sorted by similarity descending.
        Only facts with non-null embeddings are searched.
        """
        if not query_embedding:
            return []
        try:
            from recollect.storage_utils import embedding_to_pgvector

            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT *, 1 - (embedding <=> $1::vector) AS similarity
                    FROM persona_facts
                    WHERE embedding IS NOT NULL
                      AND superseded_by IS NULL
                      AND status IN ('promoted', 'pinned')
                      AND ($3::text IS NULL OR user_id = $3)
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    embedding_to_pgvector(query_embedding),
                    limit,
                    user_id,
                )
            return [
                (row_to_persona_fact(dict(r)), float(r["similarity"])) for r in rows
            ]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to search facts semantically: {exc}") from exc
