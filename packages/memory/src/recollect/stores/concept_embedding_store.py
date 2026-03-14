"""PostgreSQL store for concept-level embeddings."""

from __future__ import annotations

import logging

import asyncpg

from recollect.exceptions import StorageError
from recollect.models import ConceptEmbedding
from recollect.pool import PoolManager
from recollect.storage_utils import concept_embedding_to_params, embedding_to_pgvector

logger = logging.getLogger(__name__)


class PgConceptEmbeddingStore:
    """PostgreSQL implementation of ConceptEmbeddingStore protocol."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def store_concept_embeddings(
        self,
        embeddings: list[ConceptEmbedding],
    ) -> None:
        """Batch-insert concept embeddings."""
        if not embeddings:
            return
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.executemany(
                    """INSERT INTO concept_embeddings
                       (id, concept, owner_type, owner_id, embedding, created_at)
                       VALUES ($1, $2, $3, $4, $5::vector, $6::timestamptz)
                       ON CONFLICT (id) DO NOTHING""",
                    [
                        _params_to_tuple(concept_embedding_to_params(ce))
                        for ce in embeddings
                    ],
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to store concept embeddings")
            raise StorageError(f"Failed to store concept embeddings: {exc}") from exc

    async def get_max_sim_per_owner(
        self,
        query_embedding: list[float],
        *,
        owner_type: str,
        owner_ids: list[str],
    ) -> dict[str, float]:
        """Compute max cosine similarity per owner across their concepts.

        Returns {owner_id: max_similarity} for owners that have
        concept embeddings.
        """
        if not owner_ids:
            return {}
        emb_str = embedding_to_pgvector(query_embedding)
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT owner_id,
                              MAX(1 - (embedding <=> $1::vector)) AS max_sim
                       FROM concept_embeddings
                       WHERE owner_type = $2
                         AND owner_id = ANY($3::text[])
                       GROUP BY owner_id""",
                    emb_str,
                    owner_type,
                    owner_ids,
                )
            return {row["owner_id"]: float(row["max_sim"]) for row in rows}
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Concept max-sim query failed")
            raise StorageError(f"Concept max-sim query failed: {exc}") from exc

    async def delete_by_owner(
        self,
        owner_type: str,
        owner_id: str,
    ) -> int:
        """Delete all concept embeddings for a specific owner."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """DELETE FROM concept_embeddings
                       WHERE owner_type = $1 AND owner_id = $2""",
                    owner_type,
                    owner_id,
                )
            # asyncpg returns "DELETE N"
            return int(result.split()[-1]) if result else 0
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to delete concept embeddings")
            raise StorageError(f"Failed to delete concept embeddings: {exc}") from exc


def _params_to_tuple(params: dict[str, object]) -> tuple[object, ...]:
    """Convert params dict to positional tuple for executemany."""
    return (
        params["id"],
        params["concept"],
        params["owner_type"],
        params["owner_id"],
        params["embedding"],
        params["created_at"],
    )
