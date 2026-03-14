"""PostgreSQL association store implementation."""

from __future__ import annotations

import logging

import asyncpg

from recollect.exceptions import StorageError
from recollect.models import Association
from recollect.pool import PoolManager
from recollect.storage_utils import association_to_params, row_to_association

logger = logging.getLogger(__name__)


class PgAssociationStore:
    """PostgreSQL implementation of AssociationStore protocol."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def store_association(self, association: Association) -> None:
        """Store or update an association between traces."""
        try:
            params = association_to_params(association)
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO associations (
                        id, source_trace_id, target_trace_id,
                        association_type, weight, forward_strength,
                        backward_strength, activation_count,
                        last_activation, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (pair_key) DO UPDATE SET
                        weight = GREATEST(associations.weight, EXCLUDED.weight),
                        forward_strength = GREATEST(
                            associations.forward_strength,
                            EXCLUDED.forward_strength
                        ),
                        backward_strength = GREATEST(
                            associations.backward_strength,
                            EXCLUDED.backward_strength
                        ),
                        activation_count =
                            associations.activation_count + 1
                    """,
                    params["id"],
                    params["source_trace_id"],
                    params["target_trace_id"],
                    params["association_type"],
                    params["weight"],
                    params["forward_strength"],
                    params["backward_strength"],
                    params["activation_count"],
                    params["last_activation"],
                    params["created_at"],
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to store association")
            raise StorageError(f"Failed to store association: {exc}") from exc

    async def get_associations(self, trace_id: str) -> list[Association]:
        """Get all associations involving a trace (as source or target)."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM associations
                    WHERE source_trace_id = $1 OR target_trace_id = $1
                    """,
                    trace_id,
                )
            return [row_to_association(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get associations")
            raise StorageError(f"Failed to get associations: {exc}") from exc
