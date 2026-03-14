"""PostgreSQL trace store implementation."""

from __future__ import annotations

import logging

import asyncpg

from recollect.datetime_utils import now_utc
from recollect.exceptions import StorageError
from recollect.models import MemoryTrace
from recollect.pool import PoolManager
from recollect.storage_utils import row_to_trace, trace_to_params

logger = logging.getLogger(__name__)


class PgTraceStore:
    """PostgreSQL implementation of TraceStore protocol."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def store_trace(self, trace: MemoryTrace) -> str:
        """Store a memory trace, returning its ID."""
        try:
            params = trace_to_params(trace)
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO memory_traces (
                        id, content, pattern, context, embedding, strength,
                        activation_count, retrieval_count, last_activation,
                        last_retrieval, consolidated, created_at, decay_rate,
                        emotional_valence, significance, session_id, user_id
                    ) VALUES (
                        $1, $2, $3::jsonb, $4::jsonb, $5::vector, $6,
                        $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
                    )
                    """,
                    params["id"],
                    params["content"],
                    params["pattern"],
                    params["context"],
                    params["embedding"],
                    params["strength"],
                    params["activation_count"],
                    params["retrieval_count"],
                    params["last_activation"],
                    params["last_retrieval"],
                    params["consolidated"],
                    params["created_at"],
                    params["decay_rate"],
                    params["emotional_valence"],
                    params["significance"],
                    params["session_id"],
                    params["user_id"],
                )
            return trace.id
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to store trace")
            raise StorageError(f"Failed to store trace: {exc}") from exc

    async def get_trace(self, trace_id: str) -> MemoryTrace | None:
        """Fetch a single trace by ID."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM memory_traces WHERE id = $1", trace_id
                )
            if row is None:
                return None
            return row_to_trace(dict(row))
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to get trace {trace_id}: {exc}") from exc

    async def get_traces_bulk(self, trace_ids: list[str]) -> list[MemoryTrace]:
        """Fetch multiple traces by ID."""
        if not trace_ids:
            return []
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM memory_traces WHERE id = ANY($1)",
                    trace_ids,
                )
            return [row_to_trace(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to get traces in bulk: {exc}") from exc

    async def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace by ID. Returns True if deleted."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM memory_traces WHERE id = $1", trace_id
                )
            return result == "DELETE 1"
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to delete trace")
            raise StorageError(f"Failed to delete trace {trace_id}: {exc}") from exc

    async def update_trace_strength(self, trace_id: str, new_strength: float) -> None:
        """Set a trace's strength to a specific value."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE memory_traces SET strength = $1 WHERE id = $2",
                    new_strength,
                    trace_id,
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to update trace strength")
            raise StorageError(f"Failed to update trace strength: {exc}") from exc

    async def mark_activated(self, trace_id: str) -> None:
        """Increment activation counter and update timestamp."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE memory_traces
                    SET activation_count = activation_count + 1,
                        last_activation = $1
                    WHERE id = $2
                    """,
                    now_utc(),
                    trace_id,
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to mark trace activated")
            raise StorageError(f"Failed to mark trace activated: {exc}") from exc

    async def mark_retrieved(self, trace_id: str) -> None:
        """Increment retrieval counter and update timestamp."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE memory_traces
                    SET retrieval_count = retrieval_count + 1,
                        last_retrieval = $1
                    WHERE id = $2
                    """,
                    now_utc(),
                    trace_id,
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to mark trace retrieved")
            raise StorageError(f"Failed to mark trace retrieved: {exc}") from exc

    async def get_recent_traces(self, limit: int = 20) -> list[MemoryTrace]:
        """Get most recently created traces."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM memory_traces
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
            return [row_to_trace(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to get recent traces: {exc}") from exc

    async def get_unconsolidated_traces(self, limit: int = 50) -> list[MemoryTrace]:
        """Get traces not yet consolidated, ordered by creation time."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM memory_traces
                    WHERE consolidated = FALSE
                    ORDER BY created_at ASC
                    LIMIT $1
                    """,
                    limit,
                )
            return [row_to_trace(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to get unconsolidated traces: {exc}") from exc

    async def mark_consolidated(self, trace_id: str) -> None:
        """Mark a trace as consolidated into long-term memory."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE memory_traces SET consolidated = TRUE WHERE id = $1",
                    trace_id,
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to mark trace consolidated")
            raise StorageError(f"Failed to mark trace consolidated: {exc}") from exc

    async def get_traces_by_session(
        self,
        session_id: str,
        *,
        limit: int = 100,
    ) -> list[MemoryTrace]:
        """Get all traces in a session, ordered by creation time."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM memory_traces
                    WHERE session_id = $1
                    ORDER BY created_at ASC
                    LIMIT $2
                    """,
                    session_id,
                    limit,
                )
            return [row_to_trace(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(
                f"Failed to get traces for session {session_id}: {exc}"
            ) from exc
