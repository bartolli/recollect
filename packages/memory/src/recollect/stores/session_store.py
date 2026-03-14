"""PostgreSQL session store implementation."""

from __future__ import annotations

import logging

import asyncpg

from recollect.datetime_utils import now_utc
from recollect.exceptions import StorageError
from recollect.models import Session
from recollect.pool import PoolManager
from recollect.storage_utils import row_to_session, session_to_params

logger = logging.getLogger(__name__)


class PgSessionStore:
    """PostgreSQL implementation of SessionStore protocol."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def create_session(self, session: Session) -> str:
        """Insert a session, returning its ID."""
        try:
            params = session_to_params(session)
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO sessions (
                        id, user_id, title, status,
                        summary_trace_id, created_at, ended_at, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                    """,
                    params["id"],
                    params["user_id"],
                    params["title"],
                    params["status"],
                    params["summary_trace_id"],
                    params["created_at"],
                    params["ended_at"],
                    params["metadata"],
                )
            return session.id
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to create session")
            raise StorageError(f"Failed to create session: {exc}") from exc

    async def get_session(self, session_id: str) -> Session | None:
        """Fetch a session by ID."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM sessions WHERE id = $1", session_id
                )
            if row is None:
                return None
            return row_to_session(dict(row))
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get session")
            raise StorageError(f"Failed to get session {session_id}: {exc}") from exc

    async def end_session(
        self,
        session_id: str,
        *,
        summary_trace_id: str | None = None,
    ) -> None:
        """Mark a session as ended."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE sessions
                    SET status = 'ended', ended_at = $2,
                        summary_trace_id = COALESCE($3, summary_trace_id)
                    WHERE id = $1
                    """,
                    session_id,
                    now_utc(),
                    summary_trace_id,
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to end session")
            raise StorageError(f"Failed to end session {session_id}: {exc}") from exc

    async def get_sessions(
        self,
        user_id: str,
        *,
        limit: int = 50,
    ) -> list[Session]:
        """List sessions for a user, most recent first."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM sessions
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    user_id,
                    limit,
                )
            return [row_to_session(dict(r)) for r in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get sessions")
            raise StorageError(
                f"Failed to get sessions for user {user_id}: {exc}"
            ) from exc

    async def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
        summary_trace_id: str | None = None,
    ) -> None:
        """Update session fields dynamically (non-None fields only)."""
        updates: list[str] = []
        values: list[object] = [session_id]
        idx = 2

        if title is not None:
            updates.append(f"title = ${idx}")
            values.append(title)
            idx += 1
        if status is not None:
            updates.append(f"status = ${idx}")
            values.append(status)
            idx += 1
        if summary_trace_id is not None:
            updates.append(f"summary_trace_id = ${idx}")
            values.append(summary_trace_id)
            idx += 1

        if not updates:
            return

        try:
            pool = await self._pool_mgr.get_pool()
            sql = f"UPDATE sessions SET {', '.join(updates)} WHERE id = $1"  # noqa: S608
            async with pool.acquire() as conn:
                await conn.execute(sql, *values)
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to update session")
            raise StorageError(f"Failed to update session {session_id}: {exc}") from exc
