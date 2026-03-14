"""PostgreSQL store for recall tokens and trace stamps."""

from __future__ import annotations

import logging
from typing import Literal, cast

import asyncpg

from recollect.exceptions import StorageError
from recollect.models import RecallToken
from recollect.pool import PoolManager

logger = logging.getLogger(__name__)


class PgRecallTokenStore:
    """PostgreSQL-backed recall token storage."""

    def __init__(self, pool_mgr: PoolManager) -> None:
        self._pool_mgr = pool_mgr

    async def create_token(self, token: RecallToken) -> str:
        """Insert a recall token, returning its ID."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO recall_tokens"
                    " (id, label, strength, significance,"
                    " created_at, last_activated_at, status)"
                    " VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    token.id,
                    token.label,
                    token.strength,
                    token.significance,
                    token.created_at,
                    token.last_activated_at,
                    token.status,
                )
            return token.id
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to create recall token")
            raise StorageError(f"Failed to create recall token: {exc}") from exc

    async def stamp_traces(self, token_id: str, trace_ids: list[str]) -> None:
        """Stamp a token onto one or more traces."""
        if not trace_ids:
            return
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.executemany(
                    "INSERT INTO token_stamps (token_id, trace_id)"
                    " VALUES ($1, $2)"
                    " ON CONFLICT DO NOTHING",
                    [(token_id, tid) for tid in trace_ids],
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to stamp traces")
            raise StorageError(f"Failed to stamp traces: {exc}") from exc

    async def update_token_label(self, token_id: str, new_label: str) -> None:
        """Update an existing token's label (e.g., to append new implications)."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE recall_tokens SET label = $1 WHERE id = $2",
                    new_label,
                    token_id,
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to update token label")
            raise StorageError(f"Failed to update token label: {exc}") from exc

    async def update_token(
        self, token_id: str, new_label: str, significance: float
    ) -> None:
        """Update an existing token's label and significance (for revise action)."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE recall_tokens SET label = $1, significance = $2"
                    " WHERE id = $3",
                    new_label,
                    significance,
                    token_id,
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to update recall token")
            raise StorageError(f"Failed to update recall token: {exc}") from exc

    async def find_groups_for_traces(
        self,
        trace_ids: list[str],
        *,
        strength_threshold: float = 0.1,
        include_archived: bool = False,
    ) -> list[dict[str, object]]:
        """Find existing token groups linked to any of the given traces.

        Returns list of dicts with keys:
            token_id, label, strength, significance, status, stamped_trace_ids
        """
        if not trace_ids:
            return []
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                query = """
                    SELECT rt.id, rt.label, rt.strength, rt.significance,
                           rt.status, array_agg(ts.trace_id) AS stamped_ids
                    FROM recall_tokens rt
                    JOIN token_stamps ts ON ts.token_id = rt.id
                    WHERE ts.trace_id = ANY($1::text[])
                """
                params: list[object] = [trace_ids]
                if include_archived:
                    # Return both active and archived, skip strength filter
                    pass
                else:
                    params.append(strength_threshold)
                    query += f" AND rt.strength > ${len(params)}"
                    query += " AND rt.status = 'active'"
                query += """
                    GROUP BY rt.id, rt.label, rt.strength, rt.significance,
                             rt.status
                    ORDER BY rt.strength DESC
                """
                rows = await conn.fetch(query, *params)
            return [
                {
                    "token_id": str(row["id"]),
                    "label": str(row["label"]),
                    "strength": float(row["strength"]),
                    "significance": float(row["significance"]),
                    "status": str(row["status"]),
                    "stamped_trace_ids": [str(tid) for tid in row["stamped_ids"]],
                }
                for row in rows
            ]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to find groups for traces")
            raise StorageError(f"Failed to find groups for traces: {exc}") from exc

    async def get_activated_trace_ids(
        self,
        seed_trace_ids: list[str],
        *,
        strength_threshold: float = 0.1,
    ) -> list[tuple[str, str, float, float, str]]:
        """One-hop token activation from seed traces.

        Returns (trace_id, token_label, token_strength, token_significance,
        anchor_trace_id) for traces linked via shared tokens but NOT in
        the seed set. anchor_trace_id is the seed trace that carried the
        token into the result set.
        """
        if not seed_trace_ids:
            return []
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT t.trace_id, rt.id AS token_id,
                           rt.label, rt.strength, rt.significance,
                           seed_stamps.trace_id AS anchor_id
                    FROM token_stamps t
                    JOIN recall_tokens rt ON rt.id = t.token_id
                    JOIN token_stamps seed_stamps
                        ON seed_stamps.token_id = t.token_id
                        AND seed_stamps.trace_id = ANY($1::text[])
                    WHERE t.trace_id != ALL($1::text[])
                    AND rt.strength > $2
                    AND rt.status = 'active'
                    ORDER BY rt.strength DESC
                    """,
                    seed_trace_ids,
                    strength_threshold,
                )
            return [
                (
                    str(row["trace_id"]),
                    str(row["label"]),
                    float(row["strength"]),
                    float(row["significance"]),
                    str(row["anchor_id"]),
                )
                for row in rows
            ]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get activated trace IDs")
            raise StorageError(f"Failed to get activated trace IDs: {exc}") from exc

    async def find_token_by_traces(
        self, trace_ids: list[str]
    ) -> RecallToken | None:
        """Find an existing token that stamps ALL of the given trace IDs."""
        if not trace_ids:
            return None
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT rt.id, rt.label, rt.strength, rt.significance,
                           rt.created_at, rt.last_activated_at, rt.status
                    FROM recall_tokens rt
                    JOIN token_stamps ts ON ts.token_id = rt.id
                    WHERE ts.trace_id = ANY($1::text[])
                    AND rt.status = 'active'
                    GROUP BY rt.id
                    HAVING COUNT(DISTINCT ts.trace_id) = $2
                    ORDER BY rt.strength DESC
                    LIMIT 1
                    """,
                    trace_ids,
                    len(trace_ids),
                )
            if row is None:
                return None
            return RecallToken(
                id=str(row["id"]),
                label=str(row["label"]),
                strength=float(row["strength"]),
                significance=float(row["significance"]),
                created_at=row["created_at"],
                last_activated_at=row["last_activated_at"],
                status=cast(Literal["active", "archived"], row["status"]),
            )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to find token by traces")
            raise StorageError(f"Failed to find token by traces: {exc}") from exc

    async def reinforce_tokens(
        self, token_ids: list[str], boost: float = 0.1
    ) -> None:
        """Increment strength of activated tokens (Hebbian reinforcement)."""
        if not token_ids:
            return
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE recall_tokens
                    SET strength = CASE
                            WHEN status = 'archived' THEN significance
                            ELSE LEAST(1.0, strength + $2)
                        END,
                        status = 'active',
                        last_activated_at = NOW()
                    WHERE id = ANY($1::text[])
                    """,
                    token_ids,
                    boost,
                )
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to reinforce tokens")
            raise StorageError(f"Failed to reinforce tokens: {exc}") from exc

    async def get_tokens_for_traces(
        self, trace_ids: list[str], *, strength_threshold: float = 0.1
    ) -> list[tuple[RecallToken, list[str]]]:
        """Find active tokens linked to any of the given traces."""
        if not trace_ids:
            return []
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT rt.id, rt.label, rt.strength, rt.significance,
                           rt.created_at, rt.last_activated_at,
                           array_agg(ts.trace_id) AS stamped_traces
                    FROM recall_tokens rt
                    JOIN token_stamps ts ON ts.token_id = rt.id
                    WHERE ts.trace_id = ANY($1::text[])
                    AND rt.strength > $2
                    AND rt.status = 'active'
                    GROUP BY rt.id
                    ORDER BY rt.strength DESC
                    """,
                    trace_ids,
                    strength_threshold,
                )
            return [_row_to_token_with_traces(row) for row in rows]
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to get tokens for traces")
            raise StorageError(f"Failed to get tokens for traces: {exc}") from exc

    async def delete_by_trace(self, trace_id: str) -> int:
        """Remove all token stamps for a trace. Returns count removed."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM token_stamps WHERE trace_id = $1",
                    trace_id,
                )
            # asyncpg returns "DELETE N"
            return int(result.split()[-1]) if result else 0
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to delete token stamps")
            raise StorageError(f"Failed to delete token stamps: {exc}") from exc

    async def decay_inactive(
        self, decay_factor: float, *, min_strength: float = 0.01
    ) -> int:
        """Decay active token strengths and archive weak tokens."""
        try:
            pool = await self._pool_mgr.get_pool()
            async with pool.acquire() as conn, conn.transaction():
                # Decay all active tokens
                result = await conn.execute(
                    "UPDATE recall_tokens SET strength = strength * $1"
                    " WHERE status = 'active'",
                    decay_factor,
                )
                # Archive tokens below threshold
                await conn.execute(
                    "UPDATE recall_tokens SET status = 'archived'"
                    " WHERE strength < $1 AND status = 'active'",
                    min_strength,
                )
            return int(result.split()[-1]) if result else 0
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to decay recall tokens")
            raise StorageError(f"Failed to decay recall tokens: {exc}") from exc


def _row_to_token_with_traces(
    row: asyncpg.Record,
) -> tuple[RecallToken, list[str]]:
    """Convert a database row to (RecallToken, stamped_trace_ids) pair."""
    token = RecallToken(
        id=str(row["id"]),
        label=str(row["label"]),
        strength=float(row["strength"]),
        significance=float(row["significance"]),
        created_at=row["created_at"],
        last_activated_at=row["last_activated_at"],
    )
    stamped = [str(tid) for tid in row["stamped_traces"]]
    return (token, stamped)
