"""Cross-cutting storage operations not owned by a single sub-store.

Embedding-contract read + verify lives here because it spans the
embedding provider and the bootstrap-owned embedding_contract table.
"""

from __future__ import annotations

import logging

import asyncpg

from recollect.exceptions import EmbeddingContractError, StorageError

logger = logging.getLogger(__name__)


async def get_embedding_contract(
    pool: asyncpg.Pool[asyncpg.Record],
) -> tuple[str, str] | None:
    """Read the single embedding_contract row.

    Returns (model, task_prefix_version) or None if the table is absent
    (pre-m003 DB) or has no row (m003 not yet applied). Only absence is
    swallowed -- any other DB error propagates so connect() cannot
    silently skip contract verification.
    """
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                "SELECT model, task_prefix_version FROM embedding_contract"
            )
        except asyncpg.UndefinedTableError:
            return None
        except asyncpg.PostgresError as exc:
            logger.exception("Failed to read embedding contract")
            raise StorageError(
                f"Failed to read embedding contract: {exc}"
            ) from exc
    if row is None:
        return None
    return (row["model"], row["task_prefix_version"])


def verify_embedding_contract(
    *,
    stored: tuple[str, str] | None,
    current: tuple[str, str],
) -> None:
    """Raise EmbeddingContractError if stored conflicts with current.

    Stored is None on a freshly-bootstrapped DB before m003 applies;
    treated as compatible (m003 will record the current contract).
    """
    if stored is None:
        return
    if stored[0] != current[0]:
        raise EmbeddingContractError(
            f"Stored embedding model {stored[0]!r} differs from running "
            f"provider model {current[0]!r}; re-embed required or restore "
            f"the matching provider."
        )
    if stored[1] != current[1]:
        raise EmbeddingContractError(
            f"Stored task_prefix_version {stored[1]!r} differs from running "
            f"provider {current[1]!r}; re-embed required."
        )
