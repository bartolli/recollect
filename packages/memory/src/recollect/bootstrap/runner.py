"""Bootstrap runner: advisory-locked migration application.

Lock is held on a dedicated connection for the full apply() call. Each
migration runs in its own transaction on a freshly-acquired connection so
its commit does not release the session-scoped advisory lock.

Schema-ahead detection (applied names not in the running app's registry)
raises BootstrapError before any migration runs — rollback-safety guard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from recollect.exceptions import BootstrapError

if TYPE_CHECKING:
    import asyncpg

    from recollect.bootstrap.migration import MigrationRegistry

logger = logging.getLogger(__name__)


BOOTSTRAP_LOCK_KEY_SQL = {
    "acquire": "SELECT pg_advisory_lock(hashtext('recollect.bootstrap'))",
    "release": "SELECT pg_advisory_unlock(hashtext('recollect.bootstrap'))",
}


_APPLIED_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS applied_migrations (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


_APPLIED_NAMES_SQL = "SELECT name FROM applied_migrations ORDER BY applied_at"


_INSERT_APPLIED_SQL = (
    "INSERT INTO applied_migrations (name, description) VALUES ($1, $2)"
)


@dataclass(frozen=True)
class BootstrapResult:
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    schema_ahead: list[str] = field(default_factory=list)


async def apply(
    pool: asyncpg.Pool,
    registry: MigrationRegistry,
) -> BootstrapResult:
    async with pool.acquire() as lock_conn:
        await lock_conn.execute(BOOTSTRAP_LOCK_KEY_SQL["acquire"])
        try:
            await lock_conn.execute(_APPLIED_MIGRATIONS_DDL)
            rows = await lock_conn.fetch(_APPLIED_NAMES_SQL)
            applied_names = [r["name"] for r in rows]

            ahead = registry.schema_ahead(applied_names)
            if ahead:
                raise BootstrapError(
                    f"Schema ahead of running app: applied={ahead} "
                    f"registered={registry.names()}"
                )

            pending = registry.pending(applied_names)
            for migration in pending:
                async with pool.acquire() as mig_conn, mig_conn.transaction():
                    await migration.up(
                        cast("asyncpg.Connection[Any]", mig_conn)
                    )
                    await mig_conn.execute(
                        _INSERT_APPLIED_SQL, migration.name, migration.description
                    )
                logger.info("Applied migration: %s", migration.name)

            result = BootstrapResult(
                applied=[m.name for m in pending],
                skipped=applied_names,
                schema_ahead=[],
            )
            logger.info(
                "Bootstrap complete: applied=%d skipped=%d (applied_names=%s)",
                len(result.applied),
                len(result.skipped),
                result.applied or "none",
            )
            return result
        finally:
            await lock_conn.execute(BOOTSTRAP_LOCK_KEY_SQL["release"])
