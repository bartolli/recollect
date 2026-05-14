"""m002_user_id_backfill — promote persona_facts.user_id to NOT NULL.

Pre-2026-03 rows carried NULL user_id; MCP recall filtered them silently
while reflect surfaced them. Migration:

  1. Read default user_id (server.user_id config / env override).
  2. If NULL rows exist AND default is empty -> BootstrapError, no writes.
  3. UPDATE persona_facts SET user_id = <default> WHERE user_id IS NULL.
  4. ALTER COLUMN user_id SET NOT NULL.

`get_default_user_id` is injectable for testing — production uses the
module-level config singleton.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import asyncpg

from recollect.config import config
from recollect.exceptions import BootstrapError


def _default_user_id_from_config() -> str:
    return config.server_user_id


class _M002UserIdBackfill:
    name = "m002_user_id_backfill"
    description = (
        "Backfill persona_facts.user_id with server default; "
        "enforce NOT NULL constraint."
    )

    def __init__(
        self,
        get_default_user_id: Callable[[], str] | None = None,
    ) -> None:
        self._get_user_id = get_default_user_id or _default_user_id_from_config

    async def up(self, conn: asyncpg.Connection[Any]) -> None:
        default_uid = self._get_user_id()
        null_count = await conn.fetchval(
            "SELECT COUNT(*) FROM persona_facts WHERE user_id IS NULL"
        )
        if null_count and not default_uid:
            raise BootstrapError(
                f"persona_facts has {null_count} rows with NULL user_id; "
                "set server.user_id in config (or MEMORY_SERVER_USER_ID env) "
                "before this migration can apply."
            )
        if default_uid:
            await conn.execute(
                "UPDATE persona_facts SET user_id = $1 WHERE user_id IS NULL",
                default_uid,
            )
        await conn.execute(
            "ALTER TABLE persona_facts ALTER COLUMN user_id SET NOT NULL"
        )


m002_user_id_backfill = _M002UserIdBackfill()
