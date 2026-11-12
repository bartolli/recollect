"""Store snapshot: one seeding, restored before every sweep point."""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse, urlunparse

import asyncpg

from probe_cli.store_clock import require_probe_database

_SUFFIX = "_snapshot"
_FOREIGN_SESSIONS_SQL = """
    SELECT count(*) FROM pg_stat_activity
    WHERE datname = current_database() AND pid <> pg_backend_pid()
      AND backend_type = 'client backend'
"""
# A closed pool's backends outlive the client close by a moment; the
# server grants CREATE and DROP DATABASE the same wait.
_EXIT_GRACE_SECONDS = 5.0


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


async def _require_no_foreign_session(conn: asyncpg.Connection, store: str) -> None:
    # A foreign session would run its own passes between a point's steps.
    deadline = time.monotonic() + _EXIT_GRACE_SECONDS
    while sessions := await conn.fetchval(_FOREIGN_SESSIONS_SQL):
        if time.monotonic() >= deadline:
            raise ValueError(
                f"database {store!r} has {sessions} foreign session(s); "
                "a sweep point needs the store to itself"
            )
        await asyncio.sleep(0.1)


class StoreSnapshot:
    """The seeded store as a template database beside it.

    Evaluation writes boosts, reinforcement, and reactivations, so sweep
    points are never chained: each one restores the snapshot.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    async def capture(self) -> None:
        store = await self._guarded_store(exclusive=True)
        snapshot = _quoted(store + _SUFFIX)
        await self._maintenance(
            f"DROP DATABASE IF EXISTS {snapshot}",
            f"CREATE DATABASE {snapshot} TEMPLATE {_quoted(store)}",
        )

    async def restore(self) -> None:
        store = await self._guarded_store(exclusive=True)
        await self._maintenance(
            f"DROP DATABASE {_quoted(store)}",
            f"CREATE DATABASE {_quoted(store)} TEMPLATE {_quoted(store + _SUFFIX)}",
            requires=store + _SUFFIX,
        )

    async def discard(self) -> None:
        store = await self._guarded_store(exclusive=False)
        await self._maintenance(f"DROP DATABASE IF EXISTS {_quoted(store + _SUFFIX)}")

    async def _guarded_store(self, *, exclusive: bool) -> str:
        conn = await asyncpg.connect(self._db_url)
        try:
            store = await require_probe_database(conn)
            if exclusive:
                await _require_no_foreign_session(conn, store)
            return store
        finally:
            await conn.close()

    async def _maintenance(
        self, *statements: str, requires: str | None = None
    ) -> None:
        # A database cannot be dropped or copied from a session inside it.
        url = urlunparse(urlparse(self._db_url)._replace(path="/postgres"))
        conn = await asyncpg.connect(url)
        try:
            # Checked before the drop: a restore without its snapshot would
            # leave no store at all.
            if requires is not None and not await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", requires
            ):
                raise ValueError(f"no snapshot database {requires!r} to restore from")
            for statement in statements:
                await conn.execute(statement)
        finally:
            await conn.close()
