"""Unit tests for bootstrap.apply() — mocked Connection, no real DB.

Verifies SQL sequence, lock_conn != migration_conn invariant (advisor's
correction), schema-ahead detection, transactional per-migration boundary,
and unlock-in-finally even on failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from recollect.bootstrap.migration import MigrationRegistry
from recollect.bootstrap.runner import (
    BOOTSTRAP_LOCK_KEY_SQL,
    BootstrapResult,
    apply,
)
from recollect.exceptions import BootstrapError

if TYPE_CHECKING:
    import asyncpg


class _FakeConn:
    def __init__(self, fetch_rows: list[dict[str, Any]] | None = None) -> None:
        self.executes: list[str] = []
        self._fetch_rows = fetch_rows or []
        self.tx_entered = 0
        self.tx_exited = 0

    async def execute(self, sql: str, *args: object) -> str:
        self.executes.append(sql)
        return "OK"

    async def fetch(self, sql: str, *args: object) -> list[dict[str, Any]]:
        self.executes.append(sql)
        return self._fetch_rows

    def transaction(self) -> _FakeTxn:
        return _FakeTxn(self)


class _FakeTxn:
    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn

    async def __aenter__(self) -> _FakeTxn:
        self.conn.tx_entered += 1
        return self

    async def __aexit__(self, *_a: object) -> None:
        self.conn.tx_exited += 1


class _FakePool:
    def __init__(self, conns: list[_FakeConn]) -> None:
        self._conns = list(conns)
        self.acquired: list[_FakeConn] = []

    def acquire(self) -> _Acquire:
        return _Acquire(self)


class _Acquire:
    def __init__(self, pool: _FakePool) -> None:
        self._pool = pool

    async def __aenter__(self) -> _FakeConn:
        conn = self._pool._conns.pop(0)
        self._pool.acquired.append(conn)
        return conn

    async def __aexit__(self, *_a: object) -> None:
        return None


class _Mig:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.description = f"{name} stub"
        self.up_conn: object | None = None
        self.up_called = 0
        self._fail = fail

    async def up(self, conn: asyncpg.Connection[Any]) -> None:
        self.up_called += 1
        self.up_conn = conn
        if self._fail:
            raise RuntimeError(f"{self.name} blew up")


def _registry(*names: str) -> MigrationRegistry:
    r = MigrationRegistry()
    for n in names:
        r.register(_Mig(n))
    return r


# -- SQL sequence + lock invariant --


class TestSqlSequence:
    @pytest.mark.asyncio
    async def test_lock_acquired_before_any_other_sql(self) -> None:
        registry = _registry()
        lock_conn = _FakeConn(fetch_rows=[])
        pool = _FakePool([lock_conn])
        await apply(cast("asyncpg.Pool[Any]", pool), registry)
        assert lock_conn.executes[0] == BOOTSTRAP_LOCK_KEY_SQL["acquire"]

    @pytest.mark.asyncio
    async def test_lock_released_on_success(self) -> None:
        registry = _registry()
        lock_conn = _FakeConn(fetch_rows=[])
        pool = _FakePool([lock_conn])
        await apply(cast("asyncpg.Pool[Any]", pool), registry)
        assert lock_conn.executes[-1] == BOOTSTRAP_LOCK_KEY_SQL["release"]

    @pytest.mark.asyncio
    async def test_lock_released_when_migration_fails(self) -> None:
        registry = MigrationRegistry()
        registry.register(_Mig("m001", fail=True))
        lock_conn = _FakeConn(fetch_rows=[])
        mig_conn = _FakeConn()
        pool = _FakePool([lock_conn, mig_conn])
        with pytest.raises(RuntimeError, match="m001 blew up"):
            await apply(cast("asyncpg.Pool[Any]", pool), registry)
        assert lock_conn.executes[-1] == BOOTSTRAP_LOCK_KEY_SQL["release"]

    @pytest.mark.asyncio
    async def test_migration_runs_on_separate_conn_from_lock(self) -> None:
        # advisor invariant: lock_conn != migration_conn so per-migration
        # transactions don't release the lock when they commit.
        registry = MigrationRegistry()
        mig = _Mig("m001")
        registry.register(mig)
        lock_conn = _FakeConn(fetch_rows=[])
        mig_conn = _FakeConn()
        pool = _FakePool([lock_conn, mig_conn])
        await apply(cast("asyncpg.Pool[Any]", pool), registry)
        assert mig.up_conn is mig_conn
        assert mig.up_conn is not lock_conn


# -- Per-migration transaction --


class TestPerMigrationTransaction:
    @pytest.mark.asyncio
    async def test_each_migration_wrapped_in_transaction(self) -> None:
        registry = _registry("m001", "m002")
        lock_conn = _FakeConn(fetch_rows=[])
        c1, c2 = _FakeConn(), _FakeConn()
        pool = _FakePool([lock_conn, c1, c2])
        await apply(cast("asyncpg.Pool[Any]", pool), registry)
        assert c1.tx_entered == 1 and c1.tx_exited == 1
        assert c2.tx_entered == 1 and c2.tx_exited == 1

    @pytest.mark.asyncio
    async def test_applied_migrations_insert_after_up(self) -> None:
        registry = _registry("m001")
        lock_conn = _FakeConn(fetch_rows=[])
        mig_conn = _FakeConn()
        pool = _FakePool([lock_conn, mig_conn])
        await apply(cast("asyncpg.Pool[Any]", pool), registry)
        joined = " ".join(mig_conn.executes)
        assert "INSERT INTO applied_migrations" in joined


# -- Idempotency + schema-ahead --


class TestIdempotencyAndSchemaAhead:
    @pytest.mark.asyncio
    async def test_already_applied_skipped(self) -> None:
        registry = _registry("m001", "m002")
        lock_conn = _FakeConn(fetch_rows=[{"name": "m001"}])
        mig_conn = _FakeConn()
        pool = _FakePool([lock_conn, mig_conn])
        result = await apply(cast("asyncpg.Pool[Any]", pool), registry)
        assert result.applied == ["m002"]
        assert result.skipped == ["m001"]

    @pytest.mark.asyncio
    async def test_all_applied_returns_empty_newly_applied(self) -> None:
        registry = _registry("m001", "m002")
        lock_conn = _FakeConn(fetch_rows=[{"name": "m001"}, {"name": "m002"}])
        pool = _FakePool([lock_conn])
        result = await apply(cast("asyncpg.Pool[Any]", pool), registry)
        assert result.applied == []
        assert result.skipped == ["m001", "m002"]

    @pytest.mark.asyncio
    async def test_schema_ahead_raises_without_running_pending(self) -> None:
        registry = _registry("m001")
        lock_conn = _FakeConn(fetch_rows=[{"name": "m001"}, {"name": "m999_future"}])
        pool = _FakePool([lock_conn])
        with pytest.raises(BootstrapError, match="m999_future"):
            await apply(cast("asyncpg.Pool[Any]", pool), registry)
        # lock still released
        assert lock_conn.executes[-1] == BOOTSTRAP_LOCK_KEY_SQL["release"]


# -- Result shape --


class TestResultShape:
    @pytest.mark.asyncio
    async def test_result_is_bootstrap_result_dataclass(self) -> None:
        registry = _registry()
        lock_conn = _FakeConn(fetch_rows=[])
        pool = _FakePool([lock_conn])
        result = await apply(cast("asyncpg.Pool[Any]", pool), registry)
        assert isinstance(result, BootstrapResult)
        assert result.applied == []
        assert result.skipped == []
        assert result.schema_ahead == []
