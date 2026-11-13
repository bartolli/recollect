"""Store clock: ages a probe store by moving stored time.

Emulated time advances by shifting every lifecycle timestamp column
back, so "now - stamp" grows on every clock the SDK reads -- Python
and SQL NOW() alike -- with no clock injected into the code under
measurement.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING

from recollect.datetime_utils import now_utc

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    import asyncpg

SHIFTED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("sessions", "created_at"),
    ("sessions", "ended_at"),
    ("memory_traces", "created_at"),
    ("memory_traces", "last_activation"),
    ("memory_traces", "last_retrieval"),
    ("memory_traces", "last_decayed_at"),
    ("associations", "created_at"),
    ("associations", "last_activation"),
    ("entities", "created_at"),
    ("persona_facts", "created_at"),
    ("persona_facts", "updated_at"),
    ("entity_relations", "created_at"),
    ("concept_embeddings", "created_at"),
    ("recall_tokens", "created_at"),
    ("recall_tokens", "last_activated_at"),
    ("token_stamps", "stamped_at"),
)

# Schema lifecycle stamps; no lifecycle behavior reads them.
EXCLUDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("applied_migrations", "applied_at"),
    ("embedding_contract", "applied_at"),
)

_MICROSECOND = timedelta(microseconds=1)
_PROBE_PREFIX = "probe_"


async def require_probe_database(
    conn: asyncpg.Connection | asyncpg.pool.PoolConnectionProxy,
) -> str:
    """Guard of every store-wide write; returns the guarded database name."""
    # Read from a live connection, not a URL: an empty arm db_url falls
    # back to DATABASE_URL, which a string check never sees.
    name = str(await conn.fetchval("SELECT current_database()"))
    if not name.startswith(_PROBE_PREFIX):
        raise ValueError(
            f"store-wide write refused on database {name!r} -- its name "
            f"must start with {_PROBE_PREFIX!r}"
        )
    return name


_TIME_COLUMNS_SQL = """
    SELECT c.table_name, c.column_name
    FROM information_schema.columns c
    JOIN information_schema.tables t USING (table_schema, table_name)
    WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE'
      AND (c.data_type LIKE 'timestamp%' OR c.data_type LIKE 'time %'
           OR c.data_type IN ('date', 'interval'))
"""


async def _require_classified_columns(conn: asyncpg.pool.PoolConnectionProxy) -> None:
    # Enforced at run time, not only by a schema test: a migration that
    # adds a time column would otherwise be aged around silently.
    rows = await conn.fetch(_TIME_COLUMNS_SQL)
    schema = {(row["table_name"], row["column_name"]) for row in rows}
    unknown = sorted(schema - set(SHIFTED_COLUMNS) - set(EXCLUDED_COLUMNS))
    if unknown:
        names = ", ".join(f"{table}.{column}" for table, column in unknown)
        raise ValueError(
            f"store clock does not classify time column(s) {names}; add each "
            "to SHIFTED_COLUMNS or EXCLUDED_COLUMNS"
        )


def _columns_by_table() -> dict[str, list[str]]:
    by_table: dict[str, list[str]] = defaultdict(list)
    for table, column in SHIFTED_COLUMNS:
        by_table[table].append(column)
    return by_table


def _shift_statements() -> list[str]:
    # Microsecond-valued interval: day-valued arithmetic on timestamptz
    # follows the session zone and is an hour off across a DST boundary.
    step = "$1::bigint * interval '1 microsecond'"
    # One UPDATE per table: each row is rewritten once per shift.
    return [
        f"UPDATE {table} SET "  # noqa: S608 -- identifiers are module constants
        + ", ".join(f"{column} = {column} - {step}" for column in columns)
        for table, columns in _columns_by_table().items()
    ]


class StoreClock:
    """Emulated time = real time since the anchor + total shift.

    Emulated time never freezes: it runs at the real rate between and
    during operations. advance_to samples the wall clock before its own
    shift, so real time spent between advances is absorbed while each
    operation's own duration is not: the store sits at the target plus
    that duration. After the last advance, settle and whatever follows
    add theirs. emulated_offset() measures the sum; the age reached is
    never exact.
    """

    def __init__(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        *,
        now: Callable[[], datetime] = now_utc,
    ) -> None:
        self._pool = pool
        self._now = now
        self._anchor = now()
        self._shifted = timedelta(0)
        self._classified = False

    def emulated_offset(self) -> timedelta:
        return (self._now() - self._anchor) + self._shifted

    async def advance_to(self, offset: timedelta) -> None:
        delta = offset - self.emulated_offset()
        if delta < timedelta(0):
            # A forward shift would un-age the store and skew every later
            # step. A repeated target lands here too: the clock already
            # sits past it by the previous operation's duration.
            raise ValueError(
                f"emulated instant {offset} already passed by {-delta}; "
                "targets must increase by more than the real time a step takes"
            )
        async with self._pool.acquire() as conn, conn.transaction():
            await require_probe_database(conn)
            if not self._classified:
                await _require_classified_columns(conn)
                self._classified = True
            for statement in _shift_statements():
                await conn.execute(statement, delta // _MICROSECOND)
        self._shifted += delta

    async def settle(self) -> None:
        """Rebuild what the shifts bloated; run at every age point, age 0 too.

        created_at is indexed, so each shift is a non-HOT update that
        leaves one dead HNSW entry per row; a vector index scan over them
        returns a handful of live rows for a top-k query. VACUUM FULL
        rebuilds heap and indexes, ANALYZE gives every point the same
        planner inputs. VACUUM cannot run inside a transaction.
        """
        async with self._pool.acquire() as conn:
            await require_probe_database(conn)
            for table in _columns_by_table():
                await conn.execute(f"VACUUM (FULL, ANALYZE) {table}")
