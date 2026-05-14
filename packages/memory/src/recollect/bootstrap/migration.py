"""Migration protocol + ordered registry.

A Migration is a named, one-shot DDL/data step with an `up(conn)` coroutine.
The registry preserves registration order — `pending(applied)` returns
unapplied migrations in that order; `schema_ahead(applied)` flags applied
names the running app doesn't know about (rollback safety).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from recollect.exceptions import DuplicateMigrationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import asyncpg


@runtime_checkable
class Migration(Protocol):
    name: str
    description: str

    async def up(self, conn: asyncpg.Connection) -> None: ...


class MigrationRegistry:
    def __init__(self) -> None:
        self._items: list[Migration] = []
        self._names: set[str] = set()

    def register(self, migration: Migration) -> None:
        if migration.name in self._names:
            raise DuplicateMigrationError(
                f"Migration name already registered: {migration.name}"
            )
        self._items.append(migration)
        self._names.add(migration.name)

    def __iter__(self) -> Iterator[Migration]:
        return iter(self._items)

    def names(self) -> list[str]:
        return [m.name for m in self._items]

    def pending(self, applied: Iterable[str]) -> list[Migration]:
        applied_set = set(applied)
        return [m for m in self._items if m.name not in applied_set]

    def schema_ahead(self, applied: Iterable[str]) -> list[str]:
        return [name for name in applied if name not in self._names]
