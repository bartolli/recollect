"""Tests for migration registry + Migration protocol.

Pure unit: stub migrations declared in-file; no asyncpg, no fixtures.
Covers protocol shape, ordered pending diff, duplicate detection, and
schema-ahead identification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from recollect.bootstrap.migration import (
    Migration,
    MigrationRegistry,
)
from recollect.exceptions import DuplicateMigrationError

if TYPE_CHECKING:
    import asyncpg


class _StubMigration:
    """In-test stub satisfying the Migration protocol."""

    def __init__(self, name: str, description: str = "stub") -> None:
        self.name = name
        self.description = description
        self.up_calls: list[asyncpg.Connection] = []

    async def up(self, conn: asyncpg.Connection) -> None:
        self.up_calls.append(conn)


# -- Migration protocol --


class TestMigrationProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        stub = _StubMigration("m001")
        assert isinstance(stub, Migration)

    def test_protocol_requires_name_description_up(self) -> None:
        class BadMigration:
            name = "x"

        bad = BadMigration()
        assert not isinstance(bad, Migration)


# -- Registration --


class TestRegistration:
    def test_empty_registry_has_no_migrations(self) -> None:
        registry = MigrationRegistry()
        assert registry.names() == []
        assert list(registry) == []

    def test_register_appends_in_order(self) -> None:
        registry = MigrationRegistry()
        registry.register(_StubMigration("m001"))
        registry.register(_StubMigration("m002"))
        registry.register(_StubMigration("m003"))
        assert registry.names() == ["m001", "m002", "m003"]

    def test_duplicate_name_raises(self) -> None:
        registry = MigrationRegistry()
        registry.register(_StubMigration("m001"))
        with pytest.raises(DuplicateMigrationError, match="m001"):
            registry.register(_StubMigration("m001"))

    def test_duplicate_check_is_name_only_not_identity(self) -> None:
        registry = MigrationRegistry()
        registry.register(_StubMigration("m001", description="first"))
        with pytest.raises(DuplicateMigrationError):
            registry.register(_StubMigration("m001", description="second"))


# -- Pending diff --


class TestPendingDiff:
    def test_pending_with_no_applied_returns_all(self) -> None:
        registry = MigrationRegistry()
        registry.register(_StubMigration("m001"))
        registry.register(_StubMigration("m002"))
        pending = registry.pending(applied=set())
        assert [m.name for m in pending] == ["m001", "m002"]

    def test_pending_with_all_applied_returns_empty(self) -> None:
        registry = MigrationRegistry()
        registry.register(_StubMigration("m001"))
        registry.register(_StubMigration("m002"))
        pending = registry.pending(applied={"m001", "m002"})
        assert pending == []

    def test_pending_preserves_registration_order(self) -> None:
        registry = MigrationRegistry()
        registry.register(_StubMigration("m001"))
        registry.register(_StubMigration("m002"))
        registry.register(_StubMigration("m003"))
        pending = registry.pending(applied={"m002"})
        assert [m.name for m in pending] == ["m001", "m003"]

    def test_pending_accepts_iterable_not_just_set(self) -> None:
        registry = MigrationRegistry()
        registry.register(_StubMigration("m001"))
        registry.register(_StubMigration("m002"))
        pending = registry.pending(applied=["m001"])
        assert [m.name for m in pending] == ["m002"]


# -- Schema-ahead detection --


class TestSchemaAhead:
    def test_no_schema_ahead_when_applied_subset(self) -> None:
        registry = MigrationRegistry()
        registry.register(_StubMigration("m001"))
        registry.register(_StubMigration("m002"))
        ahead = registry.schema_ahead(applied={"m001"})
        assert ahead == []

    def test_schema_ahead_detects_unknown_applied(self) -> None:
        registry = MigrationRegistry()
        registry.register(_StubMigration("m001"))
        ahead = registry.schema_ahead(applied={"m001", "m999_future"})
        assert ahead == ["m999_future"]

    def test_schema_ahead_preserves_applied_order(self) -> None:
        registry = MigrationRegistry()
        registry.register(_StubMigration("m001"))
        ahead = registry.schema_ahead(applied=["m001", "m998", "m999"])
        assert ahead == ["m998", "m999"]

    def test_empty_registry_treats_all_applied_as_ahead(self) -> None:
        registry = MigrationRegistry()
        ahead = registry.schema_ahead(applied={"m001"})
        assert ahead == ["m001"]
