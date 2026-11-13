"""Entity registry: schema single-sourcing, m007 twin, public reader."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from recollect.bootstrap.migrations import default_registry, m007_entity_registry
from recollect.core import CognitiveMemory
from recollect.models import KnownEntity
from recollect.pool import ENTITIES_SQL, ENTITY_NAME_KEY_SQL, SCHEMA_SQL


class TestNameKeySingleSource:
    def test_schema_generates_key_from_the_shared_expression(self) -> None:
        generated = f"GENERATED ALWAYS AS ({ENTITY_NAME_KEY_SQL.format('name')})"
        assert generated in ENTITIES_SQL
        assert ENTITIES_SQL in SCHEMA_SQL

    def test_identity_is_unique_per_user_including_null(self) -> None:
        assert "ON entities (name_key, user_id) NULLS NOT DISTINCT" in ENTITIES_SQL

    def test_key_collapses_whitespace_before_trimming(self) -> None:
        assert ENTITY_NAME_KEY_SQL.index("regexp_replace") > (
            ENTITY_NAME_KEY_SQL.index("btrim")
        )

    def test_registry_precedes_the_mentions_that_reference_it(self) -> None:
        assert SCHEMA_SQL.index("CREATE TABLE IF NOT EXISTS entities") < (
            SCHEMA_SQL.index("CREATE TABLE IF NOT EXISTS trace_entities")
        )
        assert "entity_id TEXT REFERENCES entities(id),\n" in SCHEMA_SQL
        assert "idx_trace_entities_entity" in SCHEMA_SQL


class TestM007EntityRegistry:
    def test_registered_after_m006(self) -> None:
        names = default_registry().names()
        assert names.index("m007_entity_registry") == (
            names.index("m006_decay_anchor") + 1
        )

    async def test_up_reuses_schema_ddl_idempotently(self) -> None:
        conn = AsyncMock()
        await m007_entity_registry.up(conn)
        statements = [call.args[0] for call in conn.execute.await_args_list]
        assert statements[0] == ENTITIES_SQL
        assert "ADD COLUMN IF NOT EXISTS entity_id" in statements[1]
        assert "CREATE INDEX IF NOT EXISTS idx_trace_entities_entity" in statements[2]


class TestKnownEntitiesSurface:
    def test_trace_count_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            KnownEntity(id="e-1", name="Alex", trace_count=-1)

    async def test_entities_delegates_with_user_scope(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
    ) -> None:
        known = KnownEntity(id="e-1", user_id="u1", name="Alex", trace_count=2)
        mock_entity_index.get_entities.return_value = [known]
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        assert await mem.entities(user_id="u1", limit=10) == [known]
        mock_entity_index.get_entities.assert_awaited_once_with(user_id="u1", limit=10)
