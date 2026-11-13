"""Relation assertions: schema shape, m008 twin, EntityEdge, public reader."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from recollect import EntityEdge
from recollect.bootstrap.migrations import default_registry, m008_relation_assertions
from recollect.core import CognitiveMemory
from recollect.llm.types import Entity, ExtractionResult, Relation
from recollect.pool import SCHEMA_SQL

EDGE = EntityEdge(
    source="Alex",
    source_entity_id="e-alex",
    relation="works_at",
    target="Acme",
    target_entity_id=None,
    weight=0.9,
    trace_count=2,
)


class TestSchemaShape:
    def test_assertions_cascade_with_their_trace(self) -> None:
        table = SCHEMA_SQL[
            SCHEMA_SQL.index("CREATE TABLE IF NOT EXISTS entity_relations") :
        ]
        table = table[: table.index(");")]
        assert (
            "source_trace_id TEXT REFERENCES memory_traces(id) ON DELETE CASCADE"
            in (table)
        )
        assert "source_entity_id TEXT REFERENCES entities(id),\n" in table
        assert "target_entity_id TEXT REFERENCES entities(id)\n" in table

    def test_registry_precedes_the_assertions_that_reference_it(self) -> None:
        assert SCHEMA_SQL.index("CREATE TABLE IF NOT EXISTS entities") < (
            SCHEMA_SQL.index("CREATE TABLE IF NOT EXISTS entity_relations")
        )

    def test_assertion_key_replaces_the_global_triple(self) -> None:
        assert (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_rel_assertion\n"
            "    ON entity_relations(source_trace_id, source_entity, relation,"
            " target_entity);"
        ) in SCHEMA_SQL
        for kept in ("idx_entity_rel_source_id", "idx_entity_rel_target_id"):
            assert kept in SCHEMA_SQL
        for dropped in (
            "idx_entity_rel_pair",
            "idx_entity_rel_source\n",
            "idx_entity_rel_target\n",
        ):
            assert dropped not in SCHEMA_SQL


class TestM008RelationAssertions:
    def test_registered_after_m007(self) -> None:
        names = default_registry().names()
        assert names[-1] == "m008_relation_assertions"
        assert names.index("m008_relation_assertions") == (
            names.index("m007_entity_registry") + 1
        )

    async def test_up_is_idempotent_ddl(self) -> None:
        conn = AsyncMock()
        await m008_relation_assertions.up(conn)
        ddl = "\n".join(call.args[0] for call in conn.execute.await_args_list)
        assert ddl.count("ADD COLUMN IF NOT EXISTS") == 2
        assert ddl.count("DROP INDEX IF EXISTS") == 3
        assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_rel_assertion" in ddl
        # Endpoint resolution reads trace_entities by trace_id on every insert.
        for schema in (ddl, SCHEMA_SQL):
            assert "CREATE INDEX IF NOT EXISTS idx_trace_entities_trace" in schema
        assert "DROP CONSTRAINT IF EXISTS entity_relations_source_trace_id_fkey" in ddl
        assert "ON DELETE CASCADE" in ddl


class TestRelationModels:
    @pytest.mark.parametrize(
        ("update", "field"),
        [({"trace_count": 0}, "trace_count"), ({"weight": 1.5}, "weight")],
    )
    def test_edge_rejects_out_of_range(
        self, update: dict[str, object], field: str
    ) -> None:
        with pytest.raises(ValidationError) as caught:
            EntityEdge.model_validate({**EDGE.model_dump(), **update})
        assert [e["loc"] for e in caught.value.errors()] == [(field,)]


class TestEntityEdgesSurface:
    @pytest.fixture()
    def mem(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> CognitiveMemory:
        return CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)

    async def test_delegates_with_user_scope(
        self, mem: CognitiveMemory, mock_entity_relation_store: AsyncMock
    ) -> None:
        mock_entity_relation_store.get_edges.return_value = [EDGE]
        assert await mem.entity_edges("e-alex", user_id="u1", limit=5) == [EDGE]
        mock_entity_relation_store.get_edges.assert_awaited_once_with(
            "e-alex", user_id="u1", limit=5
        )

    async def test_rejects_empty_id(
        self, mem: CognitiveMemory, mock_entity_relation_store: AsyncMock
    ) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await mem.entity_edges("")
        mock_entity_relation_store.get_edges.assert_not_awaited()


class TestWriteOrder:
    async def test_experience_stores_mentions_before_relations(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
        mock_entity_relation_store: AsyncMock,
    ) -> None:
        # Endpoint resolution reads the trace's mentions in the relation insert.
        calls: list[str] = []
        mock_entity_index.store_trace_entities.side_effect = lambda *_: calls.append(
            "mentions"
        )
        mock_entity_relation_store.store_relation.side_effect = lambda *_: calls.append(
            "relation"
        )
        extractor = AsyncMock()
        extractor.extract.return_value = ExtractionResult(
            entities=[Entity(name="Alex"), Entity(name="Acme")],
            relations=[Relation(source="Alex", relation="works_at", target="Acme")],
        )
        mem = CognitiveMemory(
            storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
        )
        trace = await mem.experience("Alex works at Acme", user_id="u1")
        assert calls == ["mentions", "relation"]
        rel = mock_entity_relation_store.store_relation.await_args.args[0]
        assert rel.source_trace_id == trace.id
