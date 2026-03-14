"""Tests for trace_entities, trace_concepts, persona_facts schema and helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from recollect.models import PersonaFact
from recollect.pool import PoolManager
from recollect.storage_utils import persona_fact_to_params, row_to_persona_fact


class TestSchemaNewTables:
    def test_schema_contains_trace_entities(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "trace_entities" in schema
        assert "entity_name TEXT NOT NULL" in schema
        assert "ON DELETE CASCADE" in schema

    def test_schema_contains_trace_concepts(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "trace_concepts" in schema
        assert "concept TEXT NOT NULL" in schema

    def test_schema_contains_persona_facts(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "persona_facts" in schema
        assert "subject TEXT NOT NULL" in schema
        assert "predicate TEXT NOT NULL" in schema
        assert "superseded_by TEXT" in schema

    def test_schema_has_entity_index(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "idx_trace_entities_name" in schema

    def test_schema_has_concept_index(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "idx_trace_concepts_concept" in schema

    def test_schema_has_persona_spo_index(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "idx_persona_facts_spo" in schema

    def test_schema_has_pg_trgm_extension(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "pg_trgm" in schema

    def test_schema_has_pair_key_column(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "pair_key" in schema

    def test_schema_has_pair_key_index(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "idx_assoc_pair_key" in schema

    def test_schema_has_trgm_index(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "idx_trace_entities_trgm" in schema

    def test_schema_has_content_fts_index(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "idx_traces_content_fts" in schema

    def test_schema_has_content_tsv_column(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "content_tsv" in schema


class TestPersonaFactConversion:
    def test_to_params_roundtrip(self) -> None:
        now = datetime.now(UTC)
        fact = PersonaFact(
            id="fact-1",
            subject="Sarah",
            predicate="works_at",
            object="Acme",
            category="identity",
            content="Sarah works at Acme",
            source_trace_id="trace-1",
            confidence=0.9,
            created_at=now,
            updated_at=now,
        )
        params = persona_fact_to_params(fact)
        assert params["id"] == "fact-1"
        assert params["subject"] == "Sarah"
        assert params["predicate"] == "works_at"
        assert params["object"] == "Acme"
        assert params["category"] == "identity"
        assert params["confidence"] == 0.9

    def test_row_to_persona_fact(self) -> None:
        now = datetime.now(UTC)
        row = {
            "id": "fact-1",
            "subject": "Sarah",
            "predicate": "is_allergic_to",
            "object": "shellfish",
            "category": "health",
            "content": "Sarah is allergic to shellfish",
            "source_trace_id": "trace-1",
            "confidence": 0.95,
            "created_at": now,
            "updated_at": now,
            "superseded_by": None,
        }
        fact = row_to_persona_fact(row)
        assert fact.subject == "Sarah"
        assert fact.predicate == "is_allergic_to"
        assert fact.category == "health"
        assert fact.confidence == 0.95
