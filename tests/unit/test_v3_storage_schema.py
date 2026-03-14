"""Tests for storage association conversion, vector formatting, and schema.

Focus: association round-trips, pgvector string formatting, schema validity.
"""

from __future__ import annotations

from datetime import UTC, datetime

from recollect.models import Association, MemoryTrace
from recollect.pool import PoolManager
from recollect.storage_utils import (
    association_to_params,
    row_to_association,
    trace_to_params,
)

# -- Association conversion --


class TestAssociationConversion:
    def test_association_to_params(self) -> None:
        assoc = Association(
            source_trace_id="src-1",
            target_trace_id="tgt-1",
            association_type="causal",
            weight=0.7,
            forward_strength=0.6,
            backward_strength=0.4,
        )
        params = association_to_params(assoc)

        assert params["source_trace_id"] == "src-1"
        assert params["target_trace_id"] == "tgt-1"
        assert params["association_type"] == "causal"
        assert params["weight"] == 0.7
        assert params["forward_strength"] == 0.6
        assert params["backward_strength"] == 0.4

    def test_row_to_association(self) -> None:
        now = datetime.now(UTC)
        row = {
            "id": "assoc-1",
            "source_trace_id": "src-1",
            "target_trace_id": "tgt-1",
            "association_type": "semantic",
            "weight": 0.5,
            "forward_strength": 0.5,
            "backward_strength": 0.5,
            "activation_count": 2,
            "last_activation": now,
            "created_at": now,
        }
        assoc = row_to_association(row)

        assert assoc.source_trace_id == "src-1"
        assert assoc.target_trace_id == "tgt-1"
        assert assoc.activation_count == 2


# -- Vector formatting --


class TestVectorFormatting:
    def test_embedding_to_pgvector_string(self) -> None:
        trace = MemoryTrace(content="t", pattern={}, embedding=[0.1, 0.2, 0.3])
        params = trace_to_params(trace)
        assert params["embedding"] == "[0.1,0.2,0.3]"

    def test_none_embedding_stays_none(self) -> None:
        trace = MemoryTrace(content="t", pattern={}, embedding=None)
        params = trace_to_params(trace)
        assert params["embedding"] is None


# -- Schema --


class TestSchema:
    def test_contains_required_tables(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "memory_traces" in schema
        assert "associations" in schema

    def test_uses_pgvector_extension(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "CREATE EXTENSION" in schema
        assert "vector" in schema

    def test_uses_pg_trgm_extension(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "pg_trgm" in schema

    def test_contains_new_tables(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "trace_entities" in schema
        assert "trace_concepts" in schema
        assert "persona_facts" in schema

    def test_creates_vector_column(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "vector(768)" in schema

    def test_associations_has_pair_key(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "pair_key" in schema
        assert "idx_assoc_pair_key" in schema

    def test_has_trgm_index(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "idx_trace_entities_trgm" in schema

    def test_has_content_fts_index(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "idx_traces_content_fts" in schema

    def test_has_content_tsv_column(self) -> None:
        schema = PoolManager.get_schema_sql()
        assert "content_tsv" in schema
