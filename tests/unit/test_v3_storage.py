"""Tests for PostgreSQL storage trace conversion.

Focus: MemoryTrace <-> DB row round-trips.
Association, vector, and schema tests are in test_v3_storage_schema.py.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from recollect.models import MemoryTrace
from recollect.storage_utils import row_to_trace, trace_to_params

# -- Trace conversion --


class TestTraceConversion:
    """MemoryTrace <-> DB row round-trips."""

    def test_trace_to_params_serializes_all_fields(self) -> None:
        trace = MemoryTrace(
            id="trace-1",
            content="hello world",
            pattern={"concepts": ["greeting"]},
            context={"source": "test"},
            strength=0.5,
            decay_rate=0.15,
            emotional_valence=0.3,
            significance=0.7,
        )
        params = trace_to_params(trace)

        assert params["id"] == "trace-1"
        assert params["content"] == "hello world"
        assert json.loads(params["pattern"]) == {"concepts": ["greeting"]}
        assert json.loads(params["context"]) == {"source": "test"}
        assert params["strength"] == 0.5
        assert params["decay_rate"] == 0.15
        assert params["emotional_valence"] == 0.3
        assert params["significance"] == 0.7

    def test_row_to_trace_reconstructs_model(self) -> None:
        now = datetime.now(UTC)
        row = {
            "id": "trace-1",
            "content": "hello world",
            "pattern": '{"concepts": ["greeting"]}',
            "context": "{}",
            "embedding": None,
            "strength": 0.5,
            "activation_count": 3,
            "retrieval_count": 1,
            "last_activation": now,
            "last_retrieval": None,
            "consolidated": False,
            "created_at": now,
            "decay_rate": 0.1,
            "emotional_valence": 0.3,
            "significance": 0.7,
        }
        trace = row_to_trace(row)

        assert trace.id == "trace-1"
        assert trace.content == "hello world"
        assert trace.pattern == {"concepts": ["greeting"]}
        assert trace.strength == 0.5
        assert trace.activation_count == 3
        assert trace.retrieval_count == 1
        assert trace.last_activation == now

    def test_roundtrip_preserves_data(self) -> None:
        original = MemoryTrace(
            content="round-trip test",
            pattern={"concepts": ["test", "roundtrip"]},
            context={"session": "abc"},
            strength=0.42,
            emotional_valence=-0.1,
            significance=0.8,
        )
        params = trace_to_params(original)

        # Simulate what the DB would return
        row = {
            "id": original.id,
            "content": params["content"],
            "pattern": params["pattern"],
            "context": params["context"],
            "embedding": None,
            "strength": params["strength"],
            "activation_count": 0,
            "retrieval_count": 0,
            "last_activation": None,
            "last_retrieval": None,
            "consolidated": False,
            "created_at": original.created_at,
            "decay_rate": params["decay_rate"],
            "emotional_valence": params["emotional_valence"],
            "significance": params["significance"],
        }
        rebuilt = row_to_trace(row)

        assert rebuilt.id == original.id
        assert rebuilt.content == original.content
        assert rebuilt.pattern == original.pattern
        assert rebuilt.strength == pytest.approx(original.strength)
        assert rebuilt.emotional_valence == pytest.approx(original.emotional_valence)

    def test_handles_dict_pattern_from_jsonb(self) -> None:
        """asyncpg returns JSONB as dicts, not strings."""
        row = {
            "id": "t1",
            "content": "test",
            "pattern": {"concepts": ["already", "parsed"]},
            "context": {},
            "embedding": None,
            "strength": 0.3,
            "activation_count": 0,
            "retrieval_count": 0,
            "last_activation": None,
            "last_retrieval": None,
            "consolidated": False,
            "created_at": datetime.now(UTC),
            "decay_rate": 0.1,
            "emotional_valence": 0.0,
            "significance": 0.1,
        }
        trace = row_to_trace(row)
        assert trace.pattern == {"concepts": ["already", "parsed"]}

    def test_parses_pgvector_embedding_string(self) -> None:
        """pgvector returns embeddings as strings like '[0.1,0.2,0.3]'."""
        row = {
            "id": "t1",
            "content": "test",
            "pattern": "{}",
            "context": "{}",
            "embedding": "[0.1,0.2,0.3]",
            "strength": 0.3,
            "activation_count": 0,
            "retrieval_count": 0,
            "last_activation": None,
            "last_retrieval": None,
            "consolidated": False,
            "created_at": datetime.now(UTC),
            "decay_rate": 0.1,
            "emotional_valence": 0.0,
            "significance": 0.1,
        }
        trace = row_to_trace(row)
        assert trace.embedding == [0.1, 0.2, 0.3]
