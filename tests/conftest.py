"""Shared test fixtures for Memory SDK v3."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.llm.types import ExtractionResult

_EMB_DIM = 768


def _fake_embedding(seed: float = 0.1) -> list[float]:
    return [seed + i * 0.001 for i in range(_EMB_DIM)]


@pytest.fixture()
def mock_trace_store() -> AsyncMock:
    s = AsyncMock()
    s.store_trace = AsyncMock(return_value="id")
    s.get_trace = AsyncMock(return_value=None)
    s.get_traces_bulk = AsyncMock(return_value=[])
    s.delete_trace = AsyncMock(return_value=True)
    s.update_trace_strength = AsyncMock()
    s.mark_activated = AsyncMock()
    s.mark_retrieved = AsyncMock()
    s.get_recent_traces = AsyncMock(return_value=[])
    s.get_unconsolidated_traces = AsyncMock(return_value=[])
    s.mark_consolidated = AsyncMock()
    s.get_traces_by_session = AsyncMock(return_value=[])
    return s


@pytest.fixture()
def mock_vector_index() -> AsyncMock:
    v = AsyncMock()
    v.search_semantic = AsyncMock(return_value=[])
    v.spread_activation = AsyncMock(return_value=[])
    return v


@pytest.fixture()
def mock_entity_index() -> AsyncMock:
    e = AsyncMock()
    e.store_trace_entities = AsyncMock()
    e.store_trace_concepts = AsyncMock()
    e.get_traces_by_entity = AsyncMock(return_value=[])
    e.get_traces_by_concept = AsyncMock(return_value=[])
    e.match_entities = AsyncMock(return_value=[])
    return e


@pytest.fixture()
def mock_association_store() -> AsyncMock:
    a = AsyncMock()
    a.store_association = AsyncMock()
    a.get_associations = AsyncMock(return_value=[])
    return a


@pytest.fixture()
def mock_fact_store() -> AsyncMock:
    f = AsyncMock()
    f.store_persona_fact = AsyncMock(return_value="fact-id")
    f.get_persona_facts = AsyncMock(return_value=[])
    f.get_persona_facts_by_entities = AsyncMock(return_value=[])
    f.get_facts_by_entities_and_scopes = AsyncMock(return_value=[])
    f.supersede_persona_fact = AsyncMock(return_value="new-id")
    f.delete_persona_fact = AsyncMock(return_value=True)
    f.increment_mention_count = AsyncMock(return_value=2)
    f.update_fact_status = AsyncMock()
    f.get_facts_by_context = AsyncMock(return_value=[])
    f.search_facts_semantic = AsyncMock(return_value=[])
    return f


@pytest.fixture()
def mock_entity_relation_store() -> AsyncMock:
    er = AsyncMock()
    er.store_relation = AsyncMock(return_value="rel-id")
    er.get_relations = AsyncMock(return_value=[])
    er.get_related_entities = AsyncMock(return_value=[])
    er.delete_by_trace = AsyncMock(return_value=0)
    return er


@pytest.fixture()
def mock_concept_embedding_store() -> AsyncMock:
    ce = AsyncMock()
    ce.store_concept_embeddings = AsyncMock()
    ce.get_max_sim_per_owner = AsyncMock(return_value={})
    ce.delete_by_owner = AsyncMock(return_value=0)
    return ce


@pytest.fixture()
def mock_session_store() -> AsyncMock:
    s = AsyncMock()
    s.create_session = AsyncMock(return_value="session-id")
    s.get_session = AsyncMock(return_value=None)
    s.end_session = AsyncMock()
    s.get_sessions = AsyncMock(return_value=[])
    s.update_session = AsyncMock()
    return s


@pytest.fixture()
def mock_recall_token_store() -> AsyncMock:
    rt = AsyncMock()
    rt.create_token = AsyncMock(return_value="token-id")
    rt.stamp_traces = AsyncMock()
    rt.get_activated_trace_ids = AsyncMock(return_value=[])
    rt.find_token_by_traces = AsyncMock(return_value=None)
    rt.reinforce_tokens = AsyncMock()
    rt.get_tokens_for_traces = AsyncMock(return_value=[])
    rt.delete_by_trace = AsyncMock(return_value=0)
    rt.decay_inactive = AsyncMock(return_value=0)
    return rt


@pytest.fixture()
def mock_storage(
    mock_trace_store: AsyncMock,
    mock_vector_index: AsyncMock,
    mock_entity_index: AsyncMock,
    mock_association_store: AsyncMock,
    mock_fact_store: AsyncMock,
    mock_entity_relation_store: AsyncMock,
    mock_concept_embedding_store: AsyncMock,
    mock_session_store: AsyncMock,
    mock_recall_token_store: AsyncMock,
) -> MagicMock:
    """Mock StorageContext with all sub-stores."""
    ctx = MagicMock()
    ctx.traces = mock_trace_store
    ctx.vectors = mock_vector_index
    ctx.entities = mock_entity_index
    ctx.associations = mock_association_store
    ctx.facts = mock_fact_store
    ctx.entity_relations = mock_entity_relation_store
    ctx.concept_embeddings = mock_concept_embedding_store
    ctx.sessions = mock_session_store
    ctx.recall_tokens = mock_recall_token_store
    ctx.initialize = AsyncMock()
    ctx.close = AsyncMock()
    return ctx


@pytest.fixture()
def mock_embeddings() -> AsyncMock:
    e = AsyncMock()
    e.generate_embedding = AsyncMock(return_value=_fake_embedding())
    e.generate_embeddings_batch = AsyncMock(
        side_effect=lambda texts: [
            _fake_embedding(0.1 + i * 0.01) for i in range(len(texts))
        ]
    )
    e.dimensions = _EMB_DIM
    return e


@pytest.fixture()
def mock_extractor() -> AsyncMock:
    x = AsyncMock()
    x.extract = AsyncMock(
        return_value=ExtractionResult(concepts=["test"], significance=0.5)
    )
    return x
