"""Shared-key association writes are user-scoped (adr-retrieval-user-isolation).

trace_entities / trace_concepts reads carry a max_links cap: unscoped,
earlier users' rows absorb the window (first-writer seniority) and a late
user's trace links cross-user -- edges read-masked at query time, so the
symptom is within-user graph starvation, not leakage. The stub honors the
user_id kwarg, so these tests pin behavior: an unscoped read sees the other
user's rows first and takes the wrong branch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from recollect.config import MemoryConfig
from recollect.core import CognitiveMemory
from recollect.llm.types import Entity, ExtractionResult

# trace_id -> user_id, mirroring the memory_traces JOIN the store performs.
# Six u1 rows exceed the default max_links window (5): an unscoped read
# returns u1 rows only and u2's siblings never link.
_OWNER = {
    "a1": "u1",
    "a2": "u1",
    "a3": "u1",
    "a4": "u1",
    "a5": "u1",
    "a6": "u1",
    "b1": "u2",
    "b2": "u2",
}


def _install_scoped_reads(entity_index: AsyncMock) -> None:
    async def get(
        key: str, *, limit: int = 20, user_id: str | None = None, **_: Any
    ) -> list[str]:
        rows = [t for t, u in _OWNER.items() if user_id is None or u == user_id]
        return rows[:limit]

    entity_index.get_traces_by_entity = AsyncMock(side_effect=get)
    entity_index.get_traces_by_concept = AsyncMock(side_effect=get)


def _memory_with_extraction(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    config: MemoryConfig | None = None,
) -> CognitiveMemory:
    result = ExtractionResult(
        fact_type="episodic",
        entities=[Entity(name="Elm Street")],
        concepts=["gardening"],
        significance=0.5,
    )
    extractor = AsyncMock()
    extractor.extract = AsyncMock(return_value=result)
    return CognitiveMemory(
        storage=mock_storage,
        embeddings=mock_embeddings,
        extractor=extractor,
        config=config,
    )


def _shared_key_targets(association_store: AsyncMock) -> set[str]:
    return {
        call.args[0].target_trace_id
        for call in association_store.store_association.await_args_list
        if call.args[0].association_type in ("entity", "concept")
    }


class TestAssociationWriteUserScope:
    async def test_late_user_links_own_graph_despite_full_window(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
        mock_association_store: AsyncMock,
    ) -> None:
        """u2's trace links to u2's siblings; u1's rows never absorb the cap."""
        _install_scoped_reads(mock_entity_index)
        mem = _memory_with_extraction(mock_storage, mock_embeddings)
        await mem.experience("Elm Street garden center sale", user_id="u2")
        targets = _shared_key_targets(mock_association_store)
        assert targets == {"b1", "b2"}

    async def test_no_association_row_pairs_users(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
        mock_association_store: AsyncMock,
    ) -> None:
        """No entity/concept edge targets another user's trace."""
        _install_scoped_reads(mock_entity_index)
        mem = _memory_with_extraction(mock_storage, mock_embeddings)
        await mem.experience("Elm Street garden center sale", user_id="u2")
        u1_rows = {t for t, u in _OWNER.items() if u == "u1"}
        assert not (_shared_key_targets(mock_association_store) & u1_rows)

    async def test_userless_trace_degenerates_to_unscoped_read(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_entity_index: AsyncMock,
        mock_association_store: AsyncMock,
    ) -> None:
        """user_id=None keeps single-user SDK behavior byte-identical.

        Reachable only with persona.auto_extract=false -- experience()
        refuses userless writes at defaults.
        """
        _install_scoped_reads(mock_entity_index)
        cfg = MemoryConfig()
        cfg._set("persona.auto_extract", False)
        mem = _memory_with_extraction(mock_storage, mock_embeddings, config=cfg)
        await mem.experience("Elm Street garden center sale")
        targets = _shared_key_targets(mock_association_store)
        assert targets == {"a1", "a2", "a3", "a4", "a5"}
