"""Extraction-path fact dedup is user-scoped (adr-retrieval-user-isolation).

The unscoped subject read let a later user's identical SPO increment and
promote the EARLIER user's row while the later user's own write was
swallowed -- cross-user mention crediting plus fact-channel starvation.
The stub honors the user_id kwarg, so these tests pin behavior, not call
shape: an unscoped read sees the other user's row and takes the wrong
branch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from recollect.core import CognitiveMemory
from recollect.llm.types import Entity, ExtractionResult, Relation
from recollect.models import PersonaFact


def _fact(user_id: str, obj: str = "Mediterranean food") -> PersonaFact:
    return PersonaFact(
        subject="Sarah",
        predicate="prefers",
        object=obj,
        category="preference",
        content=f"Sarah prefers {obj}",
        confidence=0.9,
        status="candidate",
        user_id=user_id,
    )


def _install_user_scoped_read(
    fact_store: AsyncMock, existing: list[PersonaFact]
) -> None:
    async def get(
        subject: str | None = None,
        *,
        limit: int = 50,
        user_id: str | None = None,
        **_: Any,
    ) -> list[PersonaFact]:
        rows = [f for f in existing if subject is None or f.subject == subject]
        if user_id is not None:
            rows = [f for f in rows if f.user_id == user_id]
        return rows[:limit]

    fact_store.get_persona_facts = AsyncMock(side_effect=get)


def _memory_with(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    relation: Relation,
) -> CognitiveMemory:
    result = ExtractionResult(
        fact_type="semantic",
        entities=[Entity(name=relation.source)],
        relations=[relation],
        significance=0.5,
    )
    extractor = AsyncMock()
    extractor.extract = AsyncMock(return_value=result)
    return CognitiveMemory(
        storage=mock_storage, embeddings=mock_embeddings, extractor=extractor
    )


class TestCrossUserIsolation:
    async def test_other_users_twin_neither_swallows_nor_credits(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        """User B's identical SPO stores B's own row; A's row untouched."""
        _install_user_scoped_read(mock_fact_store, [_fact("u1")])
        mem = _memory_with(
            mock_storage,
            mock_embeddings,
            Relation(
                source="Sarah",
                relation="prefers",
                target="Mediterranean food",
                category="preference",
                confidence=0.9,
            ),
        )
        await mem.experience("Sarah prefers Mediterranean food", user_id="u2")
        mock_fact_store.store_persona_fact.assert_awaited()
        stored = mock_fact_store.store_persona_fact.call_args.args[0]
        assert stored.user_id == "u2"
        mock_fact_store.increment_mention_count.assert_not_awaited()

    async def test_same_user_restatement_still_dedups_and_promotes(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        """Within one user the mention-count promotion path is unchanged."""
        own = _fact("u1")
        _install_user_scoped_read(mock_fact_store, [own])
        mem = _memory_with(
            mock_storage,
            mock_embeddings,
            Relation(
                source="Sarah",
                relation="prefers",
                target="Mediterranean food",
                category="preference",
                confidence=0.9,
            ),
        )
        await mem.experience("Sarah prefers Mediterranean food", user_id="u1")
        mock_fact_store.increment_mention_count.assert_awaited_once_with(own.id)
        mock_fact_store.update_fact_status.assert_awaited_once_with(
            own.id, "promoted"
        )
        mock_fact_store.store_persona_fact.assert_not_awaited()

    async def test_supersession_stays_user_local(
        self,
        mock_storage: MagicMock,
        mock_embeddings: AsyncMock,
        mock_fact_store: AsyncMock,
    ) -> None:
        """User B's different object never supersedes A's fact."""
        _install_user_scoped_read(mock_fact_store, [_fact("u1")])
        mem = _memory_with(
            mock_storage,
            mock_embeddings,
            Relation(
                source="Sarah",
                relation="prefers",
                target="Japanese food",
                category="preference",
                confidence=0.9,
            ),
        )
        await mem.experience("Sarah prefers Japanese food", user_id="u2")
        mock_fact_store.supersede_persona_fact.assert_not_awaited()
        mock_fact_store.store_persona_fact.assert_awaited()
        stored = mock_fact_store.store_persona_fact.call_args.args[0]
        assert stored.user_id == "u2"
