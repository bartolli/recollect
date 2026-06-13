"""Read-path recall floor: blended-S floor, safety bypass, pin-when-ranked."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.models import PersonaFact

_EMB_DIM = 768

Recall = Callable[[list[tuple[PersonaFact, float]]], Awaitable[list]]


def _fact(
    fid: str,
    obj: str,
    category: str,
    content: str,
    *,
    status: str = "promoted",
    confidence: float = 0.9,
) -> PersonaFact:
    return PersonaFact(
        id=fid,
        subject="Alex",
        predicate="likes",
        object=obj,
        category=category,
        content=content,
        status=status,
        confidence=confidence,
        embedding=[0.0] * _EMB_DIM,
    )


@pytest.fixture
def recall(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_fact_store: AsyncMock,
    mock_vector_index: AsyncMock,
    mock_entity_index: AsyncMock,
    mock_concept_embedding_store: AsyncMock,
    mock_recall_token_store: AsyncMock,
) -> Recall:
    # Surface the given (fact, blended-S) pairs through think_about with every
    # trace channel empty and concept blend off, so S == the provided score.
    async def _run(pairs: list[tuple[PersonaFact, float]]) -> list:
        mock_fact_store.search_facts_semantic.return_value = list(pairs)
        mock_fact_store.get_persona_facts.return_value = [f for f, _ in pairs]
        mock_fact_store.get_persona_facts_by_entities.return_value = []
        mock_vector_index.search_semantic.return_value = []
        mock_entity_index.match_entities.return_value = []
        mock_concept_embedding_store.get_max_sim_per_owner.return_value = {}
        mock_recall_token_store.get_activated_trace_ids.return_value = []
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        return await mem.think_about("unrelated query about weekend plans")

    return _run


def _surfaced(thoughts: list, needle: str) -> bool:
    return any(needle in t.reconstruction.lower() for t in thoughts)


class TestRecallFloor:
    async def test_non_safety_fact_below_floor_does_not_surface(
        self, recall: Recall
    ) -> None:
        fact = _fact("f-pref", "jazz", "preference", "Alex likes jazz")
        assert not _surfaced(await recall([(fact, 0.50)]), "jazz")

    async def test_relevant_fact_above_floor_surfaces(self, recall: Recall) -> None:
        fact = _fact("f-pref", "jazz", "preference", "Alex likes jazz")
        assert _surfaced(await recall([(fact, 0.70)]), "jazz")

    async def test_safety_category_below_floor_bypasses(self, recall: Recall) -> None:
        # health is in the recall safety-bypass {health, dietary}.
        fact = _fact("f-health", "peanuts", "health", "Alex is allergic to peanuts")
        assert _surfaced(await recall([(fact, 0.10)]), "peanut")

    async def test_constraint_below_floor_is_floored(self, recall: Recall) -> None:
        # constraint is process-governance-overloaded; the recall bypass is
        # {health, dietary}, so a below-floor constraint fact is floor-gated
        # (the global write-path _FAST_TRACK_CATEGORIES still includes it).
        fact = _fact("f-con", "vpn", "constraint", "Access requires the VPN")
        assert not _surfaced(await recall([(fact, 0.10)]), "vpn")

    async def test_dietary_below_floor_bypasses(self, recall: Recall) -> None:
        fact = _fact("f-diet", "gluten", "dietary", "Alex avoids gluten")
        assert _surfaced(await recall([(fact, 0.10)]), "gluten")

    async def test_pinned_fact_below_floor_surfaces_when_ranked(
        self, recall: Recall
    ) -> None:
        # A pin that reaches top-k (here the only fact) is floor-exempt; recall
        # honors a pin only when it ranks, the primer carries it always.
        pin = _fact(
            "f-pin", "skiing", "preference", "Alex loves skiing",
            status="pinned", confidence=1.0,
        )
        assert _surfaced(await recall([(pin, 0.40)]), "skiing")

    async def test_pinned_fact_below_floor_not_surfaced_when_outranked(
        self, recall: Recall
    ) -> None:
        # No reserved slots: max_facts_per_query=5 more-relevant facts fill
        # top-k, so the below-floor pin is rank-cut before the floor exemption
        # can see it -- it does not surface (the primer carries it).
        pin = _fact(
            "f-pin", "skiing", "preference", "Alex loves skiing",
            status="pinned", confidence=1.0,
        )
        fillers = [
            _fact(f"f{i}", f"topic{i}", "preference", f"Alex likes topic{i}")
            for i in range(5)
        ]
        pairs = [(pin, 0.40), *[(f, 0.90) for f in fillers]]
        assert not _surfaced(await recall(pairs), "skiing")
