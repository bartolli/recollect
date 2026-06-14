"""Situational grounding (slice-2): the recall bridge recovers a below-floor
persona fact whose source_trace is strongly token-activated, and only then.

Drives the real think_about path: a seed trace flows through search_semantic so
_activate_recall_tokens produces token_activated, and get_activated_trace_ids
returns a hop row whose propagated_sim is anchor_cosine * hop_decay(0.85) *
strength * significance. significance tunes prop across/under the bridge floor.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace, PersonaFact

_EMB = [0.1] * 768
# prop = 0.6 (anchor cosine) * 0.85 (hop_decay) * 1.0 (strength) * significance
_STRONG = 0.9  # -> prop 0.459 >= floor 0.35
_WEAK = 0.4    # -> prop 0.204 <  floor 0.35

Run = Callable[..., Awaitable[list]]


def _fact() -> PersonaFact:
    return PersonaFact(
        id="f-pref",
        subject="Sarah",
        predicate="prefers",
        object="quiet places",
        category="preference",
        content="Sarah prefers quiet restaurants",
        status="promoted",
        confidence=0.9,
        embedding=_EMB,
        source_trace_id="T",
    )


@pytest.fixture
def bridge_run(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_fact_store: AsyncMock,
    mock_vector_index: AsyncMock,
    mock_entity_index: AsyncMock,
    mock_concept_embedding_store: AsyncMock,
    mock_recall_token_store: AsyncMock,
) -> Run:
    async def _run(*, significance: float, activated: bool = True) -> list:
        anchor = MemoryTrace(id="anchor", content="anchor topic", embedding=_EMB)
        mock_vector_index.search_semantic.return_value = [(anchor, 0.6)]
        mock_recall_token_store.get_activated_trace_ids.return_value = (
            [("T", "tok1", "label", 1.0, significance, "anchor")] if activated else []
        )
        fact = _fact()
        mock_fact_store.search_facts_semantic.return_value = [(fact, 0.40)]
        mock_fact_store.get_persona_facts.return_value = [fact]
        mock_fact_store.get_persona_facts_by_entities.return_value = []
        mock_entity_index.match_entities.return_value = []
        mock_concept_embedding_store.get_max_sim_per_owner.return_value = {}
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        # Bridge ships OFF (packaged 0.0); enable at the corpus-validated floor.
        mem._config._set("persona.bridge_activation_floor", 0.35)
        return await mem.think_about("unrelated weekend plans")

    return _run


def _surfaced(thoughts: list, needle: str) -> bool:
    return any(needle in (t.reconstruction or "").lower() for t in thoughts)


class TestRecallBridge:
    async def test_strong_activation_recovers_below_floor_fact(
        self, bridge_run: Run
    ) -> None:
        thoughts = await bridge_run(significance=_STRONG)
        assert _surfaced(thoughts, "quiet")

    async def test_weak_activation_does_not_recover(self, bridge_run: Run) -> None:
        # prop below the bridge floor is incidental, not a situational link --
        # the gate that fixed the slice-1c flood.
        thoughts = await bridge_run(significance=_WEAK)
        assert not _surfaced(thoughts, "quiet")

    async def test_unactivated_below_floor_fact_stays_dropped(
        self, bridge_run: Run
    ) -> None:
        thoughts = await bridge_run(significance=_STRONG, activated=False)
        assert not _surfaced(thoughts, "quiet")
