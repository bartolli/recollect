"""Fusion-screen replay reproduces think_about's window with no storage write."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from probe_cli.fusion_screen import _ScreenMemory, _window_key
from recollect.models import MemoryTrace, PersonaFact

_EMB_DIM = 768


def _storage(
    scored: list[tuple[MemoryTrace, float]],
    facts: list[tuple[PersonaFact, float]],
) -> MagicMock:
    storage = MagicMock()
    storage.vectors.search_semantic = AsyncMock(return_value=scored)
    storage.vectors.spread_activation = AsyncMock(return_value=[])
    storage.entities.match_entities = AsyncMock(return_value=[])
    storage.traces = AsyncMock()
    storage.traces.get_traces_bulk = AsyncMock(return_value=[])
    storage.traces.reactivate_trace = AsyncMock(return_value=True)
    storage.concept_embeddings.get_max_sim_per_owner = AsyncMock(return_value={})
    storage.recall_tokens.get_activated_trace_ids = AsyncMock(return_value=[])
    storage.facts.search_facts_semantic = AsyncMock(return_value=facts)
    storage.facts.get_persona_facts_by_entities = AsyncMock(return_value=[])
    storage.facts.get_persona_facts = AsyncMock(return_value=[])
    return storage


def _embeddings() -> AsyncMock:
    embeddings = AsyncMock()
    embeddings.generate_embedding = AsyncMock(return_value=[0.1] * _EMB_DIM)
    embeddings.dimensions = _EMB_DIM
    return embeddings


def _writes(storage: MagicMock) -> list[int]:
    traces = storage.traces
    return [
        traces.reactivate_trace.await_count,
        traces.apply_strength_factor.await_count,
        traces.mark_retrieved.await_count,
    ]


async def test_replay_matches_think_about_window_without_writes() -> None:
    scored = [
        (MemoryTrace(id="gone", content="retracted", status="forgotten"), 0.95),
        (MemoryTrace(id="big", content="x" * 2000), 0.9),
        (MemoryTrace(id="arch", content="faded note", status="archived"), 0.7),
        (MemoryTrace(id="t1", content="first"), 0.6),
        (MemoryTrace(id="t2", content="second"), 0.5),
        (MemoryTrace(id="t3", content="third"), 0.5),
    ]
    fact = PersonaFact(
        subject="alex",
        predicate="likes",
        object="tea",
        content="alex likes tea",
        category="preference",
        confidence=0.9,
        status="promoted",
    )
    storage = _storage(scored, [(fact, 0.95)])
    memory = _ScreenMemory(storage=storage, embeddings=_embeddings())

    call = await memory.capture_call(
        "what does alex like", label="q1", token_budget=120, user_id="alex"
    )
    before = _writes(storage)
    replay = await memory.replay(call, form="prod", weight=None, token_budget=120)

    assert _window_key(replay.window) == _window_key(call.window)
    assert {t.trace.id for t in call.window} >= {"arch", "t1"}
    assert "big" not in {t.trace.id for t in call.window}
    assert len(call.facts) == 1
    assert _writes(storage) == before
