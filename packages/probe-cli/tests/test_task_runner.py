"""Task runner seeding: deterministic shuffle breaks temporal seams."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli.corpus import TaskQuestion, TaskSeedTrace
from probe_cli.task_runner import TaskArmRunner, _Seeding, ask
from recollect.config import MemoryConfig
from recollect.models import SURFACING_STATUSES


def _seeds(n: int) -> list[TaskSeedTrace]:
    return [
        TaskSeedTrace(id=f"s{i:02d}", text=f"trace {i}", density_tier=0)
        for i in range(n)
    ]


@pytest.fixture()
def runner() -> TaskArmRunner:
    return TaskArmRunner(
        arm_name="test",
        provider=MagicMock(),
        answer_provider=MagicMock(),
        config=MemoryConfig(),
        seed_traces_path="",
        questions_path="",
    )


def _recording_memory() -> tuple[MagicMock, list[str]]:
    order: list[str] = []
    memory = MagicMock()

    async def experience(text: str, **_: object) -> MagicMock:
        order.append(text)
        trace = MagicMock()
        trace.id = f"uuid-{len(order)}"
        return trace

    memory.experience = AsyncMock(side_effect=experience)
    return memory, order


async def test_seed_order_is_shuffled_and_deterministic(
    runner: TaskArmRunner,
) -> None:
    # Block-ordered fixture seeding fabricates temporal-association seams
    # between adjacent chains; the shuffle models interleaved arrival.
    seen: list[list[str]] = []
    for _ in range(2):
        memory, order = _recording_memory()
        id_map = await runner._seed(memory, _seeds(20), user_id="u")
        assert len(id_map) == 20
        seen.append(order)
    assert seen[0] == seen[1], "shuffle must be deterministic across runs"
    assert seen[0] != [f"trace {i}" for i in range(20)], (
        "seeding must not preserve fixture file order"
    )


def _ranked_memory(order: list[str]) -> MagicMock:
    memory = MagicMock()
    memory._embeddings.generate_embedding = AsyncMock(return_value=[0.1] * 4)
    ranked = []
    for i, tid in enumerate(order):
        trace = MagicMock()
        trace.id = tid
        ranked.append((trace, 1.0 - i * 0.1))
    memory.storage.vectors.search_semantic = AsyncMock(return_value=ranked)
    return memory


async def test_raw_rank_is_position_of_required_trace(
    runner: TaskArmRunner,
) -> None:
    memory = _ranked_memory(["a", "b", "c"])
    question = MagicMock()
    question.question = "q"
    rank = await runner._raw_rank(memory, question, {"b"}, user_id="u")
    assert rank == 2


async def test_raw_rank_none_when_absent_from_top(
    runner: TaskArmRunner,
) -> None:
    memory = _ranked_memory(["a", "b", "c"])
    question = MagicMock()
    question.question = "q"
    rank = await runner._raw_rank(memory, question, {"zz"}, user_id="u")
    assert rank is None


async def test_persona_lines_read_is_status_filtered_and_unbounded(
    runner: TaskArmRunner,
) -> None:
    # The priming block is the harness twin of reflect: a windowed read
    # would measure the window, not the promoted graph.
    memory = MagicMock()
    memory.storage.facts.get_persona_facts = AsyncMock(return_value=[])
    await runner._persona_lines(memory, user_id="u1")
    memory.storage.facts.get_persona_facts.assert_awaited_once_with(
        limit=None, user_id="u1", statuses=SURFACING_STATUSES
    )


async def test_an_answer_records_the_window_it_was_given(
    runner: TaskArmRunner,
) -> None:
    # The window rides the answer's own retrieval call: an aged first read
    # is decomposable from the run file without a second exposure.
    thoughts = [
        MagicMock(trace=MagicMock(id="uuid-1"), relevance=0.91, reconstruction="a"),
        MagicMock(trace=MagicMock(id="fact-0001"), relevance=0.5, reconstruction="b"),
    ]
    memory = MagicMock()
    memory.think_about = AsyncMock(return_value=thoughts)
    runner._answer_provider.complete = AsyncMock(return_value="unknown")
    question = TaskQuestion(
        id="q1", question="?", answers=["noon"], answer_type="time", tier_label="any"
    )
    seeding = _Seeding("u", "probe-test-seed-1", {"s00": "uuid-1"}, [question])

    result = await runner._answer_one(memory, question, [], seeding)

    assert result.window == ["1. s00 0.910", "2. ~fact-000 0.500"]
    memory.think_about.assert_awaited_once()


async def test_ask_keeps_the_task_arm_answer_budget_by_default() -> None:
    provider = MagicMock()
    provider.complete = AsyncMock(return_value="answer")
    await ask(provider, "question", "context")
    assert provider.complete.await_args.kwargs["max_tokens"] == 256
