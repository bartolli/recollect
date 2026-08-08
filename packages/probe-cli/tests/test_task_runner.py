"""Task runner seeding: deterministic shuffle breaks temporal seams."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli.corpus import TaskSeedTrace
from probe_cli.task_runner import TaskArmRunner
from recollect.config import MemoryConfig


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
