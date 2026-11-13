"""Entity-reach replicate arms and the pair-label fixture load."""

from __future__ import annotations

from pathlib import Path

import pytest
from probe_cli.arm import load_arm
from probe_cli.corpus import load_entity_pair_labels

_FIX = Path(__file__).parent.parent / "fixtures" / "task"


@pytest.mark.parametrize("seeding", ["s1", "s2", "s3"])
def test_replicate_arm_loads_as_a_one_run_d0_task_arm(
    monkeypatch: pytest.MonkeyPatch, seeding: str
) -> None:
    monkeypatch.setenv("RECOLLECT_PROBE_TASK_DB_URL", "postgresql://probe/probe_task")
    reference = load_arm(_FIX / "after-d0-r1.toml")

    arm = load_arm(_FIX / f"reach-d0-{seeding}.toml")

    assert arm.name == f"task-reach-d0-{seeding}"
    assert arm.runs == 1
    assert arm.task.enabled
    assert arm.task.density_tier == 0
    assert arm.task.db_url == "postgresql://probe/probe_task"
    assert arm.extraction.pydantic_ai_model == reference.extraction.pydantic_ai_model
    assert arm.task.answer_model == reference.task.answer_model


def test_pair_labels_load_with_unordered_keys_unique() -> None:
    labels = load_entity_pair_labels(_FIX / "entity_pair_labels.jsonl")

    keys = [frozenset((row.a, row.b)) for row in labels]
    assert labels
    assert len(set(keys)) == len(keys)
