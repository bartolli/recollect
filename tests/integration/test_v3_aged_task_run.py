"""Integration: an aged task run on a real store, LLM scripted at the provider.

Seeding, stepping, and evaluation each open their own CognitiveMemory
against one scratch store; only the two model calls are faked.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from probe_cli.aging import Aging
from probe_cli.task_runner import TaskArmRunner
from recollect.config import MemoryConfig
from recollect.datetime_utils import now_utc
from recollect.llm.types import ExtractionResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

DB_URL = os.environ.get("DATABASE_URL", "")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
    pytest.mark.skipif(not DB_URL, reason="DATABASE_URL unset"),
]


def _url(db_name: str) -> str:
    return urlunparse(urlparse(DB_URL)._replace(path=f"/{db_name}"))


async def _admin(sql: str) -> None:
    conn = await asyncpg.connect(_url("postgres"))
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@pytest.fixture()
async def db_name() -> AsyncGenerator[str, None]:
    name = f"probe_aged_{uuid.uuid4().hex[:10]}"
    await _admin(f'CREATE DATABASE "{name}"')
    try:
        yield name
    finally:
        await _admin(f'DROP DATABASE "{name}" WITH (FORCE)')


def _runner(tmp_path: Path, db_name: str) -> TaskArmRunner:
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(
        "\n".join(
            json.dumps({"id": f"s{i}", "text": text, "density_tier": 0})
            for i, text in enumerate(
                [
                    "The garden gate key hangs behind the blue door.",
                    "Rehearsal moved to the church hall at noon.",
                    "The ferry timetable changes on the first of March.",
                ]
            )
        ),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "When is rehearsal?",
                "answers": ["noon"],
                "answer_type": "time",
                "tier_label": "any",
            }
        ),
        encoding="utf-8",
    )
    config = MemoryConfig()
    config._set("recall_tokens.enabled", False)
    config._set("consolidation.batch_size", 1)
    provider = MagicMock(model_name="scripted")
    provider.complete_structured = AsyncMock(
        return_value=ExtractionResult(concepts=["logistics"], significance=0.7)
    )
    answer = MagicMock(model_name="scripted")
    answer.complete = AsyncMock(return_value="unknown")
    return TaskArmRunner(
        arm_name="aged",
        provider=provider,
        answer_provider=answer,
        config=config,
        seed_traces_path=str(seeds),
        questions_path=str(questions),
        db_url=_url(db_name),
        aging=Aging(age_days=2, passes_per_day=2),
    )


async def test_aged_task_run_steps_and_evaluates_a_real_store(
    tmp_path: Path, db_name: str
) -> None:
    runner = _runner(tmp_path, db_name)

    reports, _oracle = await runner.run_all()

    report = runner.aging_report
    assert report is not None
    assert (report.passes_run, report.seeded_traces) == (4, 3)
    assert report.consolidation_batch_size == 3
    assert 0.0 < report.evaluation_skew_seconds < 60.0
    assert sum(report.health.traces.values()) == 3
    assert [r.success for r in reports[0].results] == [True]

    conn = await asyncpg.connect(_url(db_name))
    try:
        newest = await conn.fetchval("SELECT max(created_at) FROM memory_traces")
    finally:
        await conn.close()
    assert now_utc() - newest > timedelta(days=2)
