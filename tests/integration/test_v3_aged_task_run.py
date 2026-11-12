"""Integration: aged task runs on a real store, LLM scripted at the provider.

Seeding, stepping, and evaluation each open their own CognitiveMemory
against one scratch store; only the two model calls are faked. A sweep
restores the seeded store from its snapshot before every point.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from probe_cli.aging import Aging, SweepPoint
from probe_cli.task_runner import TaskArmRunner
from recollect.config import MemoryConfig
from recollect.datetime_utils import now_utc
from recollect.llm.types import ExtractionResult, TokenAssessment

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

DB_URL = os.environ.get("DATABASE_URL", "")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
    pytest.mark.skipif(not DB_URL, reason="DATABASE_URL unset"),
]


async def _snapshot_count(db_name: str) -> int:
    return int(
        await _fetchval(
            "postgres",
            "SELECT count(*) FROM pg_database WHERE datname = $1",
            f"{db_name}_snapshot",
        )
    )


_EXTRACTION = ExtractionResult(concepts=["logistics"], significance=0.7)


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
        await _admin(f'DROP DATABASE IF EXISTS "{name}_snapshot" WITH (FORCE)')
        await _admin(f'DROP DATABASE "{name}" WITH (FORCE)')


async def _fetchval(db_name: str, sql: str, *args: object) -> Any:
    conn = await asyncpg.connect(_url(db_name))
    try:
        return await conn.fetchval(sql, *args)
    finally:
        await conn.close()


def _config(overrides: dict[str, object] | None = None) -> MemoryConfig:
    config = MemoryConfig()
    config._set("recall_tokens.enabled", False)
    config._set("consolidation.batch_size", 1)
    for key, value in (overrides or {}).items():
        config._set(key, value)
    return config


def _runner(
    tmp_path: Path,
    db_name: str,
    aging: Aging | None = None,
    *,
    config: MemoryConfig | None = None,
    structured: AsyncMock | None = None,
    complete: AsyncMock | None = None,
) -> TaskArmRunner:
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
    provider = MagicMock(model_name="scripted")
    provider.complete_structured = structured or AsyncMock(return_value=_EXTRACTION)
    answer = MagicMock(model_name="scripted")
    answer.complete = complete or AsyncMock(return_value="unknown")
    return TaskArmRunner(
        arm_name="aged",
        provider=provider,
        answer_provider=answer,
        config=config or _config(),
        seed_traces_path=str(seeds),
        questions_path=str(questions),
        db_url=_url(db_name),
        aging=aging,
    )


async def test_aged_task_run_steps_and_evaluates_a_real_store(
    tmp_path: Path, db_name: str
) -> None:
    runner = _runner(tmp_path, db_name, Aging(age_days=2, passes_per_day=2))

    reports, _oracle = await runner.run_all()

    report = runner.aging_report
    assert report is not None
    assert (report.passes_run, report.seeded_traces) == (4, 3)
    assert report.consolidation_batch_size == 3
    assert 0.0 < report.evaluation_skew_seconds < 60.0
    assert sum(report.health.traces.values()) == 3
    assert [r.success for r in reports[0].results] == [True]

    newest = await _fetchval(db_name, "SELECT max(created_at) FROM memory_traces")
    assert now_utc() - newest > timedelta(days=2)


async def test_each_point_starts_from_the_snapshot(
    tmp_path: Path, db_name: str
) -> None:
    runner = _runner(tmp_path, db_name)
    points = [
        SweepPoint(aging=Aging(age_days=2, passes_per_day=2)),
        SweepPoint(aging=Aging(age_days=0)),
    ]

    aged, unaged = [result async for result in runner.run_sweep(points)]

    assert (aged.arm_name, unaged.arm_name) == ("aged-age2-p2", "aged-age0")
    assert (aged.aging.passes_run, unaged.aging.passes_run) == (4, 0)
    assert aged.aging.seeding_user_id == unaged.aging.seeding_user_id
    assert [r.success for r in unaged.reports[0].results] == [True]
    assert aged.aging.health.traces == {"archived": 3}
    assert unaged.aging.health.traces == {"active": 3}
    # The last point ran at age 0 after a two-day point: the store it
    # leaves is as old as the seeding, not two days older.
    newest = await _fetchval(db_name, "SELECT max(created_at) FROM memory_traces")
    assert now_utc() - newest < timedelta(hours=1)
    assert not await _snapshot_count(db_name)


async def test_point_refuses_a_foreign_session(tmp_path: Path, db_name: str) -> None:
    foreign: list[asyncpg.Connection] = []

    async def answer_and_stay_connected(*_args: object, **_kwargs: object) -> str:
        if not foreign:
            foreign.append(await asyncpg.connect(_url(db_name)))
        return "unknown"

    runner = _runner(
        tmp_path, db_name, complete=AsyncMock(side_effect=answer_and_stay_connected)
    )
    points = [
        SweepPoint(aging=Aging(age_days=2, passes_per_day=2)),
        SweepPoint(aging=Aging(age_days=0)),
    ]
    try:
        with pytest.raises(ValueError, match="foreign session"):
            _ = [result async for result in runner.run_sweep(points)]

        # Refused before the restore: the store is as the first point left it.
        newest = await foreign[0].fetchval("SELECT max(created_at) FROM memory_traces")
        assert now_utc() - newest > timedelta(days=2)
        assert not await _snapshot_count(db_name)
    finally:
        for conn in foreign:
            await conn.close()


def _pair_member(text: str) -> str | None:
    return next((key for key in ("garden gate", "Rehearsal") if key in text), None)


async def _group_gate_key_with_rehearsal(
    messages: list[Any], output_type: object = None, **_: object
) -> object:
    """Scripted writes: the gate-key and rehearsal traces form one group."""
    if output_type is not TokenAssessment:
        return _EXTRACTION
    prompt = messages[-1].content
    new = re.search(r'^New memory: "(.+)"$', prompt, re.M)
    assert new is not None
    for index, text in re.findall(r"^(\d+)\. \[lone\] (.+)$", prompt, re.M):
        members = {_pair_member(new.group(1)), _pair_member(text)}
        if members == {"garden gate", "Rehearsal"}:
            return TokenAssessment(
                action="create",
                person_ref="choir",
                situation="hall booking",
                implication="gate key needed",
                significance=0.9,
                linked_indices=[int(index)],
            )
    return TokenAssessment()


async def test_tokens_off_control_reads_the_snapshot_with_the_hop_disabled(
    tmp_path: Path, db_name: str
) -> None:
    reinforced: list[bool] = []

    async def answer_and_observe(*_args: object, **_kwargs: object) -> str:
        # A read through the hop stamps the group's recall token.
        reinforced.append(
            await _fetchval(
                db_name,
                "SELECT bool_or(last_activated_at - created_at"
                " > interval '10 milliseconds') FROM recall_tokens",
            )
        )
        return "unknown"

    config = _config(
        {
            "recall_tokens.enabled": True,
            "recall_tokens.write_time_threshold": 0.0,
            # One direct result (the rehearsal trace): the grouped gate-key
            # trace is reachable through the hop alone.
            "retrieval.search_limit": 1,
        }
    )
    runner = _runner(
        tmp_path,
        db_name,
        config=config,
        structured=AsyncMock(side_effect=_group_gate_key_with_rehearsal),
        complete=AsyncMock(side_effect=answer_and_observe),
    )
    points = [
        SweepPoint(aging=Aging(age_days=0)),
        SweepPoint(aging=Aging(age_days=0), tokens_off=True),
    ]

    unaged, control = [result async for result in runner.run_sweep(points)]

    assert control.arm_name == "aged-age0-tokens-off"
    # Same groups: the control reads the seeding the age-0 point read.
    assert unaged.aging.health.tokens_propagating == 1
    assert control.aging.health.tokens_propagating == 1
    assert unaged.aging.recall_tokens_at_evaluation is True
    assert control.aging.recall_tokens_at_evaluation is False
    # Two answer calls per question. The control runs second and finds the
    # token unstamped: the age-0 point's read was restored away, and the
    # control's own read did not take the hop.
    assert reinforced == [True, True, False, False]
