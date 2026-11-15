"""Surfacing run report: fact population, degraded count, failed seeds and queries."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, call

from probe_cli import surfacing_runner
from probe_cli.arm import Arm, CorpusConfig
from probe_cli.provenance import stamp_prompt
from probe_cli.surfacing_runner import (
    FactPopulation,
    PoolFact,
    SurfacingArmRunner,
    SurfacingRunReport,
)
from recollect.exceptions import MemorySDKError, StorageError
from recollect.extraction import PatternExtractor
from recollect.models import SURFACING_STATUSES, MemoryTrace, PersonaFact

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_TRACES = '{"id": "a", "text": "trace a"}\n{"id": "b", "text": "trace b"}\n'
_QUERIES = (
    '{"id": "q1", "text": "about a", "relevant_trace_ids": ["a"]}\n'
    '{"id": "q2", "text": "about b", "relevant_trace_ids": ["b"]}\n'
)


def _fact(status: str, category: str = "general") -> PersonaFact:
    return PersonaFact(
        subject="user",
        predicate="prefers",
        object=status,
        content="c",
        status=status,
        category=category,
    )


def _arm(overrides: dict[str, object] | None = None) -> Arm:
    overrides = overrides or {}
    return Arm(name="s", corpus=CorpusConfig(path=""), recollect_overrides=overrides)


def _memory(before: list[PersonaFact]) -> MagicMock:
    mem = MagicMock()
    mem.connect = AsyncMock()
    mem.close = AsyncMock()
    seeded = MemoryTrace(id="db-a", content="trace a")
    mem.experience = AsyncMock(side_effect=[seeded, MemorySDKError("boom")])
    # A second read would see every candidate promoted.
    after = [f.model_copy(update={"status": "promoted"}) for f in before]
    mem.storage.facts.get_persona_facts = AsyncMock(side_effect=[before, after])
    mem.storage.facts.update_fact_status = AsyncMock(return_value=True)
    mem.storage.traces.count_degraded = AsyncMock(return_value=3)
    mem._config.get.return_value = 10
    embed = AsyncMock(side_effect=[[0.1], StorageError("x")])
    mem._embeddings.generate_embedding = embed
    mem.storage.vectors.search_semantic = AsyncMock(return_value=[])
    mem._activate_recall_tokens = AsyncMock(return_value=({}, {}))
    mem._find_relevant_persona_facts = AsyncMock(return_value=([], {}))
    return mem


async def _run(
    mem: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, object] | None = None,
) -> SurfacingRunReport:
    def _connected(**_: object) -> MagicMock:
        return mem

    monkeypatch.setattr(surfacing_runner, "CognitiveMemory", _connected)
    traces, queries = tmp_path / "t.jsonl", tmp_path / "q.jsonl"
    traces.write_text(_TRACES, encoding="utf-8")
    queries.write_text(_QUERIES, encoding="utf-8")
    runner = SurfacingArmRunner(
        arm_name="s",
        provider=MagicMock(model_name="m"),
        config=_arm(overrides).to_memory_config(),
        traces_path=str(traces),
        queries_path=str(queries),
    )
    return await runner.run()


async def test_population_is_read_once_before_force_promotion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = [_fact("promoted", "health"), _fact("candidate"), _fact("candidate")]
    mem = _memory(before)
    report = await _run(mem, monkeypatch, tmp_path)
    assert report.fact_population == FactPopulation(
        total=3,
        by_status={"promoted": 1, "candidate": 2},
        by_category={"health": 1, "general": 2},
    )
    assert report.promoted_facts == 2
    first_read = mem.storage.facts.get_persona_facts.await_args_list[0]
    assert first_read == call(limit=None, user_id=report.user_id)
    names = [name for name, _, _ in mem.mock_calls]
    first_promotion = names.index("storage.facts.update_fact_status")
    assert names.index("storage.facts.get_persona_facts") < first_promotion


async def test_failures_and_the_degraded_count_are_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mem = _memory([])
    report = await _run(mem, monkeypatch, tmp_path)
    assert (report.failed_seed_ids, report.failed_query_ids) == (["b"], ["q2"])
    assert [(q.query_id, q.required_eval_ids) for q in report.queries] == [
        ("q1", ["a"])
    ]
    assert report.user_id.startswith("probe-s-")
    assert report.degraded_traces == 3
    count = mem.storage.traces.count_degraded
    count.assert_awaited_once_with(user_id=report.user_id)
    names = [name for name, _, _ in mem.mock_calls]
    last_seed = max(i for i, name in enumerate(names) if name == "experience")
    assert names.index("storage.traces.count_degraded") > last_seed
    assert report.swallowed_failures == {
        "probe_cli.surfacing_runner: seed ingest failed for %s": 1,
        "probe_cli.surfacing_runner: score failed for query %s": 1,
    }
    extractor = PatternExtractor(MagicMock(), config=_arm().to_memory_config())
    assert report.prompt == stamp_prompt(extractor)


async def test_the_pool_and_the_fact_limit_are_recorded_after_promotion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sourced = _fact("candidate", "health").model_copy(
        update={"source_trace_id": "db-a"}
    )
    before = [sourced, _fact("promoted")]
    mem = _memory(before)
    overrides: dict[str, object] = {"persona.max_facts_per_query": 7}
    report = await _run(mem, monkeypatch, tmp_path, overrides)
    assert report.surfacing_pool == [
        PoolFact(
            fact_id=sourced.id, source_eval_id="a", category="health", status="promoted"
        ),
        PoolFact(fact_id=before[1].id, category="general", status="promoted"),
    ]
    assert report.max_facts_per_query == 7
    pool_read = mem.storage.facts.get_persona_facts.await_args_list[1]
    assert pool_read == call(
        user_id=report.user_id, statuses=SURFACING_STATUSES, limit=None
    )
    names = [name for name, _, _ in mem.mock_calls]
    reads = [i for i, n in enumerate(names) if n == "storage.facts.get_persona_facts"]
    promotion = names.index("storage.facts.update_fact_status")
    assert reads[1] > promotion


async def test_passes_gate_is_the_production_recall_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kept, dropped = _fact("promoted", "health"), _fact("promoted")
    mem = _memory([])
    scores = {kept.id: 0.4, dropped.id: 0.6}
    mem._find_relevant_persona_facts = AsyncMock(return_value=([kept, dropped], scores))
    mem._apply_recall_floor = MagicMock(return_value=[kept])
    report = await _run(mem, monkeypatch, tmp_path)
    (query,) = report.queries
    assert {f.fact_id: f.passes_gate for f in query.surfaced} == {
        kept.id: True,
        dropped.id: False,
    }
    mem._apply_recall_floor.assert_called_once_with([kept, dropped], scores, {})


_PRE_CHANGE = """{"arm_name": "s", "model": "m", "seeded_traces": 2,
"promoted_facts": 1, "queries": [{"query_id": "q1", "phrasing_style": "",
"is_distractor": false, "surfaced": [{"fact_id": "f1", "source_trace_id": "t1",
"score": 0.81, "relevant": true, "source_trace_activated": false,
"source_trace_propagated_sim": 0.0}]}]}"""


def test_a_pre_change_run_report_still_validates() -> None:
    report = SurfacingRunReport.model_validate_json(_PRE_CHANGE)
    assert (report.user_id, report.degraded_traces) == ("", 0)
    assert report.fact_population == FactPopulation()
    assert (report.failed_seed_ids, report.failed_query_ids) == ([], [])
    assert report.queries[0].required_eval_ids == []
