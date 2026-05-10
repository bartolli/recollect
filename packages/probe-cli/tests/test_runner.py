"""ArmRunner + RetrievalArmRunner tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
from probe_cli.arm import load_arm
from probe_cli.corpus import Corpus, CorpusEntry
from probe_cli.metrics import FOREIGN_PREFIX, aggregate_runs
from probe_cli.runner import ArmRunner, RetrievalArmRunner, resolve_thoughts
from pydantic import BaseModel
from recollect.exceptions import ExtractionError
from recollect.llm.types import Entity, ExtractionResult, Message, Relation
from recollect.models import MemoryTrace, Thought

T = TypeVar("T", bound=BaseModel)


class MockProvider:
    """Deterministic provider that returns canned ExtractionResult per text key."""

    model_name = "mock"

    def __init__(self, responder: Callable[[str], ExtractionResult]) -> None:
        self._responder = responder

    async def complete(self, messages: list[Message], **_: Any) -> str:
        return ""

    async def complete_structured(
        self,
        messages: list[Message],
        output_type: type[T],
        **_: Any,
    ) -> T:
        text = next((m.content for m in messages if m.role == "user"), "")
        result = self._responder(text)
        if not isinstance(result, output_type):
            raise ExtractionError(f"mock returned {type(result).__name__}")
        return result


def _arm_file(tmp_path: Path, name: str = "test", runs: int = 2) -> Path:
    p = tmp_path / "arm.toml"
    p.write_text(
        f'[arm]\nname = "{name}"\nruns = {runs}\n[corpus]\npath = "x.jsonl"\n',
        encoding="utf-8",
    )
    return p


def _corpus(entries: int) -> Corpus:
    return Corpus(
        entries=[
            CorpusEntry(id=f"e{i}", text=f"sample {i}") for i in range(entries)
        ]
    )


def _ok_result(text: str) -> ExtractionResult:
    return ExtractionResult(
        concepts=["a", "b"],
        relations=[
            Relation(
                source="x",
                relation="works_at",
                target="y",
                category="identity",
            )
        ],
        entities=[Entity(name="x", entity_type="person")],
    )


@pytest.mark.asyncio
async def test_run_once_collects_all_entries(tmp_path: Path) -> None:
    arm = load_arm(_arm_file(tmp_path))
    runner = ArmRunner(arm, MockProvider(_ok_result))
    report = await runner.run_once(_corpus(3), run_index=0)
    assert len(report.entries) == 3
    assert all(e.success for e in report.entries)
    assert report.prompt_version  # template version surfaced


@pytest.mark.asyncio
async def test_run_all_executes_n_times(tmp_path: Path) -> None:
    arm = load_arm(_arm_file(tmp_path, runs=4))
    runner = ArmRunner(arm, MockProvider(_ok_result))
    reports = await runner.run_all(_corpus(2))
    assert len(reports) == 4
    assert [r.run_index for r in reports] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_extraction_error_recorded_not_raised(tmp_path: Path) -> None:
    arm = load_arm(_arm_file(tmp_path, runs=1))

    def fail(text: str) -> ExtractionResult:
        raise ExtractionError(f"boom: {text}")

    runner = ArmRunner(arm, MockProvider(fail))
    reports = await runner.run_all(_corpus(2))
    assert len(reports) == 1
    assert all(not e.success for e in reports[0].entries)
    assert "boom" in reports[0].entries[0].error


@pytest.mark.asyncio
async def test_aggregate_after_run_all(tmp_path: Path) -> None:
    arm = load_arm(_arm_file(tmp_path, runs=3))
    runner = ArmRunner(arm, MockProvider(_ok_result))
    reports = await runner.run_all(_corpus(5))
    agg = aggregate_runs(arm.name, runner.model, runner.template_version, reports)
    assert agg.runs == 3
    assert agg.total_entries_per_run == 5
    assert agg.validity_rate_mean == 1.0
    assert agg.predicate_distribution["works_at"] == 15


def _retrieval_arm_file(
    tmp_path: Path,
    *,
    enabled: bool = True,
    traces: str = "t.jsonl",
    queries: str = "q.jsonl",
) -> Path:
    p = tmp_path / "arm.toml"
    body = (
        '[arm]\nname = "r"\n'
        '[corpus]\npath = "x.jsonl"\n'
        "[retrieval]\n"
        f"enabled = {'true' if enabled else 'false'}\n"
        f'traces_corpus_path = "{traces}"\n'
        f'query_corpus_path = "{queries}"\n'
    )
    p.write_text(body, encoding="utf-8")
    return p


def test_retrieval_runner_rejects_disabled_arm(tmp_path: Path) -> None:
    arm = load_arm(_retrieval_arm_file(tmp_path, enabled=False))
    with pytest.raises(ValueError, match=r"retrieval\.enabled"):
        RetrievalArmRunner(arm, MockProvider(_ok_result))


def test_retrieval_runner_rejects_missing_traces_path(tmp_path: Path) -> None:
    arm = load_arm(_retrieval_arm_file(tmp_path, traces=""))
    with pytest.raises(ValueError, match="traces_corpus_path"):
        RetrievalArmRunner(arm, MockProvider(_ok_result))


def test_retrieval_runner_rejects_missing_query_path(tmp_path: Path) -> None:
    arm = load_arm(_retrieval_arm_file(tmp_path, queries=""))
    with pytest.raises(ValueError, match="query_corpus_path"):
        RetrievalArmRunner(arm, MockProvider(_ok_result))


def _thought(trace_id: str, *, persona_fact: bool = False) -> Thought:
    pattern = {"persona_fact": True} if persona_fact else {}
    return Thought(
        trace=MemoryTrace(id=trace_id, content="x", pattern=pattern),
        relevance=0.5,
    )


def test_resolve_thoughts_maps_corpus_ids() -> None:
    thoughts = [_thought("uuid-a"), _thought("uuid-b")]
    inverse = {"uuid-a": "h1", "uuid-b": "d1"}
    ranked, pf = resolve_thoughts(thoughts, inverse)
    assert ranked == ["h1", "d1"]
    assert pf == 0


def test_resolve_thoughts_marks_unknown_as_foreign() -> None:
    thoughts = [_thought("uuid-a"), _thought("uuid-stray")]
    inverse = {"uuid-a": "h1"}
    ranked, pf = resolve_thoughts(thoughts, inverse)
    assert ranked == ["h1", f"{FOREIGN_PREFIX}uuid-stray"]
    assert pf == 0


def test_resolve_thoughts_filters_persona_facts() -> None:
    thoughts = [
        _thought("uuid-a"),
        _thought("uuid-pf-1", persona_fact=True),
        _thought("uuid-b"),
        _thought("uuid-pf-2", persona_fact=True),
    ]
    inverse = {"uuid-a": "h1", "uuid-b": "d1"}
    ranked, pf = resolve_thoughts(thoughts, inverse)
    assert ranked == ["h1", "d1"]
    assert pf == 2
