"""Recall answers: free-form queries answered from the MCP recall render."""

from __future__ import annotations

import hashlib
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from probe_cli import __main__ as cli
from probe_cli import recall_answers
from probe_cli.arm import Arm, CorpusConfig, TaskConfig, load_arm
from probe_cli.fusion_screen import ScreenPreconditionError
from probe_cli.recall_answers import ANSWER_MAX_TOKENS, REPEATS, answer, read
from probe_cli.weak_label_answers import current_text
from probe_cli.weak_match_screen import QueryCapture, ScreenQuery, ScreenValidity, _Pass
from recollect.models import MemoryTrace, Thought

if TYPE_CHECKING:
    from probe_cli.weak_match_screen import _WeakPlan
    from recollect.llm.types import Message

_FIX = Path(__file__).parent.parent / "fixtures" / "qualification"


def _trace(tid: str, relevance: float) -> Thought:
    text = f"memory {tid}"
    trace = MemoryTrace(id=tid, content=text)
    return Thought(trace=trace, relevance=relevance, reconstruction=text)


def _fact(tid: str, relevance: float) -> Thought:
    trace = MemoryTrace(id=tid, pattern={"persona_fact": True})
    return Thought(trace=trace, relevance=relevance, reconstruction=f"fact {tid}")


def _query(qid: str) -> ScreenQuery:
    return ScreenQuery(id=qid, query=f"question {qid}")


def _capture(window: list[Thought], query: ScreenQuery | None = None) -> QueryCapture:
    return QueryCapture(
        query=query or _query("q1"),
        window=window,
        token_bonuses={},
        similarities={t.trace.id: 0.5 for t in window},
    )


class _EchoAnswerer:
    model_name = "fake"

    def __init__(self, events: list[str] | None = None) -> None:
        self.contents: list[str] = []
        self.budgets: list[int] = []
        self.events = [] if events is None else events

    async def complete(self, messages: list[Message], **kwargs: Any) -> str:
        content = messages[-1].content
        self.contents.append(content)
        self.budgets.append(kwargs["max_tokens"])
        self.events.append("ask")
        return content.rsplit("\n", 1)[-1]


def _prompt(window: list[Thought], qid: str) -> str:
    return f"{current_text(window)}\n\nQuestion: question {qid}"


async def test_each_query_answers_repeats_times_from_its_recall_render() -> None:
    windows = {"q1": [_trace("a", 0.80)], "q2": [_trace("b", 0.70), _trace("c", 0.60)]}
    provider = _EchoAnswerer()
    captures = [_capture(w, _query(qid)) for qid, w in windows.items()]
    results = await answer(captures, provider=provider)
    for n, (qid, window) in enumerate(windows.items()):
        asked = provider.contents[n * REPEATS : (n + 1) * REPEATS]
        assert asked == [_prompt(window, qid)] * REPEATS
        result = results[n]
        assert (result.query_id, result.query) == (qid, f"question {qid}")
        assert result.recall_text == current_text(window)
        assert result.responses == [f"Question: question {qid}"] * REPEATS
    assert provider.budgets == [ANSWER_MAX_TOKENS] * (2 * REPEATS)


async def test_delivered_counts_trace_thoughts_not_persona_facts() -> None:
    window = [_fact("f", 0.95), _trace("a", 0.80), _trace("b", 0.60)]
    (result,) = await answer([_capture(window)], provider=_EchoAnswerer())
    assert result.delivered == 2
    assert "fact f" in result.recall_text


async def test_empty_window_answers_from_the_empty_render() -> None:
    provider = _EchoAnswerer()
    (result,) = await answer([_capture([])], provider=provider)
    assert (result.delivered, result.recall_text) == (0, current_text([]))
    assert provider.contents == [_prompt([], "q1")] * REPEATS


_QUERIES = (
    "# retrieval query-corpus format\n"
    '{"id": "q-a", "text": "Who said it?", "relevant_trace_ids": ["a"]}\n'
    '{"id": "q-b", "text": "When is it?", "relevant_trace_ids": []}\n'
)


def _arm() -> Arm:
    task = TaskConfig(enabled=True, answer_model="fake", with_tokens=False)
    return Arm(name="answers", corpus=CorpusConfig(path=""), task=task)


def _refuse(*_: object, **__: object) -> None:
    raise AssertionError("reached the store")


async def test_read_maps_the_query_corpus_and_answers_after_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(_QUERIES, encoding="utf-8")
    events: list[str] = []
    plans: list[_WeakPlan] = []

    async def _captured(plan: _WeakPlan) -> _Pass:
        plans.append(plan)
        events.append("captured")
        captures = [_capture([_trace("a", 0.80)], q) for q in plan.queries]
        validity = ScreenValidity(matched=2, checks=2, archived_candidates=0)
        return _Pass(captures, validity, {})

    monkeypatch.setattr(recall_answers, "_captured", _captured)
    provider = _EchoAnswerer(events)
    report = await read(
        _arm(), template="keep", user_id="u", queries_path=path, provider=provider
    )
    (plan,) = plans
    texts = [("q-a", "Who said it?"), ("q-b", "When is it?")]
    assert [(q.id, q.query) for q in plan.queries] == texts
    assert (plan.template, plan.user_id) == ("keep", "u")
    assert (plan.questions, plan.seeds) == ({}, [])
    assert plan.config.get("recall_tokens.enabled") is False
    assert events == ["captured"] + ["ask"] * (2 * REPEATS)
    assert [(q.query_id, q.query) for q in report.queries] == texts
    assert report.queries_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    head = (report.queries_path, report.template_db, report.user_id)
    assert head == (str(path), "keep", "u")
    run = (report.answer_model, report.repeats, report.validity.checks)
    assert run == ("fake", REPEATS, 2)


async def test_read_refuses_an_empty_queries_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(recall_answers, "_captured", _refuse)
    path = tmp_path / "empty.jsonl"
    path.write_text("# no queries\n", encoding="utf-8")
    provider = _EchoAnswerer()
    with pytest.raises(ScreenPreconditionError, match="no queries"):
        await read(
            _arm(), template="keep", user_id="u", queries_path=path, provider=provider
        )
    assert provider.contents == []


def test_answers_arm_is_a_haiku55_task_arm_without_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECOLLECT_PROBE_DB_URL", "postgresql://probe/probe_eval")
    path = _FIX / "answers.toml"
    arm = load_arm(path)
    assert (arm.name, arm.runs) == ("qualification-answers-haiku55", 1)
    assert arm.task.enabled
    haiku = "anthropic:claude-haiku-5-5"
    assert arm.extraction.pydantic_ai_model == arm.task.answer_model == haiku
    assert arm.task.db_url == "postgresql://probe/probe_eval"
    assert arm.recollect_overrides == {}
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    assert raw["task"]["db_url"] == "${RECOLLECT_PROBE_DB_URL}"


def test_cli_refuses_an_arm_without_an_answer_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "PydanticAIProvider", _refuse)
    monkeypatch.setattr(cli, "read_recall_answers", _refuse)
    monkeypatch.setattr(recall_answers, "_captured", _refuse)
    arm = tmp_path / "a.toml"
    arm.write_text(
        '[arm]\nname = "a"\n[corpus]\npath = ""\n[task]\nenabled = true\n',
        encoding="utf-8",
    )
    out = tmp_path / "out"
    argv = ["probe", "recall-answers", "--arm", str(arm), "--template-db", "keep"]
    argv += ["--queries", str(tmp_path / "q.jsonl"), "--out-dir", str(out)]
    argv += ["--env-file", str(tmp_path / "absent.env")]
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 2
    assert not out.exists()
