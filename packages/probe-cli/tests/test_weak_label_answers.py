"""Weak-label answers: labelled render, paired answering over one capture."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import pytest
from probe_cli import weak_label_answers
from probe_cli.corpus import TaskQuestion
from probe_cli.fusion_screen import ScreenPreconditionError
from probe_cli.weak_label_answers import (
    DELTA,
    LOWER_RANKED_HEADER,
    REPEATS,
    RULE,
    QuestionAnswers,
    RenderAnswers,
    answer,
    current_text,
    labelled_text,
    noise_floor,
)
from probe_cli.weak_match_screen import QueryCapture, ScreenQuery, weak_traces
from recollect.core import CognitiveMemory
from recollect.models import MemoryTrace, Thought

if TYPE_CHECKING:
    from probe_cli.corpus import AnswerType, TierLabel
    from recollect.llm.types import Message


def _trace(tid: str, relevance: float) -> Thought:
    text = f"memory {tid}"
    trace = MemoryTrace(id=tid, content=text)
    return Thought(trace=trace, relevance=relevance, reconstruction=text)


def _fact(tid: str, relevance: float) -> Thought:
    trace = MemoryTrace(id=tid, pattern={"persona_fact": True})
    return Thought(trace=trace, relevance=relevance, reconstruction=f"fact {tid}")


def _capture(
    window: list[Thought], *, qid: str = "q1", required: str = ""
) -> QueryCapture:
    return QueryCapture(
        query=ScreenQuery(id=qid, query=f"question {qid}"),
        window=window,
        token_bonuses={},
        similarities={t.trace.id: 0.5 for t in window},
        required=frozenset({required} if required else ()),
    )


def _question(
    qid: str, *, tier: TierLabel = "t3", answer_type: AnswerType = "name"
) -> TaskQuestion:
    return TaskQuestion(
        id=qid,
        question=f"question {qid}",
        answers=["unknown"] if answer_type == "abstain" else ["Lisbon"],
        answer_type=answer_type,
        tier_label=tier,
    )


class _HeaderAnswerer:
    model_name = "fake"

    def __init__(self) -> None:
        self.contents: list[str] = []

    async def complete(self, messages: list[Message], **_: Any) -> str:
        content = messages[-1].content
        self.contents.append(content)
        return "Lisbon" if LOWER_RANKED_HEADER in content else "unknown"


async def _answer(
    window: list[Thought],
    questions: list[TaskQuestion],
    *,
    required: str = "",
    keys: dict[str, str] | None = None,
) -> tuple[list[QuestionAnswers], list[str]]:
    provider = _HeaderAnswerer()
    captures = [_capture(window, qid=q.id, required=required) for q in questions]
    qs = {q.id: q for q in questions}
    results = await answer(captures, questions=qs, keys=keys or {}, provider=provider)
    return results, provider.contents


def _items(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("- ")]


def _texts(lines: list[str]) -> list[str]:
    return [line.split("] ")[-1] for line in _items(lines)]


def test_labelled_render_moves_lines_and_drops_none() -> None:
    window = [_fact("f", 0.95), _trace("a", 0.80), _trace("w1", 0.55)]
    window += [_trace("b", 0.70), _trace("w2", 0.50)]
    weak = weak_traces(_capture(window), rule=RULE, parameter=DELTA)
    assert weak == {"w1", "w2"}
    current = current_text(window).split("\n")
    lines = labelled_text(window, weak).split("\n")
    assert Counter(_items(lines)) == Counter(_items(current))
    assert len(set(_items(current))) == len(window)
    split = lines.index(LOWER_RANKED_HEADER)
    head, tail = lines[:split], lines[split + 1 :]
    assert head[:2] == ["IMPORTANT CONTEXT:", _items(current)[0]]
    assert "Related memories:" in head
    assert _texts(head) == ["fact f", "memory a", "memory b"]
    assert _texts(tail) == ["memory w1", "memory w2"] and tail == _items(tail)


def test_formatter_drift_refuses_the_labelled_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _drifted(thoughts: list[Thought], *, safety_facts: object) -> str:
        return "Memories:\n" + "\n".join(f"- {t.reconstruction}" for t in thoughts)

    monkeypatch.setattr(weak_label_answers, "_format_thoughts", _drifted)
    window = [_trace("a", 0.80), _trace("w", 0.50)]
    with pytest.raises(ScreenPreconditionError, match="Related memories:"):
        labelled_text(window, frozenset({"w"}))


def _fail(*_: object, **__: object) -> None:
    raise AssertionError("retrieval ran while answering")


async def test_both_renders_answer_from_one_captured_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CognitiveMemory, "think_about", _fail)
    questions = [_question("t3q01"), _question("t3q02")]
    window = [_trace("a", 0.80), _trace("w", 0.50)]
    results, contents = await _answer(window, questions)
    texts = (current_text(window), labelled_text(window, frozenset({"w"})))
    for n, (question, result) in enumerate(zip(questions, results, strict=True)):
        asked = contents[n * 2 * REPEATS : (n + 1) * 2 * REPEATS]
        prompts = [f"{text}\n\nQuestion: {question.question}" for text in texts]
        assert Counter(asked) == {prompts[0]: REPEATS, prompts[1]: REPEATS}
        assert sum(LOWER_RANKED_HEADER in content for content in asked) == REPEATS
        assert (result.weak, result.identical) == (1, False)
        assert result.current.responses == ["unknown"] * REPEATS
        assert (result.current.correct, result.labelled.correct) == (0, REPEATS)


async def test_none_tier_counts_abstention_as_correct() -> None:
    question = _question("abq01", tier="none", answer_type="abstain")
    window = [_trace("a", 0.80), _trace("w", 0.50)]
    (result,), _ = await _answer(window, [question])
    assert result.labelled.responses == ["Lisbon"] * REPEATS
    assert (result.current.correct, result.labelled.correct) == (REPEATS, 0)


async def test_required_traces_under_the_header_are_named() -> None:
    window = [_trace("a", 0.80), _trace("w", 0.50)]
    keys = {"a": "s-a", "w": "s-w"}
    (weak,), _ = await _answer(window, [_question("q1")], required="w", keys=keys)
    (strong,), _ = await _answer(window, [_question("q2")], required="a", keys=keys)
    assert weak.required_weak == ["s-w"]
    assert weak.window == ["1. s-a 0.800 strong", "2. s-w 0.500 weak"]
    assert strong.required_weak == []


async def test_no_trace_under_the_delta_renders_identically() -> None:
    window = [_fact("f", 0.95), _trace("a", 0.80), _trace("b", 0.61)]
    assert labelled_text(window, frozenset()) == current_text(window)
    (result,), contents = await _answer(window, [_question("q1")])
    assert (result.weak, result.identical) == (0, True)
    assert len(set(contents)) == 1 and len(contents) == 2 * REPEATS


def _counted(identical: bool, *, current: int, labelled: int) -> QuestionAnswers:
    return QuestionAnswers(
        question_id="q",
        tier_label="t2",
        answer_type="name",
        window=[],
        weak=int(not identical),
        required_weak=[],
        identical=identical,
        current=RenderAnswers(correct=current, responses=[]),
        labelled=RenderAnswers(correct=labelled, responses=[]),
    )


def test_noise_floor_reads_identical_questions_only() -> None:
    gap_1 = _counted(True, current=3, labelled=4)
    gap_2 = _counted(True, current=4, labelled=2)
    moved = _counted(False, current=0, labelled=5)
    assert noise_floor([gap_1, gap_2, moved]) == 2
    assert noise_floor([moved]) == 0
