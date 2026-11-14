"""Weak-label answers: one captured window answered from two recall texts."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from recollect_mcp.server import _format_thoughts

from probe_cli.corpus import AnswerType, TierLabel
from probe_cli.fusion_screen import ScreenPreconditionError
from probe_cli.task_runner import ask
from probe_cli.task_scoring import score_answer
from probe_cli.weak_match_screen import (
    RuleName,
    ScreenValidity,
    _captured,
    _plan,
    cut_window,
    weak_traces,
    window_lines,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from recollect.llm.protocol import LLMProvider
    from recollect.models import Thought

    from probe_cli.arm import Arm
    from probe_cli.corpus import TaskQuestion
    from probe_cli.weak_match_screen import QueryCapture

logger = logging.getLogger(__name__)

RULE: RuleName = "top-relative"
DELTA = 0.20
REPEATS = 5
LOWER_RANKED_HEADER = "Lower-ranked matches (scored well below the top match):"
_SECTION = "Related memories:"


class RenderAnswers(BaseModel):
    correct: int = Field(ge=0)
    responses: list[str]


class QuestionAnswers(BaseModel):
    question_id: str
    tier_label: TierLabel
    answer_type: AnswerType
    window: list[str]
    weak: int = Field(ge=0)
    # Seed keys of required traces printed under the lower-ranked header.
    required_weak: list[str]
    identical: bool
    current: RenderAnswers
    labelled: RenderAnswers


class WeakLabelAnswersReport(BaseModel):
    arm_name: str
    template_db: str
    user_id: str
    density_tier: int
    answer_model: str
    rule: RuleName
    parameter: float
    header: str
    repeats: int
    validity: ScreenValidity
    # Largest correct-count gap on byte-identical renders: sampling noise alone.
    noise_floor: int
    questions: list[QuestionAnswers]


def current_text(window: list[Thought]) -> str:
    return _format_thoughts(window, safety_facts=())


def labelled_text(window: list[Thought], weak: frozenset[str]) -> str:
    head = _format_thoughts(cut_window(window, weak), safety_facts=())
    tail = [t for t in window if t.trace.id in weak]
    if not tail:
        return head
    first, *rest = _format_thoughts(tail, safety_facts=()).split("\n")
    # The header replaces the tail's own section line; after formatter drift the
    # first line could be a memory item, and replacing it would drop that item.
    if first != _SECTION:
        raise ScreenPreconditionError(f"tail render does not open with {_SECTION!r}")
    return "\n".join([head, LOWER_RANKED_HEADER, *rest])


def _scored(responses: Sequence[str], question: TaskQuestion) -> RenderAnswers:
    return RenderAnswers(
        correct=sum(score_answer(r, question) for r in responses),
        responses=list(responses),
    )


async def _answer_question(
    capture: QueryCapture,
    *,
    question: TaskQuestion,
    keys: dict[str, str],
    provider: LLMProvider,
) -> QuestionAnswers:
    weak = weak_traces(capture, rule=RULE, parameter=DELTA)
    current = current_text(capture.window)
    labelled = labelled_text(capture.window, weak)
    logger.info("weak-label-answers question=%s weak=%d", question.id, len(weak))
    # One coroutine per ask: gather given one awaitable twice awaits it once.
    contexts = (current,) * REPEATS + (labelled,) * REPEATS
    responses = await asyncio.gather(
        *(ask(provider, question.question, context) for context in contexts)
    )
    return QuestionAnswers(
        question_id=question.id,
        tier_label=question.tier_label,
        answer_type=question.answer_type,
        window=window_lines(capture.window, keys=keys, weak=weak),
        weak=len(weak),
        required_weak=sorted(keys.get(t, t[:8]) for t in capture.required & weak),
        identical=current == labelled,
        current=_scored(responses[:REPEATS], question),
        labelled=_scored(responses[REPEATS:], question),
    )


async def answer(
    captures: Sequence[QueryCapture],
    *,
    questions: Mapping[str, TaskQuestion],
    keys: dict[str, str],
    provider: LLMProvider,
) -> list[QuestionAnswers]:
    """Both renders of every captured window; no retrieval runs here."""
    return [
        await _answer_question(
            c, question=questions[c.query.id], keys=keys, provider=provider
        )
        for c in captures
    ]


def noise_floor(questions: Sequence[QuestionAnswers]) -> int:
    gaps = (
        abs(q.current.correct - q.labelled.correct) for q in questions if q.identical
    )
    return max(gaps, default=0)


async def compare(
    arm: Arm, *, template: str, user_id: str, provider: LLMProvider
) -> WeakLabelAnswersReport:
    """The store copy is dropped before the first answer call."""
    plan = _plan(arm, template=template, user_id=user_id, queries_path=None)
    result = await _captured(plan)
    questions = await answer(
        result.captures, questions=plan.questions, keys=result.keys, provider=provider
    )
    return WeakLabelAnswersReport(
        arm_name=arm.name,
        template_db=template,
        user_id=user_id,
        density_tier=arm.task.density_tier,
        answer_model=provider.model_name,
        rule=RULE,
        parameter=DELTA,
        header=LOWER_RANKED_HEADER,
        repeats=REPEATS,
        validity=result.validity,
        noise_floor=noise_floor(questions),
        questions=questions,
    )
