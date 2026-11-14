"""Recall answers: free-form queries answered from the recall text a consumer reads."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from probe_cli.corpus import load_query_corpus
from probe_cli.fusion_screen import ScreenPreconditionError
from probe_cli.task_runner import ask
from probe_cli.weak_label_answers import current_text
from probe_cli.weak_match_screen import (
    ScreenQuery,
    ScreenValidity,
    _captured,
    _trace_thoughts,
    _WeakPlan,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from recollect.llm.protocol import LLMProvider

    from probe_cli.arm import Arm
    from probe_cli.weak_match_screen import QueryCapture

logger = logging.getLogger(__name__)

REPEATS = 5
# A thinking model spends this budget before its answer: at the task arm's 256,
# claude-haiku-5-5 can exhaust it and return nothing.
ANSWER_MAX_TOKENS = 2048


class QueryAnswers(BaseModel):
    query_id: str
    query: str
    # Trace thoughts in the window; persona-fact thoughts are not counted.
    delivered: int = Field(ge=0)
    recall_text: str
    # Unscored: a reader's verdict is external.
    responses: list[str]


class RecallAnswersReport(BaseModel):
    arm_name: str
    template_db: str
    user_id: str
    queries_path: str
    queries_sha256: str
    answer_model: str
    repeats: int = Field(ge=1)
    validity: ScreenValidity
    queries: list[QueryAnswers]


async def _answer_query(
    capture: QueryCapture, *, provider: LLMProvider
) -> QueryAnswers:
    recall_text = current_text(capture.window)
    delivered = len(_trace_thoughts(capture.window))
    logger.info("recall-answers query=%s delivered=%d", capture.query.id, delivered)
    # One coroutine per ask: gather given one awaitable twice awaits it once.
    responses = await asyncio.gather(
        *(
            ask(
                provider,
                capture.query.query,
                recall_text,
                max_tokens=ANSWER_MAX_TOKENS,
            )
            for _ in range(REPEATS)
        )
    )
    return QueryAnswers(
        query_id=capture.query.id,
        query=capture.query.query,
        delivered=delivered,
        recall_text=recall_text,
        responses=list(responses),
    )


async def answer(
    captures: Sequence[QueryCapture], *, provider: LLMProvider
) -> list[QueryAnswers]:
    """REPEATS answers per captured window; no retrieval runs here."""
    return [await _answer_query(c, provider=provider) for c in captures]


def _query_plan(
    arm: Arm, *, template: str, user_id: str, queries_path: Path
) -> _WeakPlan:
    entries = load_query_corpus(queries_path).entries
    if not entries:
        raise ScreenPreconditionError(f"no queries in {queries_path}")
    config = arm.to_memory_config()
    config._set("recall_tokens.enabled", arm.task.with_tokens)
    return _WeakPlan(
        arm=arm,
        config=config,
        template=template,
        user_id=user_id,
        queries=[ScreenQuery(id=q.id, query=q.text) for q in entries],
        queries_path=queries_path,
    )


async def read(
    arm: Arm,
    *,
    template: str,
    user_id: str,
    queries_path: Path,
    provider: LLMProvider,
) -> RecallAnswersReport:
    """The store copy is dropped before the first answer call."""
    plan = _query_plan(
        arm, template=template, user_id=user_id, queries_path=queries_path
    )
    digest = hashlib.sha256(queries_path.read_bytes()).hexdigest()
    result = await _captured(plan)
    queries = await answer(result.captures, provider=provider)
    return RecallAnswersReport(
        arm_name=arm.name,
        template_db=template,
        user_id=user_id,
        queries_path=str(queries_path),
        queries_sha256=digest,
        answer_model=provider.model_name,
        repeats=REPEATS,
        validity=result.validity,
        queries=queries,
    )
