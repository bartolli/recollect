"""pydantic-ai provider: a model that rejects temperature is served without it."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from recollect.exceptions import ExtractionError
from recollect.llm.pydantic_ai import PydanticAIProvider
from recollect.llm.types import Message

pydantic_ai_exceptions = pytest.importorskip("pydantic_ai.exceptions")

_MESSAGES = [Message(role="user", content="hi")]


class _Out(BaseModel):
    value: str = ""


_Rejection = Callable[[str], Exception]


def _http_400(message: str) -> Exception:
    body = {"type": "error", "error": {"message": message}}
    error: Exception = pydantic_ai_exceptions.ModelHTTPError(400, "m", body)
    return error


def _request_rejected(message: str) -> Exception:
    body = {"type": "error", "error": {"message": message}}
    error: Exception = pydantic_ai_exceptions.ModelRequestRejectedError(
        "m", f"status_code: 400, model_name: m, body: {body}", body=body
    )
    return error


# pydantic-ai reports a refused request as either class, by provider and version.
@pytest.fixture(params=[_http_400, _request_rejected])
def _rejection(request: pytest.FixtureRequest) -> _Rejection:
    rejection: _Rejection = request.param
    return rejection


_TEMPERATURE = "`temperature` is deprecated for this model."


def _agent(*outcomes: object) -> MagicMock:
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=list(outcomes))
    return agent


def _sent(agent: MagicMock) -> list[dict[str, Any]]:
    return [call.kwargs["model_settings"] for call in agent.run.await_args_list]


async def _call(provider: PydanticAIProvider, method: str) -> object:
    if method == "complete":
        return await provider.complete(_MESSAGES)
    return await provider.complete_structured(_MESSAGES, output_type=_Out)


@pytest.fixture()
def provider() -> PydanticAIProvider:
    return PydanticAIProvider(model="anthropic:claude-haiku-5-5")


@pytest.mark.parametrize("method", ["complete", "complete_structured"])
async def test_a_rejected_temperature_is_dropped_and_the_call_retried(
    provider: PydanticAIProvider, method: str, _rejection: _Rejection
) -> None:
    agent = _agent(_rejection(_TEMPERATURE), MagicMock(output="ok"))

    with patch("pydantic_ai.Agent", return_value=agent):
        assert await _call(provider, method) == "ok"

    first, second = _sent(agent)
    assert first["temperature"] == 0.0
    assert "temperature" not in second
    assert second["max_tokens"] == first["max_tokens"]


async def test_the_rejection_is_remembered_for_the_provider(
    provider: PydanticAIProvider, _rejection: _Rejection
) -> None:
    agent = _agent(
        _rejection(_TEMPERATURE), MagicMock(output="ok"), MagicMock(output="again")
    )

    with patch("pydantic_ai.Agent", return_value=agent):
        await provider.complete(_MESSAGES)
        assert await provider.complete(_MESSAGES, temperature=0.0) == "again"

    # The later call spends no request rediscovering the rejection.
    assert ["temperature" in sent for sent in _sent(agent)] == [True, False, False]


async def test_concurrent_calls_each_retry_their_own_rejection(
    provider: PydanticAIProvider,
    _rejection: _Rejection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The yield puts every request in flight, temperature sent, before the
    # first rejection is handled.
    async def _run(_: str, *, model_settings: dict[str, Any]) -> MagicMock:
        await asyncio.sleep(0)
        if "temperature" in model_settings:
            raise _rejection(_TEMPERATURE)
        return MagicMock(output="ok")

    agent = MagicMock()
    agent.run = AsyncMock(side_effect=_run)

    with patch("pydantic_ai.Agent", return_value=agent):
        calls = (provider.complete(_MESSAGES) for _ in range(3))
        assert await asyncio.gather(*calls) == ["ok"] * 3

    carried = sorted("temperature" in sent for sent in _sent(agent))
    assert carried == [False] * 3 + [True] * 3
    logged = [r for r in caplog.records if "rejects temperature" in r.message]
    assert len(logged) == 1


async def test_a_model_that_accepts_temperature_keeps_it(
    provider: PydanticAIProvider,
) -> None:
    agent = _agent(MagicMock(output="ok"), MagicMock(output="ok"))

    with patch("pydantic_ai.Agent", return_value=agent):
        await provider.complete(_MESSAGES)
        await provider.complete(_MESSAGES, temperature=0.3)

    assert [sent["temperature"] for sent in _sent(agent)] == [0.0, 0.3]


async def test_another_bad_request_is_not_retried(
    provider: PydanticAIProvider, _rejection: _Rejection
) -> None:
    agent = _agent(_rejection("max_tokens: 99999 > 64000"), MagicMock(output="ok"))

    with (
        patch("pydantic_ai.Agent", return_value=agent),
        pytest.raises(ExtractionError, match="pydantic-ai API error"),
    ):
        await provider.complete(_MESSAGES)

    assert agent.run.await_count == 1


async def test_a_rejection_after_the_drop_raises(
    provider: PydanticAIProvider, _rejection: _Rejection
) -> None:
    agent = _agent(_rejection(_TEMPERATURE), _rejection(_TEMPERATURE))

    with (
        patch("pydantic_ai.Agent", return_value=agent),
        pytest.raises(ExtractionError, match="pydantic-ai API error"),
    ):
        await provider.complete(_MESSAGES)

    assert agent.run.await_count == 2
