"""pydantic-ai provider adapter.

Routes LLM calls through pydantic-ai's Agent abstraction, giving access
to 20+ model providers through a single dependency. Model strings use
pydantic-ai format: "openai:gpt-4o", "anthropic:claude-haiku-4-5-20251001", etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar, cast

from recollect.exceptions import ExtractionError
from recollect.llm.types import CompletionParams, Message

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import ModelAPIError

T = TypeVar("T", bound="BaseModel")

logger = logging.getLogger(__name__)


def _rejects_temperature(exc: ModelAPIError) -> bool:
    # No model list to keep current: a model that takes no sampling
    # parameter refuses a request that carries one and names it.
    # pydantic-ai reports the refusal as either class, by provider.
    from pydantic_ai.exceptions import ModelHTTPError, ModelRequestRejectedError

    refused = isinstance(exc, ModelRequestRejectedError) or (
        isinstance(exc, ModelHTTPError) and exc.status_code == 400
    )
    return refused and "temperature" in str(exc)


class PydanticAIProvider:
    """pydantic-ai provider for multi-provider LLM access."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        defaults: CompletionParams | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
    ) -> None:
        from recollect.config import config

        self._timeout = timeout or float(config.get("extraction.timeout", 120))
        self._model = model or str(config.get("extraction.pydantic_ai_model", ""))
        if not self._model:
            raise ExtractionError(
                "pydantic-ai model not configured. Pass model= or set "
                "PYDANTIC_AI_MODEL environment variable."
            )

        if defaults is None:
            raw_settings = config.get("extraction.model_settings", {}) or {}
            defaults = CompletionParams(**raw_settings)
        self._defaults = defaults
        if max_retries is None:
            max_retries = int(config.get("extraction.max_retries", 5))
        self._max_retries = max_retries
        self._omit_temperature = False

        if api_key or base_url:
            logger.debug(
                "api_key/base_url ignored for pydantic-ai provider; "
                "credentials are managed via environment variables"
            )

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _build_model_settings(
        defaults: CompletionParams,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # extras pass through; pydantic-ai ModelSettings validates downstream.
        effective = {**defaults.model_dump(exclude_none=True), **kwargs}
        max_tokens = int(effective.pop("max_tokens", 8192))
        temperature = float(effective.pop("temperature", 0.0))
        return {"max_tokens": max_tokens, "temperature": temperature, **effective}

    async def _run(
        self, agent: Agent[None, Any], user_prompt: str, kwargs: dict[str, Any]
    ) -> Any:
        """Run the agent; serve a model that rejects temperature without it."""
        from pydantic_ai.exceptions import ModelAPIError
        from pydantic_ai.settings import ModelSettings

        settings = self._build_model_settings(self._defaults, **kwargs)
        without = {k: v for k, v in settings.items() if k != "temperature"}
        omitted = self._omit_temperature
        if omitted:
            settings = without
        try:
            return await agent.run(
                user_prompt, model_settings=cast(ModelSettings, settings)
            )
        except ModelAPIError as exc:
            # What this request sent, not the flag: a concurrent request can
            # set the flag while this one is in flight with temperature.
            if omitted or not _rejects_temperature(exc):
                raise
            logger.exception(
                "Model %s rejects temperature; omitting it for this provider",
                self._model,
            )
            # Remembered for the provider's lifetime: the rejection is a
            # property of the model, not of one request.
            self._omit_temperature = True
            return await agent.run(
                user_prompt, model_settings=cast(ModelSettings, without)
            )

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> str:
        from pydantic_ai import Agent
        from pydantic_ai.exceptions import (
            ModelAPIError,
            UnexpectedModelBehavior,
            UserError,
        )

        system_prompt = "\n".join(m.content for m in messages if m.role == "system")
        user_prompt = "\n".join(m.content for m in messages if m.role != "system")

        agent: Agent[None, str] = Agent(
            self._model,
            system_prompt=system_prompt,
            output_type=str,
            retries=self._max_retries,
        )

        try:
            result = await self._run(agent, user_prompt, kwargs)
            text: str = result.output
            if not text:
                raise ExtractionError(
                    f"Empty response from pydantic-ai ({self._model})."
                )
            return text
        except ExtractionError:
            raise
        except (ModelAPIError, UnexpectedModelBehavior) as exc:
            logger.exception("pydantic-ai API error for model %s", self._model)
            raise ExtractionError(
                f"pydantic-ai API error ({self._model}): {exc}"
            ) from exc
        except UserError as exc:
            logger.exception(
                "pydantic-ai configuration error for model %s", self._model
            )
            raise ExtractionError(
                f"pydantic-ai configuration error ({self._model}): {exc}"
            ) from exc

    async def complete_structured(
        self,
        messages: list[Message],
        output_type: type[T],
        **kwargs: Any,
    ) -> T:
        from pydantic_ai import Agent
        from pydantic_ai.exceptions import (
            ModelAPIError,
            UnexpectedModelBehavior,
            UserError,
        )

        system_prompt = "\n".join(m.content for m in messages if m.role == "system")
        user_prompt = "\n".join(m.content for m in messages if m.role != "system")

        agent: Agent[None, T] = Agent(
            self._model,
            system_prompt=system_prompt,
            output_type=output_type,
            retries=self._max_retries,
        )

        try:
            result = await self._run(agent, user_prompt, kwargs)
            output: T = result.output
            return output
        except ExtractionError:
            raise
        except (ModelAPIError, UnexpectedModelBehavior) as exc:
            logger.exception("pydantic-ai API error for model %s", self._model)
            raise ExtractionError(
                f"pydantic-ai API error ({self._model}): {exc}"
            ) from exc
        except UserError as exc:
            logger.exception(
                "pydantic-ai configuration error for model %s", self._model
            )
            raise ExtractionError(
                f"pydantic-ai configuration error ({self._model}): {exc}"
            ) from exc
