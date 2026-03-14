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

T = TypeVar("T", bound="BaseModel")

logger = logging.getLogger(__name__)


class PydanticAIProvider:
    """pydantic-ai provider for multi-provider LLM access."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        defaults: CompletionParams | None = None,
        max_retries: int = 2,
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

        self._defaults = defaults or CompletionParams()
        self._max_retries = max_retries

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
        """Merge defaults and overrides into a pydantic-ai ModelSettings dict."""
        effective = {**defaults.model_dump(exclude_none=True), **kwargs}
        return {
            "max_tokens": int(effective.pop("max_tokens", 1024)),
            "temperature": float(effective.pop("temperature", 0.0)),
        }

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> str:
        from pydantic_ai import Agent
        from pydantic_ai.exceptions import (
            ModelHTTPError,
            UnexpectedModelBehavior,
            UserError,
        )
        from pydantic_ai.settings import ModelSettings

        system_prompt = "\n".join(m.content for m in messages if m.role == "system")
        user_prompt = "\n".join(m.content for m in messages if m.role != "system")
        settings = cast(
            ModelSettings, self._build_model_settings(self._defaults, **kwargs)
        )

        agent: Agent[None, str] = Agent(
            self._model,
            system_prompt=system_prompt,
            output_type=str,
            retries=self._max_retries,
        )

        try:
            result = await agent.run(user_prompt, model_settings=settings)
            text: str = result.output
            if not text:
                raise ExtractionError(
                    f"Empty response from pydantic-ai ({self._model})."
                )
            return text
        except ExtractionError:
            raise
        except (ModelHTTPError, UnexpectedModelBehavior) as exc:
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
            ModelHTTPError,
            UnexpectedModelBehavior,
            UserError,
        )
        from pydantic_ai.settings import ModelSettings

        system_prompt = "\n".join(m.content for m in messages if m.role == "system")
        user_prompt = "\n".join(m.content for m in messages if m.role != "system")
        settings = cast(
            ModelSettings, self._build_model_settings(self._defaults, **kwargs)
        )

        agent: Agent[None, T] = Agent(
            self._model,
            system_prompt=system_prompt,
            output_type=output_type,
            retries=self._max_retries,
        )

        try:
            result = await agent.run(user_prompt, model_settings=settings)
            output: T = result.output
            return output
        except ExtractionError:
            raise
        except (ModelHTTPError, UnexpectedModelBehavior) as exc:
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
