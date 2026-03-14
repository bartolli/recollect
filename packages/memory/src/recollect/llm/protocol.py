"""LLM provider protocol.

Any class implementing this protocol can be used with PatternExtractor
and other cognitive tasks. Ships with PydanticAIProvider, which routes
to 20+ model backends via pydantic-ai.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from pydantic import BaseModel

    from recollect.llm.types import Message

T = TypeVar("T", bound="BaseModel")


class LLMProvider(Protocol):
    """Minimal interface for LLM completion."""

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> str:
        """Send messages and return the text response."""
        ...

    async def complete_structured(
        self,
        messages: list[Message],
        output_type: type[T],
        **kwargs: Any,
    ) -> T:
        """Send messages and return a validated Pydantic model instance."""
        ...

    @property
    def model_name(self) -> str:
        """Model identifier for logging and debugging."""
        ...
