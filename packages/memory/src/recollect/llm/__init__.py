"""LLM provider abstraction for Memory SDK.

Single provider: pydantic-ai routes to 20+ model backends
(Anthropic, OpenAI, Ollama, LM Studio, and more).
"""

from recollect.llm.protocol import LLMProvider
from recollect.llm.pydantic_ai import PydanticAIProvider
from recollect.llm.types import (
    CompletionParams,
    Entity,
    ExtractionResult,
    Message,
    Relation,
)

__all__ = [
    "CompletionParams",
    "Entity",
    "ExtractionResult",
    "LLMProvider",
    "Message",
    "PydanticAIProvider",
    "Relation",
]
