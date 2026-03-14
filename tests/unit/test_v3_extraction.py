"""Tests for pattern extraction.

Focus: prompt construction, provider delegation, error handling.
Structured output parsing is now handled by the provider layer.
"""

from __future__ import annotations

from typing import Any

import pytest
from recollect.exceptions import ExtractionError
from recollect.extraction import PatternExtractor
from recollect.llm.types import ExtractionResult, Message

# -- Mock provider --


class _CapturingProvider:
    """Mock provider that captures call args and returns canned result."""

    def __init__(self, result: ExtractionResult) -> None:
        self._result = result
        self.last_messages: list[Message] = []
        self.last_output_type: type[Any] | None = None
        self.last_max_tokens: int = 0
        self.last_temperature: float = 0.0

    @property
    def model_name(self) -> str:
        return "mock-model"

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> str:
        return ""

    async def complete_structured(
        self,
        messages: list[Message],
        output_type: type[Any],
        **kwargs: Any,
    ) -> ExtractionResult:
        self.last_messages = list(messages)
        self.last_output_type = output_type
        self.last_max_tokens = int(kwargs.get("max_tokens", 1024))
        self.last_temperature = float(kwargs.get("temperature", 0.0))
        return self._result


# -- Extraction flow --


class TestExtractorFlow:
    """Full extraction pipeline with mock provider."""

    async def test_sends_system_and_user_messages(self) -> None:
        provider = _CapturingProvider(ExtractionResult())
        extractor = PatternExtractor(provider)

        await extractor.extract("Some text to analyze")

        assert len(provider.last_messages) == 2
        assert provider.last_messages[0].role == "system"
        assert provider.last_messages[1].role == "user"
        assert provider.last_messages[1].content == "Some text to analyze"

    async def test_passes_extraction_result_type(self) -> None:
        provider = _CapturingProvider(ExtractionResult())
        extractor = PatternExtractor(provider)

        await extractor.extract("anything")

        assert provider.last_output_type is ExtractionResult

    async def test_system_prompt_mentions_json(self) -> None:
        provider = _CapturingProvider(ExtractionResult())
        extractor = PatternExtractor(provider)

        await extractor.extract("anything")

        system = provider.last_messages[0].content
        assert "JSON" in system or "json" in system

    async def test_uses_low_temperature(self) -> None:
        provider = _CapturingProvider(ExtractionResult())
        extractor = PatternExtractor(provider)

        await extractor.extract("anything")
        assert provider.last_temperature == 0.0

    async def test_returns_extraction_result(self) -> None:
        expected = ExtractionResult(
            concepts=["python", "coding"],
            significance=0.8,
            emotional_valence=0.5,
        )
        provider = _CapturingProvider(expected)
        extractor = PatternExtractor(provider)

        result = await extractor.extract("Guido created Python")

        assert isinstance(result, ExtractionResult)
        assert result.concepts == ["python", "coding"]
        assert result.significance == 0.8

    async def test_wraps_provider_error(self) -> None:
        class _FailingProvider:
            model_name = "fail"

            async def complete(self, *_a: object, **_kw: object) -> str:
                return ""

            async def complete_structured(
                self, *_a: object, **_kw: object
            ) -> ExtractionResult:
                msg = "connection refused"
                raise ConnectionError(msg)

        extractor = PatternExtractor(_FailingProvider())

        with pytest.raises(ExtractionError, match="connection refused"):
            await extractor.extract("anything")

    async def test_extraction_error_not_double_wrapped(self) -> None:
        class _ExtractionFailProvider:
            model_name = "fail"

            async def complete(self, *_a: object, **_kw: object) -> str:
                return ""

            async def complete_structured(
                self, *_a: object, **_kw: object
            ) -> ExtractionResult:
                raise ExtractionError("schema mismatch")

        extractor = PatternExtractor(_ExtractionFailProvider())

        with pytest.raises(ExtractionError, match="schema mismatch"):
            await extractor.extract("anything")

    async def test_respects_max_tokens(self) -> None:
        provider = _CapturingProvider(ExtractionResult())
        extractor = PatternExtractor(provider, max_tokens=2048)

        await extractor.extract("anything")
        assert provider.last_max_tokens == 2048
