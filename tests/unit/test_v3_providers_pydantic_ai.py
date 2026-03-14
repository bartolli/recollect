"""Tests for pydantic-ai LLM provider adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from recollect.exceptions import ExtractionError
from recollect.llm.types import CompletionParams, Message

try:
    import pydantic_ai as _pydantic_ai  # noqa: F401

    _has_pydantic_ai = True
except ImportError:
    _has_pydantic_ai = False

pytestmark = pytest.mark.skipif(
    not _has_pydantic_ai, reason="pydantic-ai not installed"
)


# -- _build_model_settings --


class TestPydanticAIModelSettings:
    """Verify defaults and override merging for model settings."""

    def test_defaults_applied(self) -> None:
        from recollect.llm.pydantic_ai import PydanticAIProvider

        result = PydanticAIProvider._build_model_settings(CompletionParams())
        assert result["max_tokens"] == 1024
        assert result["temperature"] == 0.0

    def test_kwargs_override_defaults(self) -> None:
        from recollect.llm.pydantic_ai import PydanticAIProvider

        result = PydanticAIProvider._build_model_settings(
            CompletionParams(), temperature=0.5
        )
        assert result["temperature"] == 0.5

    def test_custom_defaults(self) -> None:
        from recollect.llm.pydantic_ai import PydanticAIProvider

        defaults = CompletionParams(max_tokens=2048, temperature=0.3)
        result = PydanticAIProvider._build_model_settings(defaults)
        assert result["max_tokens"] == 2048
        assert result["temperature"] == 0.3


# -- Construction --


class TestPydanticAIConstruction:
    """Verify constructor model resolution and error paths."""

    def test_constructs_with_model(self) -> None:
        from recollect.llm.pydantic_ai import PydanticAIProvider

        provider = PydanticAIProvider(model="openai:gpt-4o")
        assert provider.model_name == "openai:gpt-4o"

    def test_no_model_raises(self) -> None:
        from recollect.llm.pydantic_ai import PydanticAIProvider

        with patch("recollect.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default="": {
                "extraction.pydantic_ai_model": "",
                "extraction.timeout": 120,
            }.get(key, default)
            with pytest.raises(ExtractionError, match="not configured"):
                PydanticAIProvider()

    def test_reads_model_from_config(self) -> None:
        from recollect.llm.pydantic_ai import PydanticAIProvider

        with patch("recollect.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default="": {
                "extraction.pydantic_ai_model": "anthropic:claude-haiku-4-5-20251001",
                "extraction.timeout": 120,
            }.get(key, default)
            provider = PydanticAIProvider()
        assert provider.model_name == "anthropic:claude-haiku-4-5-20251001"


# -- complete() --


class TestPydanticAIComplete:
    """Tests for PydanticAIProvider.complete()."""

    @pytest.fixture()
    def provider(self) -> Any:
        from recollect.llm.pydantic_ai import PydanticAIProvider

        return PydanticAIProvider(model="openai:gpt-4o")

    @pytest.mark.asyncio()
    async def test_complete_returns_text(self, provider: Any) -> None:
        mock_result = MagicMock()
        mock_result.output = "hello world"

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch(
            "pydantic_ai.Agent", return_value=mock_agent
        ) as mock_cls:
            result = await provider.complete([Message(role="user", content="hi")])

        assert result == "hello world"
        mock_cls.assert_called_once()

    @pytest.mark.asyncio()
    async def test_complete_empty_raises(self, provider: Any) -> None:
        mock_result = MagicMock()
        mock_result.output = ""

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with (
            patch("pydantic_ai.Agent", return_value=mock_agent),
            pytest.raises(ExtractionError, match="Empty response"),
        ):
            await provider.complete([Message(role="user", content="hi")])

    @pytest.mark.asyncio()
    async def test_complete_api_error_raises_extraction_error(
        self, provider: Any
    ) -> None:
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=UnexpectedModelBehavior("test error"))

        with (
            patch("pydantic_ai.Agent", return_value=mock_agent),
            pytest.raises(ExtractionError, match="pydantic-ai API error"),
        ):
            await provider.complete([Message(role="user", content="hi")])


# -- complete_structured() --


class TestPydanticAICompleteStructured:
    """Tests for PydanticAIProvider.complete_structured()."""

    @pytest.fixture()
    def provider(self) -> Any:
        from recollect.llm.pydantic_ai import PydanticAIProvider

        return PydanticAIProvider(model="openai:gpt-4o")

    @pytest.mark.asyncio()
    async def test_returns_validated_model(self, provider: Any) -> None:
        from recollect.llm.types import ExtractionResult

        expected = ExtractionResult(
            concepts=["test"],
            relations=[],
            entities=[],
            emotional_valence=0.0,
            significance=0.7,
            fact_type="semantic",
        )
        mock_result = MagicMock()
        mock_result.output = expected

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("pydantic_ai.Agent", return_value=mock_agent):
            result = await provider.complete_structured(
                [Message(role="user", content="hi")],
                ExtractionResult,
            )

        assert isinstance(result, ExtractionResult)
        assert result.concepts == ["test"]
        assert result.significance == 0.7

    @pytest.mark.asyncio()
    async def test_api_error_raises_extraction_error(self, provider: Any) -> None:
        from pydantic_ai.exceptions import UnexpectedModelBehavior
        from recollect.llm.types import ExtractionResult

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=UnexpectedModelBehavior("test error"))

        with (
            patch("pydantic_ai.Agent", return_value=mock_agent),
            pytest.raises(ExtractionError, match="pydantic-ai API error"),
        ):
            await provider.complete_structured(
                [Message(role="user", content="hi")],
                ExtractionResult,
            )
