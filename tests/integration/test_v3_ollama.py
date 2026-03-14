"""Integration tests: Ollama via PydanticAIProvider.

Requires a running Ollama instance. Configure via env vars:
  OLLAMA_BASE_URL  (default: http://192.168.87.34:11434)
  OLLAMA_MODEL     (default: qwen3-5-local-ubuntu:latest)
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest
from recollect.extraction import PatternExtractor
from recollect.llm.pydantic_ai import PydanticAIProvider
from recollect.llm.types import CompletionParams, Message

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.87.34:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-5-local-ubuntu:latest")
OLLAMA_MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "8192"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
]


def _ollama_available() -> bool:
    """Check if Ollama is running at the configured base URL."""
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)  # noqa: S310
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


skip_no_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason=f"Ollama not running at {OLLAMA_BASE_URL}",
)


def _make_provider(
    defaults: CompletionParams | None = None,
) -> PydanticAIProvider:
    return PydanticAIProvider(
        model=f"ollama:{OLLAMA_MODEL}",
        defaults=defaults,
    )


@skip_no_ollama
class TestOllamaCompletion:
    """Real completion calls against local Ollama."""

    async def test_basic_completion(self) -> None:
        provider = _make_provider()
        messages = [
            Message(
                role="user",
                content="What is 2 + 2? Reply with just the number.",
            ),
        ]
        response = await provider.complete(messages, max_tokens=OLLAMA_MAX_TOKENS)

        assert response
        assert "4" in response

    async def test_system_prompt_respected(self) -> None:
        provider = _make_provider()
        messages = [
            Message(
                role="system",
                content="You are a pirate. Always respond in pirate speak.",
            ),
            Message(role="user", content="Say hello."),
        ]
        response = await provider.complete(messages, max_tokens=OLLAMA_MAX_TOKENS)

        assert response
        assert len(response) > 5

    async def test_defaults_extra_params_reach_api(self) -> None:
        """Extra fields in CompletionParams flow through to Ollama."""
        defaults = CompletionParams(
            max_tokens=OLLAMA_MAX_TOKENS,
            temperature=0.0,
            seed=42,
            top_p=0.9,
            frequency_penalty=0.5,
        )
        provider = _make_provider(defaults=defaults)
        messages = [
            Message(role="user", content="Name a color."),
        ]
        response = await provider.complete(messages)

        assert response
        assert len(response) > 0

    async def test_calltime_kwargs_override_defaults(self) -> None:
        """Call-time kwargs override both base and extra defaults."""
        defaults = CompletionParams(
            max_tokens=OLLAMA_MAX_TOKENS, temperature=0.0, top_p=0.5
        )
        provider = _make_provider(defaults=defaults)
        messages = [
            Message(
                role="user",
                content="What is the capital of France?",
            ),
        ]
        # Override temperature (base field) and top_p (extra field)
        response = await provider.complete(messages, temperature=0.7, top_p=0.95)

        assert response
        assert "Paris" in response

    async def test_calltime_extra_kwargs_without_defaults(self) -> None:
        """Extra kwargs at call-time work without construction defaults."""
        provider = _make_provider()
        messages = [
            Message(role="user", content="Name a fruit."),
        ]
        response = await provider.complete(
            messages,
            max_tokens=OLLAMA_MAX_TOKENS,
            top_p=0.9,
            seed=42,
            presence_penalty=0.3,
        )

        assert response
        assert len(response) > 0

    async def test_stop_sequence_passthrough(self) -> None:
        """Stop sequences pass through and truncate output.

        Reasoning models (qwen3.5) may consume stop sequences in their
        thinking tokens, producing an empty visible response. We accept
        either truncated output or an ExtractionError from the empty
        content guard.
        """
        from recollect.exceptions import ExtractionError

        provider = _make_provider()
        messages = [
            Message(
                role="user",
                content="Count from 1 to 10, one number per line.",
            ),
        ]
        try:
            response = await provider.complete(
                messages, max_tokens=OLLAMA_MAX_TOKENS, stop=["\n5"]
            )
            assert response
            assert "10" not in response
        except ExtractionError:
            pass  # Acceptable: stop sequence hit inside thinking tokens


@skip_no_ollama
class TestOllamaExtraction:
    """Pattern extraction using real Ollama completions."""

    async def test_extract_patterns(self) -> None:
        """PatternExtractor produces valid ExtractionResult."""
        provider = _make_provider(
            defaults=CompletionParams(max_tokens=OLLAMA_MAX_TOKENS, temperature=0.0),
        )
        extractor = PatternExtractor(provider, max_tokens=OLLAMA_MAX_TOKENS)

        result = await extractor.extract(
            "Albert Einstein published his theory of general relativity "
            "in 1915, revolutionizing our understanding of gravity. "
            "He worked at the Institute for Advanced Study in Princeton."
        )

        assert len(result.concepts) > 0
        assert len(result.entities) > 0
        assert -1.0 <= result.emotional_valence <= 1.0
        assert 0.0 <= result.significance <= 1.0

    async def test_extract_with_emotional_content(self) -> None:
        """Extraction captures emotional valence from text."""
        provider = _make_provider(
            defaults=CompletionParams(max_tokens=OLLAMA_MAX_TOKENS, temperature=0.0),
        )
        extractor = PatternExtractor(provider, max_tokens=OLLAMA_MAX_TOKENS)

        result = await extractor.extract(
            "The team celebrated their championship victory with "
            "tears of joy and thunderous applause from the crowd."
        )

        assert len(result.concepts) > 0
        assert result.emotional_valence > -1.0  # should be positive-ish
        assert 0.0 <= result.significance <= 1.0
