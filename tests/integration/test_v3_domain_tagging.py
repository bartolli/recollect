"""Integration: extraction tags `domains` by world knowledge, not lexicon (1.6).

The B-over-lexicon guard. A hardcoded food-string lexicon passes every unit
test in the surfacing suite and FAILS here: these writes name no enumerable food
token, yet must tag `domains: [food]` for the safety gate to fire. Without this
guard a future `extraction.default.md` edit could silently degrade domain tagging
to lexical pattern-matching with the unit suite still green.

Requires a configured extraction model:
  PYDANTIC_AI_MODEL  (e.g. anthropic:claude-haiku-4-5-20251001)
plus that provider's key in the environment. Run with `-m slow`.
"""

from __future__ import annotations

import os

import pytest
from recollect.extraction import PatternExtractor
from recollect.llm.pydantic_ai import PydanticAIProvider

EXTRACTION_MODEL = os.environ.get("PYDANTIC_AI_MODEL", "").strip()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not EXTRACTION_MODEL, reason="PYDANTIC_AI_MODEL not set"
    ),
]


async def _domains(extractor: PatternExtractor, text: str) -> set[str]:
    result = await extractor.extract(text)
    return set(result.domains)


class TestDomainTagging:
    @pytest.fixture
    def extractor(self) -> PatternExtractor:
        return PatternExtractor(
            PydanticAIProvider(model=EXTRACTION_MODEL), max_tokens=8192
        )

    @pytest.mark.parametrize(
        "text", ["grabbing omakase with Sara", "izakaya after work"]
    )
    async def test_cuisine_token_tags_food(
        self, extractor: PatternExtractor, text: str
    ) -> None:
        # World-knowledge cue: a named cuisine/venue implies food, though the
        # word "food"/"restaurant" never appears. A lexicon misses it.
        assert "food" in await _domains(extractor, text)

    @pytest.mark.parametrize(
        "text", ["having dinner with Sara tonight", "grabbing a bite after work"]
    )
    async def test_meal_context_without_cuisine_tags_food(
        self, extractor: PatternExtractor, text: str
    ) -> None:
        # The harder boundary: a meal context with NO cuisine token at all must
        # still tag food (not social-only). This is where the safety gate is
        # silently lost if the model reads it as a purely social event.
        assert "food" in await _domains(extractor, text)

    async def test_non_food_write_omits_food(
        self, extractor: PatternExtractor
    ) -> None:
        # Negative contrast: domain tagging is selective, not "everything food."
        assert "food" not in await _domains(
            extractor, "fixed the parser bug in the auth module"
        )
