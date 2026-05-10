"""Pattern extraction using any LLM provider.

Sends text to an LLM with a structured extraction prompt,
parses the JSON response into an ExtractionResult.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from recollect.exceptions import ExtractionError, PromptValidationError
from recollect.llm.types import ExtractionResult, Message
from recollect.prompts import load_packaged_default, load_prompt_file

if TYPE_CHECKING:
    from recollect.config import MemoryConfig
    from recollect.llm.protocol import LLMProvider

logger = logging.getLogger(__name__)


class PatternExtractor:
    """Extracts structured patterns from text using any LLM provider."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        config: MemoryConfig | None = None,
        max_tokens: int | None = None,
    ) -> None:
        from recollect.config import config as default_config

        self._provider = provider
        self._config = config or default_config
        if max_tokens is None:
            max_tokens = int(self._config.get("extraction.max_tokens", 8192))
        self._max_tokens = max_tokens

        template_path = str(
            self._config.get("extraction.template_path", "") or ""
        ).strip()
        if template_path:
            loaded = load_prompt_file(template_path)
            if loaded.applies_to != "extraction-template":
                raise PromptValidationError(
                    f"{template_path}: applies-to must be 'extraction-template', "
                    f"got '{loaded.applies_to}'"
                )
        else:
            loaded = load_packaged_default("extraction.default.md")
        self._template_body = loaded.body
        self._template_version = loaded.version
        logger.info(
            "Extraction template loaded: version=%s source=%s",
            loaded.version,
            loaded.source,
        )

    @property
    def template_version(self) -> str:
        """Version string from the loaded template's header (for telemetry)."""
        return self._template_version

    def _build_prompt(self) -> str:
        """Format extraction template with config values."""
        prompt = self._template_body.format(
            max_concepts=int(self._config.get("extraction.max_concepts", 5)),
            max_relations=int(self._config.get("extraction.max_relations", 3)),
        )
        instructions = self._config.extraction_instructions
        if instructions:
            prompt = f"{prompt}\n\n{instructions}"
        return prompt

    async def extract(self, text: str) -> ExtractionResult:
        """Extract patterns from text via LLM structured output."""
        messages = [
            Message(role="system", content=self._build_prompt()),
            Message(role="user", content=text),
        ]
        try:
            return await self._provider.complete_structured(
                messages,
                ExtractionResult,
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Provider error during extraction: {exc}") from exc
