"""Run provenance: the extraction prompt as sent, the failures logged and swallowed."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from contextlib import contextmanager
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterator

    from recollect.extraction import PatternExtractor

_CAPTURED = ("recollect", "probe_cli")
# The provider layer logs and re-raises, or recovers by re-sending the same
# request; a failure that is swallowed is logged again by the consumer that
# swallows it.
_PROVIDER = "recollect.llm"


class PromptStamp(BaseModel):
    # sha256 and chars of the assembled system message (template body formatted,
    # plus extraction.instructions), not of the template file.
    version: str = ""
    sha256: str = ""
    chars: int = Field(default=0, ge=0)


def stamp_prompt(extractor: PatternExtractor) -> PromptStamp:
    prompt = extractor._build_prompt()
    return PromptStamp(
        version=extractor.template_version,
        sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        chars=len(prompt),
    )


class _SwallowedCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.counts: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == _PROVIDER or record.name.startswith(f"{_PROVIDER}."):
            return
        # The unformatted template: ids in the arguments do not split a key.
        self.counts[f"{record.name}: {record.msg}"] += 1


@contextmanager
def capture_swallowed() -> Iterator[Counter[str]]:
    handler = _SwallowedCounter()
    loggers = [logging.getLogger(name) for name in _CAPTURED]
    for logger in loggers:
        logger.addHandler(handler)
    try:
        yield handler.counts
    finally:
        for logger in loggers:
            logger.removeHandler(handler)
