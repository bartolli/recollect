"""Closed-form answer scoring for the task-eval arm.

Correct = any alias present in the normalized response at word boundaries.
Abstain questions invert: an abstention marker is the correct answer, any
concrete value is a fabrication. An abstention on an answerable question
scores wrong via the alias check alone -- markers carry no alias vocabulary.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probe_cli.corpus import TaskQuestion

_ABSTAIN_MARKERS = (
    "unknown",
    "don t know",
    "do not know",
    "no information",
    "not mentioned",
    "not specified",
    "cannot determine",
    "can t determine",
    "no memory of",
)


def normalize(text: str) -> str:
    """Lowercase, strip non-alphanumerics to spaces, collapse whitespace."""
    lowered = text.lower()
    stripped = re.sub(r"[^a-z0-9]+", " ", lowered)
    return stripped.strip()


def is_abstention(response: str) -> bool:
    norm = normalize(response)
    return any(marker in norm for marker in _ABSTAIN_MARKERS)


def score_answer(response: str, question: TaskQuestion) -> bool:
    if question.answer_type == "abstain":
        return is_abstention(response)
    norm = normalize(response)
    for alias in question.answers:
        alias_norm = normalize(alias)
        if not alias_norm:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", norm):
            return True
    return False
