"""Prompt loading and validation.

Header schema (markdown `# key: value` block, terminated by blank line):

    # version: <semver>
    # applies-to: extraction-template | extraction-instructions | situational
    # placeholders: name1, name2

Validation runs at load time. Bad prompt -> startup failure.
"""

from __future__ import annotations

import logging
import re
from importlib import resources
from pathlib import Path
from typing import Literal, NoReturn, cast

from pydantic import BaseModel, Field

from recollect.exceptions import PromptValidationError

logger = logging.getLogger(__name__)

AppliesTo = Literal[
    "extraction-template",
    "extraction-instructions",
    "situational",
]

_HEADER_LINE = re.compile(r"^#\s*([\w-]+)\s*:\s*(.+?)\s*$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[\w.]+)?(?:\+[\w.]+)?$")
_PLACEHOLDER = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")
_SECTION_SYSTEM = re.compile(r"^##\s+System Prompt\s*$", re.MULTILINE)
_SECTION_USER = re.compile(r"^##\s+User Prompt\s*$", re.MULTILINE)

_BANNED_TOKENS: tuple[str, ...] = (
    "let me ",
    "i've gone ahead",
    "i think ",
    "perhaps ",
    "it seems ",
    "in order to",
    "the fact that",
    "happy to",
    "of course",
)


class LoadedPrompt(BaseModel):
    """Validated prompt loaded from a file or packaged resource."""

    version: str
    applies_to: AppliesTo
    placeholders: list[str] = Field(default_factory=list)
    body: str = ""
    system: str = ""
    user: str = ""
    source: str = "<inline>"


def load_prompt_file(path: Path | str) -> LoadedPrompt:
    """Read and validate a prompt override file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return parse_prompt(text, source=str(p))


def load_packaged_default(name: str) -> LoadedPrompt:
    """Load a default prompt shipped inside the recollect.prompts package."""
    text = (
        resources.files("recollect.prompts")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )
    return parse_prompt(text, source=f"packaged:{name}")


_APPLIES_TO_VALUES: tuple[AppliesTo, ...] = (
    "extraction-template",
    "extraction-instructions",
    "situational",
)


def parse_prompt(text: str, *, source: str = "<inline>") -> LoadedPrompt:
    """Parse and validate a prompt file body. Source is used in error messages."""
    header, body = _split_header(text, source)

    version = header.get("version") or _fail(
        source, "missing required header 'version'"
    )
    if not _SEMVER.match(version):
        _fail(source, f"version '{version}' is not semver")

    applies_to_raw = header.get("applies-to") or _fail(
        source, "missing required header 'applies-to'"
    )
    if applies_to_raw not in _APPLIES_TO_VALUES:
        _fail(source, f"unknown applies-to '{applies_to_raw}'")
    applies_to = cast(AppliesTo, applies_to_raw)

    placeholders = [
        p.strip() for p in header.get("placeholders", "").split(",") if p.strip()
    ]

    body = body.strip()
    if not body:
        _fail(source, "empty body after header")

    if applies_to == "situational":
        system, user = _split_situational(body, source)
        _check_placeholders(user, placeholders, source, location="user")
        _check_no_placeholders(system, placeholders, source, location="system")
        _scan_banned(system + "\n" + user, source)
        return LoadedPrompt(
            version=version,
            applies_to=applies_to,
            placeholders=placeholders,
            system=system,
            user=user,
            source=source,
        )

    _check_placeholders(body, placeholders, source, location="body")
    _scan_banned(body, source)
    return LoadedPrompt(
        version=version,
        applies_to=applies_to,
        placeholders=placeholders,
        body=body,
        source=source,
    )


def _split_header(text: str, source: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    header: dict[str, str] = {}
    for i, line in enumerate(lines):
        if not line.strip():
            return header, "\n".join(lines[i + 1 :])
        m = _HEADER_LINE.match(line)
        if not m:
            _fail(source, f"line {i + 1}: malformed header line: {line!r}")
        header[m.group(1)] = m.group(2)
    return _fail(source, "no body found (header never terminated by blank line)")


def _split_situational(body: str, source: str) -> tuple[str, str]:
    sys_match = _SECTION_SYSTEM.search(body)
    if sys_match is None:
        _fail(source, "missing '## System Prompt' section")
    user_match = _SECTION_USER.search(body, sys_match.end())
    if user_match is None:
        _fail(source, "missing '## User Prompt' section after system")
    system = body[sys_match.end() : user_match.start()].strip()
    user = body[user_match.end() :].strip()
    if not system:
        _fail(source, "empty system body")
    if not user:
        _fail(source, "empty user body")
    return system, user


def _check_placeholders(
    body: str, declared: list[str], source: str, *, location: str
) -> None:
    found = set(_PLACEHOLDER.findall(body))
    declared_set = set(declared)
    missing = declared_set - found
    extra = found - declared_set
    if missing:
        _fail(
            source,
            f"{location}: declared placeholders absent from body: {sorted(missing)}",
        )
    if extra:
        _fail(
            source,
            f"{location}: placeholders in body absent from header: {sorted(extra)}",
        )
    # Dry-run .format() — catches stray `{...}` (e.g. comma-lists) that the
    # word-only _PLACEHOLDER regex misses but that crash runtime substitution.
    try:
        body.format(**dict.fromkeys(declared, ""))
    except (KeyError, IndexError, ValueError) as exc:
        _fail(
            source,
            f"{location}: stray brace not escaped as {{{{...}}}}: {exc}",
        )


def _check_no_placeholders(
    body: str, declared: list[str], source: str, *, location: str
) -> None:
    found = set(_PLACEHOLDER.findall(body))
    overlap = found & set(declared)
    if overlap:
        _fail(
            source,
            f"{location}: must not contain user-template placeholders: "
            f"{sorted(overlap)}",
        )


def _scan_banned(body: str, source: str) -> None:
    body_lower = body.lower()
    hits = [t for t in _BANNED_TOKENS if t in body_lower]
    if hits:
        logger.warning(
            "Prompt %s contains banned tokens (soft-warn, see glossary): %s",
            source,
            hits,
        )


def _fail(source: str, msg: str) -> NoReturn:
    raise PromptValidationError(f"{source}: {msg}")
