"""Tests for prompt loading and validation (P2)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from recollect.exceptions import PromptValidationError
from recollect.prompts import (
    load_packaged_default,
    load_prompt_file,
    parse_prompt,
)


def _hdr(applies: str = "extraction-template", placeholders: str = "x") -> str:
    return (
        "# version: 1.0.0\n"
        f"# applies-to: {applies}\n"
        f"# placeholders: {placeholders}\n"
    )


class TestParseValid:
    def test_extraction_template(self) -> None:
        text = _hdr() + "\nbody with {x} placeholder"
        p = parse_prompt(text)
        assert p.version == "1.0.0"
        assert p.applies_to == "extraction-template"
        assert p.placeholders == ["x"]
        assert "{x}" in p.body

    def test_situational_with_sections(self) -> None:
        text = (
            _hdr("situational", "new_content, numbered_list, existing_groups")
            + "\n## System Prompt\n\nsys body, no placeholders.\n\n"
            + "## User Prompt\n\nq: {new_content}\nrelated: {numbered_list}\n"
            + "groups: {existing_groups}\n"
        )
        p = parse_prompt(text)
        assert p.applies_to == "situational"
        assert "no placeholders" in p.system
        assert "{new_content}" in p.user

    def test_extraction_instructions_suffix(self) -> None:
        text = _hdr("extraction-instructions", "") + "\nplain suffix text"
        p = parse_prompt(text)
        assert p.applies_to == "extraction-instructions"
        assert p.placeholders == []

    def test_double_brace_escapes_ignored_by_placeholder_check(self) -> None:
        text = _hdr(placeholders="x") + '\n{x} and example {{"key": "val"}}'
        parse_prompt(text)

    def test_semver_with_prerelease(self) -> None:
        text = _hdr().replace("1.0.0", "0.0.0-dev") + "\nbody {x}"
        p = parse_prompt(text)
        assert p.version == "0.0.0-dev"


class TestParseRejects:
    def test_missing_version(self) -> None:
        text = "# applies-to: extraction-template\n# placeholders: x\n\nbody {x}"
        with pytest.raises(PromptValidationError, match="version"):
            parse_prompt(text)

    def test_non_semver_version(self) -> None:
        text = "# version: v1\n# applies-to: extraction-template\n\nbody"
        with pytest.raises(PromptValidationError, match="semver"):
            parse_prompt(text)

    def test_missing_applies_to(self) -> None:
        text = "# version: 1.0.0\n# placeholders: x\n\nbody {x}"
        with pytest.raises(PromptValidationError, match="applies-to"):
            parse_prompt(text)

    def test_unknown_applies_to(self) -> None:
        text = _hdr("bogus") + "\nbody"
        with pytest.raises(PromptValidationError, match="unknown applies-to"):
            parse_prompt(text)

    def test_empty_body(self) -> None:
        text = _hdr() + "\n   \n"
        with pytest.raises(PromptValidationError, match="empty body"):
            parse_prompt(text)

    def test_declared_placeholder_absent(self) -> None:
        text = _hdr(placeholders="x, missing") + "\nonly {x} here"
        with pytest.raises(PromptValidationError, match="absent from body"):
            parse_prompt(text)

    def test_body_placeholder_undeclared(self) -> None:
        text = _hdr(placeholders="x") + "\n{x} and {y}"
        with pytest.raises(PromptValidationError, match="absent from header"):
            parse_prompt(text)

    def test_malformed_header_line(self) -> None:
        text = "# version: 1.0.0\nbogus line\n\nbody"
        with pytest.raises(PromptValidationError, match="malformed header"):
            parse_prompt(text)

    def test_no_blank_line_terminator(self) -> None:
        text = "# version: 1.0.0\n# applies-to: extraction-template\n"
        with pytest.raises(PromptValidationError, match="header never terminated"):
            parse_prompt(text)


class TestSituationalRejects:
    def test_missing_system_section(self) -> None:
        text = (
            _hdr("situational", "new_content, numbered_list, existing_groups")
            + "\n## User Prompt\n{new_content} {numbered_list} {existing_groups}\n"
        )
        with pytest.raises(PromptValidationError, match="System Prompt"):
            parse_prompt(text)

    def test_missing_user_section(self) -> None:
        text = (
            _hdr("situational", "new_content, numbered_list, existing_groups")
            + "\n## System Prompt\nsys\n"
        )
        with pytest.raises(PromptValidationError, match="User Prompt"):
            parse_prompt(text)

    def test_placeholder_leak_into_system(self) -> None:
        text = (
            _hdr("situational", "new_content, numbered_list, existing_groups")
            + "\n## System Prompt\nleaks {new_content}\n\n"
            + "## User Prompt\n{new_content} {numbered_list} {existing_groups}\n"
        )
        with pytest.raises(PromptValidationError, match="user-template placeholders"):
            parse_prompt(text)

    def test_stray_brace_rejected(self) -> None:
        # Comma-list inside braces would crash .format() at runtime —
        # the word-only placeholder regex misses it; dry-run guard catches it.
        text = (
            _hdr("extraction-template", "max_concepts, max_relations")
            + "\nbody {max_concepts} {max_relations} fields: {a, b, c}\n"
        )
        with pytest.raises(PromptValidationError, match="stray brace"):
            parse_prompt(text)


class TestPackagedDefaults:
    def test_extraction_default_loads(self) -> None:
        p = load_packaged_default("extraction.default.md")
        assert p.applies_to == "extraction-template"
        assert "max_concepts" in p.placeholders
        assert "max_relations" in p.placeholders

    def test_situational_default_loads(self) -> None:
        p = load_packaged_default("situational.default.md")
        assert p.applies_to == "situational"
        expected = {"new_content", "numbered_list", "existing_groups"}
        assert set(p.placeholders) == expected
        assert p.system
        assert "{new_content}" in p.user


class TestLoadPromptFile:
    def test_reads_disk_file(self, tmp_path: Path) -> None:
        f = tmp_path / "ext.md"
        f.write_text(_hdr() + "\nbody {x}")
        p = load_prompt_file(f)
        assert p.applies_to == "extraction-template"
        assert p.source == str(f)


class TestBannedTokens:
    def test_warns_not_raises(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        text = _hdr() + "\nLet me explain: body uses {x}."
        with caplog.at_level(logging.WARNING, logger="recollect.prompts"):
            parse_prompt(text)
        assert any("banned tokens" in rec.message for rec in caplog.records)
