"""Slice 1.2a: ExtractionResult.domains closed-enum field + prompt teaching."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from recollect.llm.types import ExtractionResult
from recollect.prompts import load_packaged_default


class TestExtractionDomains:
    def test_defaults_to_empty(self) -> None:
        # The named recall tail: a structurally-successful extraction may emit [].
        assert ExtractionResult().domains == []

    def test_accepts_valid_domains(self) -> None:
        assert ExtractionResult(domains=["food", "social"]).domains == [
            "food",
            "social",
        ]

    def test_rejects_unknown_domain(self) -> None:
        # Faithful path: pydantic-ai validates the model from LLM dict output.
        with pytest.raises(ValidationError):
            ExtractionResult.model_validate({"domains": ["gastronomy"]})

    def test_model_dump_carries_domains(self) -> None:
        # 1.2b reads trace.pattern["domains"]; model_dump is the write path.
        assert ExtractionResult(domains=["food"]).model_dump()["domains"] == ["food"]


class TestExtractionPromptDomains:
    def test_version_bumped(self) -> None:
        assert load_packaged_default("extraction.default.md").version == "1.2.0"

    def test_teaches_domains_field(self) -> None:
        body = load_packaged_default("extraction.default.md").body
        assert "domains" in body
        # Exemplar surface stays distinct from eval fixtures + the 1.6 test tokens.
        assert "omakase" not in body.lower()
        assert "izakaya" not in body.lower()
