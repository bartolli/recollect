"""Slice 1.2a: ExtractionResult.domains closed-enum field + prompt teaching."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from recollect.core import _DOMAIN_SAFETY_MAP, _FAST_TRACK_CATEGORIES
from recollect.llm.types import ExtractionResult
from recollect.prompts import load_packaged_default


class TestExtractionDomains:
    def test_defaults_to_empty(self) -> None:
        # The named recall tail: a structurally-successful extraction may emit [].
        assert ExtractionResult().domains == []

    def test_accepts_valid_domains(self) -> None:
        assert ExtractionResult(domains=["food", "medical", "social"]).domains == [
            "food",
            "medical",
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
        assert load_packaged_default("extraction.default.md").version == "1.2.1"

    def test_teaches_domains_field(self) -> None:
        body = load_packaged_default("extraction.default.md").body
        assert "domains" in body
        assert "medical" in body
        # Exemplar surface stays distinct from eval fixtures + the 1.6 test tokens.
        assert "omakase" not in body.lower()
        assert "izakaya" not in body.lower()


class TestDomainSafetyMap:
    def test_medical_surfaces_all_safety_categories(self) -> None:
        assert _DOMAIN_SAFETY_MAP["medical"] == _FAST_TRACK_CATEGORIES

    def test_non_safety_domains_absent(self) -> None:
        # Drains map to nothing -- inert at the gate by design.
        for d in ("social", "finance", "general"):
            assert d not in _DOMAIN_SAFETY_MAP
