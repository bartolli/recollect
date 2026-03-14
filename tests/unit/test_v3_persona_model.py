"""Tests for PersonaFact, TraceEntity, TraceConcept models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from recollect.models import PersonaFact, TraceConcept, TraceEntity


class TestPersonaFact:
    def test_creates_with_required_fields(self) -> None:
        f = PersonaFact(
            subject="Sarah",
            predicate="is_allergic_to",
            object="shellfish",
            content="Sarah is allergic to shellfish",
        )
        assert f.subject == "Sarah"
        assert f.predicate == "is_allergic_to"
        assert f.object == "shellfish"
        assert f.category == "general"
        assert f.confidence == 0.8

    def test_confidence_clamped_high(self) -> None:
        with pytest.raises(ValidationError):
            PersonaFact(
                subject="s",
                predicate="p",
                object="o",
                content="c",
                confidence=1.5,
            )

    def test_confidence_clamped_low(self) -> None:
        with pytest.raises(ValidationError):
            PersonaFact(
                subject="s",
                predicate="p",
                object="o",
                content="c",
                confidence=-0.1,
            )

    def test_category_accepts_valid_values(self) -> None:
        for cat in [
            "health",
            "dietary",
            "identity",
            "relationship",
            "preference",
            "schedule",
            "constraint",
            "general",
        ]:
            f = PersonaFact(
                subject="s",
                predicate="p",
                object="o",
                content="c",
                category=cat,
            )
            assert f.category == cat

    def test_has_generated_id(self) -> None:
        f = PersonaFact(subject="s", predicate="p", object="o", content="c")
        assert f.id  # non-empty UUID string

    def test_superseded_by_defaults_none(self) -> None:
        f = PersonaFact(subject="s", predicate="p", object="o", content="c")
        assert f.superseded_by is None


class TestTraceEntity:
    def test_creates_with_fields(self) -> None:
        e = TraceEntity(entity_name="Google", entity_type="organization", trace_id="t1")
        assert e.entity_name == "Google"
        assert e.entity_type == "organization"
        assert e.trace_id == "t1"

    def test_default_entity_type(self) -> None:
        e = TraceEntity(entity_name="X", trace_id="t1")
        assert e.entity_type == "unknown"


class TestTraceConcept:
    def test_creates_with_fields(self) -> None:
        c = TraceConcept(concept="machine learning", trace_id="t1")
        assert c.concept == "machine learning"
        assert c.trace_id == "t1"
