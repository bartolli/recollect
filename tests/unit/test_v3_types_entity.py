"""Tests for Entity, Relation sub-models and ExtractionResult fact_type."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from recollect.llm.types import Entity, ExtractionResult, Relation


class TestEntity:
    def test_creates_with_name(self) -> None:
        e = Entity(name="Google")
        assert e.name == "Google"
        assert e.entity_type == "unknown"

    def test_creates_with_type(self) -> None:
        e = Entity(name="Sarah", entity_type="person")
        assert e.entity_type == "person"

    def test_serializes_to_dict(self) -> None:
        e = Entity(name="NYC", entity_type="place")
        d = e.model_dump()
        assert d == {"name": "NYC", "entity_type": "place", "confidence": 0.8}

    def test_confidence_default(self) -> None:
        e = Entity(name="Google")
        assert e.confidence == 0.8

    def test_confidence_explicit(self) -> None:
        e = Entity(name="jaguar", entity_type="unknown", confidence=0.4)
        assert e.confidence == 0.4


class TestRelation:
    def test_creates_with_fields(self) -> None:
        r = Relation(source="Sarah", relation="works_at", target="Acme")
        assert r.source == "Sarah"
        assert r.relation == "works_at"
        assert r.target == "Acme"

    def test_serializes_to_dict(self) -> None:
        r = Relation(source="A", relation="is_associated_with", target="B")
        d = r.model_dump()
        assert d == {
            "source": "A",
            "relation": "is_associated_with",
            "target": "B",
            "confidence": 0.8,
            "context": "",
            "context_tags": [],
            "category": "general",
        }

    def test_confidence_default(self) -> None:
        r = Relation(source="A", relation="is_associated_with", target="B")
        assert r.confidence == 0.8

    def test_confidence_explicit(self) -> None:
        r = Relation(source="A", relation="is_phobic_of", target="B", confidence=0.95)
        assert r.confidence == 0.95


class TestExtractionResultWithSubModels:
    def test_entities_are_entity_objects(self) -> None:
        result = ExtractionResult(
            entities=[Entity(name="Google", entity_type="organization")]
        )
        assert len(result.entities) == 1
        assert result.entities[0].name == "Google"
        assert result.entities[0].entity_type == "organization"

    def test_relations_are_relation_objects(self) -> None:
        result = ExtractionResult(
            relations=[Relation(source="A", relation="is_associated_with", target="B")]
        )
        assert len(result.relations) == 1
        assert result.relations[0].relation == "is_associated_with"

    def test_fact_type_defaults_to_episodic(self) -> None:
        result = ExtractionResult()
        assert result.fact_type == "episodic"

    def test_fact_type_can_be_semantic(self) -> None:
        result = ExtractionResult(fact_type="semantic")
        assert result.fact_type == "semantic"

    def test_model_dump_includes_fact_type(self) -> None:
        result = ExtractionResult()
        d = result.model_dump()
        assert d["fact_type"] == "episodic"
        assert d["entities"] == []
        assert d["relations"] == []


class TestSchemaClosure:
    """P1: Predicate, EntityType, FactCategory enforced as Literal enums.

    Invalid values are passed via model_validate(dict) so the type-checker
    does not narrow the Literal at the call site; runtime validation is
    what we are exercising.
    """

    def test_predicate_accepts_known(self) -> None:
        Relation(source="a", relation="is_allergic_to", target="b")
        Relation(source="a", relation="works_at", target="b")
        Relation(source="a", relation="is_associated_with", target="b")

    def test_predicate_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            Relation.model_validate(
                {"source": "a", "relation": "bogus_predicate", "target": "b"}
            )

    def test_entity_type_accepts_known(self) -> None:
        Entity(name="x", entity_type="person")
        Entity(name="x", entity_type="condition")
        Entity(name="x", entity_type="unknown")

    def test_entity_type_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            Entity.model_validate({"name": "x", "entity_type": "animal"})

    def test_category_accepts_fact_category(self) -> None:
        Relation(source="a", relation="prefers", target="b", category="health")
        Relation(source="a", relation="prefers", target="b", category="general")

    def test_category_rejects_non_fact_category(self) -> None:
        with pytest.raises(ValidationError):
            Relation.model_validate(
                {
                    "source": "a",
                    "relation": "prefers",
                    "target": "b",
                    "category": "bogus_category",
                }
            )
