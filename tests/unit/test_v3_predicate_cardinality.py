"""Persona-fact supersession keys on predicate cardinality.

adr-persona-fact-cardinality: SET predicates coexist (a different object is an
addition, not a contradiction); CURRENT/FUNCTIONAL supersede on a different
object. Every Predicate member must carry a cardinality class.
"""

from __future__ import annotations

from typing import get_args

from recollect.core import _find_contradicting_fact
from recollect.llm.types import PREDICATE_CARDINALITY, Predicate
from recollect.models import PersonaFact


def _fact(predicate: str, obj: str) -> PersonaFact:
    return PersonaFact(subject="Angel", predicate=predicate, object=obj, content="c")


class TestCardinalityGate:
    def test_set_safety_predicate_does_not_contradict(self) -> None:
        # The headline: a second allergy must not supersede the first.
        existing = [_fact("is_allergic_to", "penicillin")]
        new = _fact("is_allergic_to", "shellfish")
        assert _find_contradicting_fact(existing, new) is None

    def test_set_relationship_coexists(self) -> None:
        existing = [_fact("is_related_to", "Tania (mother)")]
        new = _fact("is_related_to", "Dimitar (brother)")
        assert _find_contradicting_fact(existing, new) is None

    def test_current_predicate_contradicts(self) -> None:
        existing = [_fact("lives_in", "Lisbon")]
        new = _fact("lives_in", "Berlin")
        assert _find_contradicting_fact(existing, new) is existing[0]

    def test_functional_predicate_contradicts(self) -> None:
        existing = [_fact("originates_from", "Bulgaria")]
        new = _fact("originates_from", "Greece")
        assert _find_contradicting_fact(existing, new) is existing[0]


class TestCardinalityClassification:
    def test_every_predicate_classified(self) -> None:
        for predicate in get_args(Predicate):
            assert predicate in PREDICATE_CARDINALITY, f"{predicate} unclassified"

    def test_classes_are_known(self) -> None:
        assert set(PREDICATE_CARDINALITY.values()) <= {"set", "current", "functional"}

    def test_current_and_functional_members(self) -> None:
        current = {p for p, c in PREDICATE_CARDINALITY.items() if c == "current"}
        functional = {p for p, c in PREDICATE_CARDINALITY.items() if c == "functional"}
        assert current == {"lives_in", "is_partner_of"}
        assert functional == {"originates_from"}
