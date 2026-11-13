"""Entity-reach surface-form rules: containment and embedding pairs."""

from __future__ import annotations

import math

import pytest
from probe_cli.entity_reach import EntityPair, containment_pairs, cosine_pairs


def test_containment_joins_subset_token_sets_past_stop_words() -> None:
    keys = [
        "auto-injector",
        "epinephrine auto-injector",
        "main street",
        "oak street",
        "pine street",
        "thai place on main street",
        "the",
    ]

    pairs = containment_pairs(keys)

    assert pairs == [
        EntityPair(a="auto-injector", b="epinephrine auto-injector"),
        EntityPair(a="main street", b="thai place on main street"),
    ]


def test_containment_joins_equal_token_sets() -> None:
    assert containment_pairs(["auto injector", "auto-injector"]) == [
        EntityPair(a="auto injector", b="auto-injector")
    ]


def test_containment_skips_stop_word_only_names() -> None:
    assert containment_pairs(["of the", "the community garden"]) == []


def test_cosine_pairs_honor_threshold_and_carry_similarity() -> None:
    angle = math.radians(30)
    keys = ["coffee", "espresso", "tea"]
    # Unnormalized on purpose: the rule L2-normalizes before the cosine.
    vectors = [
        [2.0, 0.0, 0.0],
        [math.cos(angle) * 5, math.sin(angle) * 5, 0.0],
        [0.0, 0.0, 3.0],
    ]

    at_85 = cosine_pairs(keys, vectors, threshold=0.85)
    at_90 = cosine_pairs(keys, vectors, threshold=0.90)

    assert [(p.a, p.b) for p in at_85] == [("coffee", "espresso")]
    assert at_85[0].similarity == pytest.approx(math.cos(angle))
    assert at_90 == []


def test_cosine_pairs_include_similarity_equal_to_threshold() -> None:
    pairs = cosine_pairs(["a b", "a c"], [[1.0, 0.0], [1.0, 0.0]], threshold=1.0)

    assert [(p.a, p.b) for p in pairs] == [("a b", "a c")]
    assert pairs[0].similarity == pytest.approx(1.0)
