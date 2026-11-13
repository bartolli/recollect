"""Entity-reach graph: question entities, union reach, label join."""

from __future__ import annotations

from probe_cli.corpus import EntityPairLabel, TaskQuestion
from probe_cli.entity_reach import (
    EntityPair,
    join_labels,
    question_entities,
    reach,
    representatives,
)

_THAI = "thai place on main street"
_THAI_ALT = "thai restaurant on main street"
_MENTIONS = {
    "head": [_THAI, "group dinner"],
    "mid": [_THAI_ALT, "reservation"],
    "tail": ["deposit", "reservation"],
    "other": ["bookstore", "deposit"],
}


def _question(text: str, required: str) -> TaskQuestion:
    return TaskQuestion(
        id="t3q01",
        question=text,
        answers=["x"],
        answer_type="item",
        requires_trace_ids=[required],
        tier_label="t3",
    )


_DINNER = _question(
    "What do we pay before the group dinner at the Thai place on Main Street?", "tail"
)


def _keys(mentions: dict[str, list[str]]) -> list[str]:
    return sorted({key for keys in mentions.values() for key in keys})


def test_question_entities_respect_token_boundaries_and_length_floor() -> None:
    keys = ["ai", "air", "hardware store", "oak street hard", "oak street hardware"]
    text = "Is the AI at Oak Street Hardware open for the street fair?"

    assert question_entities(text, [*keys, "street fair"]) == [
        "oak street hardware",
        "street fair",
    ]


def test_exact_rule_leaves_the_tail_unreachable() -> None:
    rep = representatives(_keys(_MENTIONS), [])

    result = reach(_DINNER, mentions=_MENTIONS, representative=rep)

    assert result.seed_entities == ["group dinner", _THAI]
    assert (result.hop, result.candidates) == (None, None)
    assert result.required == "tail"


def test_joined_names_reach_the_tail_at_hop_one() -> None:
    rep = representatives(_keys(_MENTIONS), [EntityPair(a=_THAI, b=_THAI_ALT)])

    result = reach(_DINNER, mentions=_MENTIONS, representative=rep)

    assert (result.hop, result.candidates) == (1, 3)


def test_union_is_transitive_with_a_deterministic_representative() -> None:
    rep = representatives(
        ["a", "b", "c", "d"], [EntityPair(a="b", b="c"), EntityPair(a="a", b="c")]
    )

    assert rep == {"a": "a", "b": "a", "c": "a", "d": "d"}


def test_walk_stops_after_six_hops() -> None:
    chain = {f"t{i}": [f"entity {i}", f"entity {i + 1}"] for i in range(8)}
    rep = representatives(_keys(chain), [])

    at_six = reach(
        _question("About entity 0?", "t6"), mentions=chain, representative=rep
    )
    at_seven = reach(
        _question("About entity 0?", "t7"), mentions=chain, representative=rep
    )

    assert (at_six.hop, at_six.candidates) == (6, 7)
    assert (at_seven.hop, at_seven.candidates) == (None, None)


def test_question_without_entities_reaches_nothing() -> None:
    rep = representatives(_keys(_MENTIONS), [])

    result = reach(
        _question("Anything?", "tail"), mentions=_MENTIONS, representative=rep
    )

    assert result.seed_entities == []
    assert (result.hop, result.candidates) == (None, None)


def test_label_join_counts_by_label_and_lists_unlabeled_pairs() -> None:
    labels = [
        EntityPairLabel(a="main street", b=_THAI, label="related"),
        EntityPairLabel(a=_THAI_ALT, b=_THAI, label="same"),
        EntityPairLabel(a="lila", b="margot", label="different"),
    ]
    pairs = [
        EntityPair(a="main street", b=_THAI),
        EntityPair(a=_THAI, b=_THAI_ALT),
        EntityPair(a="oak street", b="pine street", similarity=0.91),
    ]

    counts, unlabeled = join_labels(pairs, labels)

    assert counts == {"same": 1, "related": 1, "different": 0, "unlabeled": 1}
    assert unlabeled == [EntityPair(a="oak street", b="pine street", similarity=0.91)]
