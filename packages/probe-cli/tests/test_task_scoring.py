"""Closed-form answer scoring: normalization, alias match, abstain rule."""

from __future__ import annotations

from probe_cli.corpus import TaskQuestion
from probe_cli.task_scoring import is_abstention, normalize, score_answer


def _q(answers: list[str], answer_type: str = "item") -> TaskQuestion:
    return TaskQuestion(
        id="q",
        question="?",
        answers=answers,
        answer_type=answer_type,  # type: ignore[arg-type]
        requires_trace_ids=[] if answer_type == "abstain" else ["t"],
        tier_label="none" if answer_type == "abstain" else "any",
    )


class TestNormalize:
    def test_case_punctuation_whitespace(self) -> None:
        assert normalize("  March 30th!  ") == "march 30th"
        assert normalize("seventy-five dollars") == "seventy five dollars"
        assert normalize("$75") == "75"


class TestScoreAnswer:
    def test_exact_alias(self) -> None:
        assert score_answer("Peanuts.", _q(["peanuts", "peanut"]))

    def test_alias_inside_sentence(self) -> None:
        assert score_answer(
            "The memory says Mei is allergic to peanuts.", _q(["peanuts"])
        )

    def test_word_boundary_blocks_substring(self) -> None:
        assert not score_answer("The room fits 120 people.", _q(["12"], "quantity"))

    def test_numeric_alias_with_currency(self) -> None:
        assert score_answer("It needs to be over $75.", _q(["$75", "75 dollars"]))

    def test_hyphenated_alias(self) -> None:
        assert score_answer("Seventy-five dollars total.", _q(["seventy-five dollars"]))

    def test_wrong_answer(self) -> None:
        assert not score_answer("It was walnuts.", _q(["peanuts"]))

    def test_abstention_is_not_a_correct_value(self) -> None:
        assert not score_answer("I don't know.", _q(["peanuts"]))


class TestAbstain:
    def test_markers(self) -> None:
        assert is_abstention("Unknown.")
        assert is_abstention("I don't know from the given context.")
        assert is_abstention("There is no information about that.")
        assert not is_abstention("The hotel was the Grand Palace.")

    def test_abstain_question_scoring(self) -> None:
        q = _q(["unknown"], "abstain")
        assert score_answer("unknown", q)
        assert score_answer("I do not know.", q)
        assert not score_answer("She booked the Grand Palace hotel.", q)
