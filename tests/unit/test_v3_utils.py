"""Tests for core utility functions."""

import pytest
from recollect.core import _cosine_similarity, _estimate_tokens


class TestCosine:
    def test_identical(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0, 0], [1, 1]) == pytest.approx(0.0)


class TestEstimateTokens:
    def test_normal_text(self) -> None:
        assert _estimate_tokens("hello world!") == 3

    def test_empty_string(self) -> None:
        assert _estimate_tokens("") == 1
