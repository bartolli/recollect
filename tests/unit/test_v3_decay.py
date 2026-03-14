"""Tests for significance-weighted decay rate computation."""

from __future__ import annotations

import pytest
from recollect.core import _compute_decay_rate


class TestComputeDecayRate:
    def test_base_rate_with_zero_significance(self) -> None:
        rate = _compute_decay_rate(0.1, 0.0, 0.0)
        assert rate == pytest.approx(0.1)

    def test_high_significance_reduces_decay(self) -> None:
        rate = _compute_decay_rate(0.1, 1.0, 0.0, sig_reduction=0.7)
        # 0.1 * (1 - 1.0*0.7) * (1 - 0*0.5) = 0.1 * 0.3 * 1.0 = 0.03
        assert rate == pytest.approx(0.03)

    def test_high_valence_reduces_decay(self) -> None:
        rate = _compute_decay_rate(0.1, 0.0, 1.0, val_reduction=0.5)
        # 0.1 * (1 - 0*0.7) * (1 - 1.0*0.5) = 0.1 * 1.0 * 0.5 = 0.05
        assert rate == pytest.approx(0.05)

    def test_negative_valence_same_as_positive(self) -> None:
        pos = _compute_decay_rate(0.1, 0.0, 0.8)
        neg = _compute_decay_rate(0.1, 0.0, -0.8)
        assert pos == pytest.approx(neg)

    def test_both_significance_and_valence(self) -> None:
        rate = _compute_decay_rate(0.1, 1.0, 1.0, sig_reduction=0.7, val_reduction=0.5)
        # 0.1 * 0.3 * 0.5 = 0.015
        assert rate == pytest.approx(0.015)

    def test_moderate_values(self) -> None:
        rate = _compute_decay_rate(0.1, 0.5, 0.5)
        # 0.1 * (1 - 0.5*0.7) * (1 - 0.5*0.5) = 0.1 * 0.65 * 0.75 = 0.04875
        assert rate == pytest.approx(0.04875)

    def test_zero_base_rate(self) -> None:
        rate = _compute_decay_rate(0.0, 1.0, 1.0)
        assert rate == pytest.approx(0.0)
