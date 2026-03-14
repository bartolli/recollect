"""Tests for strength/decay math functions.

These are the core cognitive model calculations -- getting them wrong
means the memory system behaves incorrectly.
"""

from datetime import timedelta

import pytest
from recollect.datetime_utils import now_utc
from recollect.models import (
    MemoryTrace,
    apply_activation_boost,
    apply_displacement_decay,
    apply_retrieval_boost,
    apply_time_decay,
)


@pytest.fixture
def trace() -> MemoryTrace:
    return MemoryTrace(content="test", pattern={"concepts": ["test"]})


class TestActivationBoost:
    def test_applies_1_percent_boost(self, trace: MemoryTrace) -> None:
        # Default initial strength is 0.3
        boosted = apply_activation_boost(trace)
        assert boosted.strength == pytest.approx(0.3 * 1.01, rel=1e-6)

    def test_clamped_at_max(self) -> None:
        trace = MemoryTrace(content="t", pattern={}, strength=0.999)
        boosted = apply_activation_boost(trace)
        assert boosted.strength <= 1.0


class TestRetrievalBoost:
    def test_applies_10_percent_boost(self, trace: MemoryTrace) -> None:
        boosted = apply_retrieval_boost(trace)
        assert boosted.strength == pytest.approx(0.3 * 1.1, rel=1e-6)

    def test_working_memory_boost_is_stronger(self, trace: MemoryTrace) -> None:
        wm_boosted = apply_retrieval_boost(trace, from_working_memory=True)
        regular_boosted = apply_retrieval_boost(trace, from_working_memory=False)
        assert wm_boosted.strength > regular_boosted.strength

    def test_wm_applies_20_percent_boost(self, trace: MemoryTrace) -> None:
        boosted = apply_retrieval_boost(trace, from_working_memory=True)
        assert boosted.strength == pytest.approx(0.3 * 1.2, rel=1e-6)

    def test_clamped_at_max(self) -> None:
        trace = MemoryTrace(content="t", pattern={}, strength=0.95)
        boosted = apply_retrieval_boost(trace)
        assert boosted.strength <= 1.0


class TestDisplacementDecay:
    def test_applies_20_percent_decay(self, trace: MemoryTrace) -> None:
        decayed = apply_displacement_decay(trace)
        assert decayed.strength == pytest.approx(0.3 * 0.8, rel=1e-6)

    def test_never_goes_negative(self) -> None:
        trace = MemoryTrace(content="t", pattern={}, strength=0.01)
        decayed = apply_displacement_decay(trace)
        assert decayed.strength >= 0.0


class TestTimeDecay:
    def test_no_decay_when_fresh(self, trace: MemoryTrace) -> None:
        decayed = apply_time_decay(trace)
        # Just created, virtually no time passed
        assert decayed.strength == pytest.approx(0.3, abs=0.01)

    def test_decays_over_time(self) -> None:
        old_time = now_utc() - timedelta(hours=7)
        trace = MemoryTrace(content="t", pattern={}, strength=0.5, created_at=old_time)
        decayed = apply_time_decay(trace)
        # After ~7 hours with rate 0.1: exp(-0.1 * 7) ~ 0.497
        # So 0.5 * 0.497 ~ 0.248
        assert decayed.strength < 0.3
        assert decayed.strength > 0.1

    def test_never_goes_negative(self) -> None:
        ancient = now_utc() - timedelta(days=365)
        trace = MemoryTrace(content="t", pattern={}, strength=0.5, created_at=ancient)
        decayed = apply_time_decay(trace)
        assert decayed.strength >= 0.0


class TestConfidenceLabel:
    """MemoryTrace.confidence should return a human-readable label."""

    def test_vivid_above_0_8(self) -> None:
        t = MemoryTrace(content="t", pattern={}, strength=0.9)
        assert t.confidence == "vivid"

    def test_clear_between_0_5_and_0_8(self) -> None:
        t = MemoryTrace(content="t", pattern={}, strength=0.6)
        assert t.confidence == "clear"

    def test_fuzzy_between_0_2_and_0_5(self) -> None:
        t = MemoryTrace(content="t", pattern={}, strength=0.3)
        assert t.confidence == "fuzzy"

    def test_fading_below_0_2(self) -> None:
        t = MemoryTrace(content="t", pattern={}, strength=0.1)
        assert t.confidence == "fading"
