"""Recency-anchored telescoping decay -- window selection, grace anchor.

recency_anchor = max(last_retrieval, last_activation, created_at);
apply_time_decay windows from max(recency_anchor, last_decayed_at) so
per-pass factors telescope; the consolidation grace check reads the
recency anchor only -- a decay stamp must never reset the forget clock.
"""

from __future__ import annotations

import math
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.datetime_utils import memory_timestamp_for_comparison, now_utc
from recollect.models import MemoryTrace, apply_time_decay, recency_anchor

RATE = 0.1  # MemoryTrace.decay_rate field default


@pytest.fixture()
def mem(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
    mock_extractor: AsyncMock,
) -> CognitiveMemory:
    return CognitiveMemory(
        storage=mock_storage,
        embeddings=mock_embeddings,
        extractor=mock_extractor,
    )


class TestRecencyAnchor:
    def test_falls_back_to_created_at(self) -> None:
        created = now_utc() - timedelta(hours=10)
        t = MemoryTrace(content="t", created_at=created)
        assert recency_anchor(t) == memory_timestamp_for_comparison(created)

    def test_picks_latest_touch(self) -> None:
        t = MemoryTrace(
            content="t",
            created_at=now_utc() - timedelta(hours=100),
            last_activation=now_utc() - timedelta(hours=50),
            last_retrieval=now_utc() - timedelta(hours=1),
        )
        now = memory_timestamp_for_comparison(now_utc())
        age_hours = (now - recency_anchor(t)).total_seconds() / 3600.0
        assert age_hours == pytest.approx(1.0, abs=0.01)

    def test_tz_mix_normalizes_without_typeerror(self) -> None:
        naive = (now_utc() - timedelta(hours=2)).replace(tzinfo=None)
        t = MemoryTrace(
            content="t",
            created_at=now_utc() - timedelta(hours=9),
            last_retrieval=naive,
        )
        assert recency_anchor(t).tzinfo is None


class TestTelescopingWindow:
    def test_windows_from_last_touch_not_creation(self) -> None:
        t = MemoryTrace(
            content="t",
            strength=0.8,
            created_at=now_utc() - timedelta(hours=100),
            last_retrieval=now_utc() - timedelta(hours=1),
        )
        decayed = apply_time_decay(t)
        assert decayed.strength == pytest.approx(
            0.8 * math.exp(-RATE * 1.0), rel=1e-3
        )

    def test_stamp_clips_window(self) -> None:
        t = MemoryTrace(
            content="t",
            strength=0.8,
            created_at=now_utc() - timedelta(hours=10),
            last_decayed_at=now_utc() - timedelta(hours=4),
        )
        decayed = apply_time_decay(t)
        assert decayed.strength == pytest.approx(
            0.8 * math.exp(-RATE * 4.0), rel=1e-3
        )

    def test_two_passes_compose_to_single_pass(self) -> None:
        # exp(-r*d1) * exp(-r*d2) == exp(-r*(d1+d2)): a pass stamped at
        # -4h then a pass now must equal one unstamped full-window pass.
        created = now_utc() - timedelta(hours=10)
        stamp = now_utc() - timedelta(hours=4)
        one_pass = apply_time_decay(
            MemoryTrace(content="t", strength=0.8, created_at=created)
        )
        after_first = 0.8 * math.exp(-RATE * 6.0)
        second = apply_time_decay(
            MemoryTrace(
                content="t",
                strength=after_first,
                created_at=created,
                last_decayed_at=stamp,
            )
        )
        assert second.strength == pytest.approx(one_pass.strength, rel=1e-3)


class TestGraceAnchor:
    async def test_reactivated_weak_trace_survives(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        trace = MemoryTrace(
            content="revived",
            strength=0.2,
            created_at=now_utc() - timedelta(hours=48),
            last_activation=now_utc(),
        )
        mock_trace_store.get_unconsolidated_traces.return_value = [trace]
        result = await mem.consolidate()
        assert result.still_pending == 1
        mock_trace_store.delete_trace.assert_not_awaited()
        mock_trace_store.apply_decay_factor.assert_awaited_once()

    async def test_touched_then_idle_forgets_from_last_touch(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        trace = MemoryTrace(
            content="idle",
            strength=0.05,
            created_at=now_utc() - timedelta(hours=200),
            last_retrieval=now_utc() - timedelta(hours=50),
        )
        mock_trace_store.get_unconsolidated_traces.return_value = [trace]
        result = await mem.consolidate()
        assert result.forgotten == 1
        mock_trace_store.delete_trace.assert_awaited_once()

    async def test_decay_stamp_does_not_reset_grace(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        # Immortality regression: a fresh stamp clips the decay window
        # but must not rescue a past-grace weak trace.
        trace = MemoryTrace(
            content="stamped",
            strength=0.05,
            created_at=now_utc() - timedelta(hours=48),
            last_decayed_at=now_utc(),
        )
        mock_trace_store.get_unconsolidated_traces.return_value = [trace]
        result = await mem.consolidate()
        assert result.forgotten == 1

    async def test_consolidation_writes_through_decay_write(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        strong = MemoryTrace(content="strong", strength=0.8)
        mock_trace_store.get_unconsolidated_traces.return_value = [strong]
        await mem.consolidate()
        mock_trace_store.apply_decay_factor.assert_awaited_once()
        mock_trace_store.apply_strength_factor.assert_not_awaited()
