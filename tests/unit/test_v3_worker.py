"""Tests for ConsolidationWorker."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from recollect.models import ConsolidationResult
from recollect.worker import ConsolidationWorker


@pytest.fixture
def mock_memory() -> AsyncMock:
    memory = AsyncMock()
    memory.consolidate.return_value = ConsolidationResult(
        consolidated=2, forgotten=1, still_pending=3
    )
    return memory


def make_worker(memory: AsyncMock, interval: float = 0.05) -> ConsolidationWorker:
    return ConsolidationWorker(memory, interval_seconds=interval)


async def test_start_stop_lifecycle(mock_memory: AsyncMock) -> None:
    worker = make_worker(mock_memory)
    assert not worker.running

    worker.start()
    assert worker.running

    await asyncio.sleep(0.01)
    worker.stop()
    assert not worker.running


async def test_calls_consolidate_on_tick(mock_memory: AsyncMock) -> None:
    worker = make_worker(mock_memory, interval=0.05)
    worker.start()
    await asyncio.sleep(0.15)
    worker.stop()

    assert mock_memory.consolidate.call_count >= 2


async def test_handles_errors_without_crashing(
    mock_memory: AsyncMock,
) -> None:
    mock_memory.consolidate.side_effect = [
        RuntimeError("db down"),
        ConsolidationResult(consolidated=1, forgotten=0, still_pending=0),
    ]
    worker = make_worker(mock_memory, interval=0.05)
    worker.start()
    await asyncio.sleep(0.15)
    worker.stop()

    assert mock_memory.consolidate.call_count >= 2
    assert worker.running is False


async def test_stop_is_idempotent(mock_memory: AsyncMock) -> None:
    worker = make_worker(mock_memory)
    worker.stop()
    worker.stop()
    assert not worker.running


async def test_start_twice_is_safe(mock_memory: AsyncMock) -> None:
    worker = make_worker(mock_memory, interval=0.05)
    worker.start()
    worker.start()
    await asyncio.sleep(0.01)
    worker.stop()
    assert not worker.running


async def test_respects_interval(mock_memory: AsyncMock) -> None:
    worker = make_worker(mock_memory, interval=0.1)
    worker.start()
    await asyncio.sleep(0.05)
    worker.stop()

    # Should not have called yet -- interval not reached
    assert mock_memory.consolidate.call_count == 0


async def test_logs_results(mock_memory: AsyncMock) -> None:
    worker = make_worker(mock_memory, interval=0.05)
    with patch("recollect.worker.logger") as mock_logger:
        worker.start()
        await asyncio.sleep(0.1)
        worker.stop()

        assert mock_logger.info.called


async def test_logs_errors(mock_memory: AsyncMock) -> None:
    mock_memory.consolidate.side_effect = RuntimeError("boom")
    worker = make_worker(mock_memory, interval=0.05)
    with patch("recollect.worker.logger") as mock_logger:
        worker.start()
        await asyncio.sleep(0.1)
        worker.stop()

        assert mock_logger.exception.called
