"""Tests for CognitiveMemory: forget, reinforce, introspection, validation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import TraceNotFoundError
from recollect.models import HealthStatus, MemoryStats, MemoryTrace


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


class TestForgetAndReinforce:
    async def test_forget_archives_not_deletes(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        assert await mem.forget("some-id") is True
        mock_trace_store.archive_trace.assert_awaited_once_with("some-id")
        mock_trace_store.delete_trace.assert_not_awaited()

    async def test_forget_raises_not_found(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        mock_trace_store.archive_trace.return_value = False
        with pytest.raises(TraceNotFoundError):
            await mem.forget("nonexistent")

    async def test_erase_hard_deletes_and_evicts(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        trace = MemoryTrace(id="erase-1", content="x")
        mem._buffer.add(trace)
        assert await mem.erase("erase-1") is True
        mock_trace_store.delete_trace.assert_awaited_once_with("erase-1")
        mock_trace_store.archive_trace.assert_not_awaited()
        assert all(t.id != "erase-1" for t in mem._buffer.get_active())

    async def test_erase_missing_returns_false(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        mock_trace_store.delete_trace.return_value = False
        assert await mem.erase("nonexistent") is False

    async def test_erase_rejects_empty_id(self, mem: CognitiveMemory) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await mem.erase("")

    async def test_forget_evicts_buffered_trace(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        trace = MemoryTrace(id="buf-1", content="x")
        mem._buffer.add(trace)
        assert any(t.id == "buf-1" for t in mem._buffer.get_active())
        await mem.forget("buf-1")
        assert all(t.id != "buf-1" for t in mem._buffer.get_active())

    async def test_reinforce_increases_strength(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        trace = MemoryTrace(content="Test", strength=0.5)
        mock_trace_store.get_trace.return_value = trace
        result = await mem.reinforce(trace.id, factor=1.5)
        assert result.strength == pytest.approx(0.75)

    async def test_reinforce_raises_not_found(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        with pytest.raises(TraceNotFoundError):
            await mem.reinforce("nonexistent")


class TestIntrospection:
    async def test_timeline_delegates(
        self, mem: CognitiveMemory, mock_trace_store: AsyncMock
    ) -> None:
        await mem.timeline(limit=10)
        mock_trace_store.get_recent_traces.assert_awaited_once_with(10)

    def test_stats(self, mem: CognitiveMemory) -> None:
        s = mem.stats()
        assert isinstance(s, MemoryStats)
        assert s.working_memory_items >= 0
        assert s.connected is False

    def test_health_disconnected(self, mem: CognitiveMemory) -> None:
        h = mem.health()
        assert isinstance(h, HealthStatus)
        assert h.status == "disconnected"


class TestValidation:
    async def test_experience_rejects_empty(self, mem: CognitiveMemory) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await mem.experience("")

    async def test_experience_rejects_whitespace(self, mem: CognitiveMemory) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await mem.experience("   ")

    async def test_think_about_rejects_empty(self, mem: CognitiveMemory) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await mem.think_about("")

    async def test_think_about_rejects_zero_budget(self, mem: CognitiveMemory) -> None:
        with pytest.raises(ValueError, match="positive"):
            await mem.think_about("test", token_budget=0)

    async def test_forget_rejects_empty_id(self, mem: CognitiveMemory) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await mem.forget("")

    async def test_reinforce_rejects_empty_id(self, mem: CognitiveMemory) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await mem.reinforce("", factor=1.5)

    async def test_reinforce_rejects_zero_factor(self, mem: CognitiveMemory) -> None:
        with pytest.raises(ValueError, match="positive"):
            await mem.reinforce("some-id", factor=0.0)

    async def test_timeline_rejects_zero_limit(self, mem: CognitiveMemory) -> None:
        with pytest.raises(ValueError, match="positive"):
            await mem.timeline(limit=0)


class TestContextManager:
    async def test_async_context_manager(
        self, mock_storage: MagicMock, mock_embeddings: AsyncMock
    ) -> None:
        mem = CognitiveMemory(storage=mock_storage, embeddings=mock_embeddings)
        async with mem:
            mock_storage.initialize.assert_awaited_once()
        mock_storage.close.assert_awaited_once()
