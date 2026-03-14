"""Unit tests for session summarization."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import SessionNotFoundError
from recollect.models import MemoryTrace, Session


@pytest.fixture()
def memory(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
) -> CognitiveMemory:
    return CognitiveMemory(
        storage=mock_storage,
        embeddings=mock_embeddings,
    )


def _make_trace(content: str, session_id: str = "s1") -> MemoryTrace:
    return MemoryTrace(
        content=content,
        session_id=session_id,
        user_id="u1",
    )


class TestSummarizeSession:
    @pytest.mark.asyncio()
    async def test_generates_summary_trace(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        session = Session(id="s1", user_id="u1", title="Test Session")
        mock_storage.sessions.get_session.return_value = session
        mock_storage.traces.get_traces_by_session.return_value = [
            _make_trace("First memory"),
            _make_trace("Second memory"),
        ]
        result = await memory.summarize_session("s1")
        assert result.session_id == "s1"
        assert result.user_id == "u1"
        assert result.pattern.get("session_summary") is True
        assert result.pattern.get("source_session_id") == "s1"
        assert result.pattern.get("trace_count") == 2
        mock_storage.traces.store_trace.assert_awaited()

    @pytest.mark.asyncio()
    async def test_summary_has_high_strength(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        session = Session(id="s1", user_id="u1")
        mock_storage.sessions.get_session.return_value = session
        mock_storage.traces.get_traces_by_session.return_value = [
            _make_trace("memory"),
        ]
        result = await memory.summarize_session("s1")
        assert result.strength == pytest.approx(0.8)
        assert result.significance == pytest.approx(0.7)
        assert result.decay_rate == pytest.approx(0.02)

    @pytest.mark.asyncio()
    async def test_raises_for_missing_session(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        mock_storage.sessions.get_session.return_value = None
        with pytest.raises(SessionNotFoundError):
            await memory.summarize_session("missing")

    @pytest.mark.asyncio()
    async def test_raises_for_empty_session(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        session = Session(id="s1", user_id="u1")
        mock_storage.sessions.get_session.return_value = session
        mock_storage.traces.get_traces_by_session.return_value = []
        with pytest.raises(ValueError, match="has no traces"):
            await memory.summarize_session("s1")

    @pytest.mark.asyncio()
    async def test_fallback_without_extractor(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        session = Session(id="s1", user_id="u1")
        mock_storage.sessions.get_session.return_value = session
        mock_storage.traces.get_traces_by_session.return_value = [
            _make_trace("Alpha"),
            _make_trace("Beta"),
        ]
        result = await memory.summarize_session("s1")
        assert "Alpha" in result.content
        assert "Beta" in result.content
        assert "Session (2 memories)" in result.content

    @pytest.mark.asyncio()
    async def test_updates_session_status(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        session = Session(id="s1", user_id="u1")
        mock_storage.sessions.get_session.return_value = session
        mock_storage.traces.get_traces_by_session.return_value = [
            _make_trace("memory"),
        ]
        result = await memory.summarize_session("s1")
        mock_storage.sessions.update_session.assert_awaited_once_with(
            "s1",
            status="summarized",
            summary_trace_id=result.id,
        )

    @pytest.mark.asyncio()
    async def test_embeds_concepts(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        session = Session(id="s1", user_id="u1")
        mock_storage.sessions.get_session.return_value = session
        mock_storage.traces.get_traces_by_session.return_value = [
            _make_trace("memory"),
        ]
        await memory.summarize_session("s1")
        # Extraction links and concept embeddings are gathered
        mock_storage.entities.store_trace_entities.assert_awaited()
