"""Unit tests for session support in CognitiveMemory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.core import CognitiveMemory
from recollect.exceptions import SessionNotFoundError
from recollect.models import Session


@pytest.fixture()
def memory(
    mock_storage: MagicMock,
    mock_embeddings: AsyncMock,
) -> CognitiveMemory:
    return CognitiveMemory(
        storage=mock_storage,
        embeddings=mock_embeddings,
    )


class TestStartSession:
    @pytest.mark.asyncio()
    async def test_creates_session_via_store(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        session = await memory.start_session(user_id="u1", title="Test")
        assert session.user_id == "u1"
        assert session.title == "Test"
        assert session.status == "active"
        mock_storage.sessions.create_session.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_auto_generates_id(self, memory: CognitiveMemory) -> None:
        session = await memory.start_session(user_id="u1")
        assert len(session.id) == 36  # UUID format

    @pytest.mark.asyncio()
    async def test_custom_id(self, memory: CognitiveMemory) -> None:
        session = await memory.start_session(
            user_id="u1",
            session_id="custom-id",
        )
        assert session.id == "custom-id"


class TestEndSession:
    @pytest.mark.asyncio()
    async def test_ends_existing_session(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        existing = Session(id="s1", user_id="u1")
        mock_storage.sessions.get_session.return_value = existing
        result = await memory.end_session("s1")
        assert result.status == "ended"
        assert result.ended_at is not None
        mock_storage.sessions.end_session.assert_awaited_once_with("s1")

    @pytest.mark.asyncio()
    async def test_raises_for_missing(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        mock_storage.sessions.get_session.return_value = None
        with pytest.raises(SessionNotFoundError):
            await memory.end_session("missing")


class TestExperienceWithSession:
    @pytest.mark.asyncio()
    async def test_passes_session_and_user_to_trace(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        mock_storage.sessions.get_session.return_value = Session(
            id="s1",
            user_id="u1",
        )
        trace = await memory.experience(
            "test content",
            session_id="s1",
            user_id="u1",
        )
        assert trace.session_id == "s1"
        assert trace.user_id == "u1"

    @pytest.mark.asyncio()
    async def test_auto_creates_session(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        mock_storage.sessions.get_session.return_value = None
        await memory.experience(
            "test content",
            session_id="new-s",
            user_id="u1",
        )
        mock_storage.sessions.create_session.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_raises_without_user_on_new_session(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        mock_storage.sessions.get_session.return_value = None
        with pytest.raises(ValueError, match="user_id required"):
            await memory.experience("test", session_id="new-s")

    @pytest.mark.asyncio()
    async def test_existing_session_no_create(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        mock_storage.sessions.get_session.return_value = Session(
            id="s1",
            user_id="u1",
        )
        await memory.experience(
            "content",
            session_id="s1",
            user_id="u1",
        )
        mock_storage.sessions.create_session.assert_not_awaited()


class TestThinkAboutWithSession:
    @pytest.mark.asyncio()
    async def test_passes_session_to_search(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        await memory.think_about("query", session_id="s1")
        call_kwargs = mock_storage.vectors.search_semantic.call_args
        assert call_kwargs.kwargs.get("session_id") == "s1"

    @pytest.mark.asyncio()
    async def test_passes_user_to_search(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        await memory.think_about("query", user_id="u1")
        call_kwargs = mock_storage.vectors.search_semantic.call_args
        assert call_kwargs.kwargs.get("user_id") == "u1"

    @pytest.mark.asyncio()
    async def test_none_searches_all(
        self,
        memory: CognitiveMemory,
        mock_storage: MagicMock,
    ) -> None:
        await memory.think_about("query")
        call_kwargs = mock_storage.vectors.search_semantic.call_args
        assert call_kwargs.kwargs.get("session_id") is None
        assert call_kwargs.kwargs.get("user_id") is None
