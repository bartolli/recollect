"""Tests for MCP server tools and resources."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.exceptions import (
    ExtractionError,
    MemorySDKError,
    StorageError,
    TraceNotFoundError,
)
from recollect.models import (
    ForgetResult,
    HealthStatus,
    MemoryTrace,
    PersonaFact,
    Session,
    Thought,
)
from recollect_mcp.server import (
    AppContext,
    _ensure_session,
    _format_facts,
    _generate_primer,
    forget,
    pin,
    recall,
    remember,
    unpin,
)


@pytest.fixture
def mock_memory() -> AsyncMock:
    memory = AsyncMock()
    memory.experience.return_value = MemoryTrace(content="test")
    memory.surface_relevant_facts.return_value = []
    memory.think_about.return_value = [
        Thought(
            trace=MemoryTrace(content="recalled"),
            relevance=0.9,
            token_count=10,
            reconstruction="recalled",
        )
    ]
    memory.forget.return_value = ForgetResult(trace_id="trace-123")
    memory.pin.return_value = [
        PersonaFact(
            subject="Alex",
            predicate="likes",
            object="coffee",
            content="Alex likes coffee",
            status="pinned",
        )
    ]
    memory.unpin.return_value = True
    memory.facts.return_value = [
        PersonaFact(
            subject="Alex",
            predicate="mother_of",
            object="Sarah",
            content="Alex's mother is Sarah who lives in Portland",
            status="pinned",
        )
    ]
    memory.start_session.return_value = Session(id="sess-123", user_id="alex")
    memory.health = MagicMock(return_value=HealthStatus(status="ok"))
    return memory


def _make_ctx(mock_memory: AsyncMock, user_id: str = "alex") -> MagicMock:
    fake_ctx = MagicMock()
    app = AppContext(memory=mock_memory, worker=MagicMock(), user_id=user_id)
    fake_ctx.request_context.lifespan_context = app
    return fake_ctx


@pytest.fixture
def ctx(mock_memory: AsyncMock) -> MagicMock:
    return _make_ctx(mock_memory, user_id="alex")


@pytest.fixture
def ctx_no_user(mock_memory: AsyncMock) -> MagicMock:
    return _make_ctx(mock_memory, user_id="")


# -- Tools --


async def test_remember(ctx: MagicMock, mock_memory: AsyncMock) -> None:
    result = await remember("hello world", ctx)
    assert isinstance(result, str)
    assert "Remembered" in result
    mock_memory.experience.assert_awaited_once()
    call_kwargs = mock_memory.experience.call_args.kwargs
    assert call_kwargs["session_id"] == "sess-123"
    assert call_kwargs["user_id"] == "alex"


async def test_remember_with_context(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    meta = {"source": "chat"}
    result = await remember("hello", ctx, context=meta)
    assert isinstance(result, str)
    assert "Remembered" in result
    assert mock_memory.experience.call_args.kwargs["context"] == meta


async def test_remember_surfaces_important_context(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    mock_memory.surface_relevant_facts.return_value = [
        PersonaFact(
            subject="Alex",
            predicate="is_allergic_to",
            object="peanuts",
            content="Alex is allergic to peanuts",
            status="pinned",
        )
    ]
    result = await remember("booking dinner at an Asian restaurant", ctx)
    assert "IMPORTANT CONTEXT:" in result
    assert "peanuts" in result
    surfaced_trace = mock_memory.surface_relevant_facts.call_args.args[0]
    assert surfaced_trace is mock_memory.experience.return_value


async def test_remember_no_context_block_when_empty(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    result = await remember("fixed the parser bug", ctx)
    assert "IMPORTANT CONTEXT" not in result


async def test_recall(ctx: MagicMock, mock_memory: AsyncMock) -> None:
    result = await recall("what happened?", ctx)
    assert isinstance(result, str)
    assert "recalled" in result


async def test_recall_custom_budget(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    await recall("query", ctx, token_budget=500)
    assert mock_memory.think_about.call_args.kwargs["token_budget"] == 500


async def test_recall_unprimed_does_not_leak_promoted_facts(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    """Unreflected recall never injects the promoted graph via the primer.

    The persona channel is the recall-floored think_about output; recall must
    not surface the full promoted+pinned graph. Guards against re-adding an
    ungated prepend -- the probe surfacing arm bypasses MCP composition, so
    this coverage exists nowhere else. Durable across story-2: a promoted
    non-safety fact is neither pinned nor health/dietary, so the safety net
    never surfaces it either.
    """
    mock_memory.facts.return_value = [
        PersonaFact(
            subject="Angel",
            predicate="implements_rule",
            object="hypothesis independence",
            content="Angel implements the hypothesis-independence rule",
            status="promoted",
            category="constraint",
        )
    ]
    result = await recall("dinner ideas", ctx)
    assert "KNOWN FACTS AND RELATIONSHIPS" not in result
    assert "hypothesis independence" not in result


async def test_recall_surfaces_gated_persona_fact_as_important_context(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    """A persona-fact thought from think_about renders under IMPORTANT CONTEXT.

    The gated channel (recall floor + situational grounding, applied inside
    think_about) is where relevant persona facts surface.
    """
    mock_memory.think_about.return_value = [
        Thought(
            trace=MemoryTrace(
                content="Angel is allergic to penicillin",
                pattern={"persona_fact": True},
            ),
            relevance=0.95,
            token_count=10,
            reconstruction="Angel is allergic to penicillin",
        )
    ]
    result = await recall("any drug allergies?", ctx)
    assert "IMPORTANT CONTEXT:" in result
    assert "penicillin" in result


async def test_pin(ctx: MagicMock, mock_memory: AsyncMock) -> None:
    mock_memory.pin.return_value = [
        PersonaFact(
            subject="Alex",
            predicate="likes",
            object="coffee",
            content="Alex likes coffee",
            status="pinned",
            embedding=[0.1] * 768,
        )
    ]
    result = await pin("trace-123", ctx)
    assert isinstance(result, str)
    assert "Pinned 1 persona fact" in result
    assert "Alex likes coffee" in result
    assert "0.1" not in result  # embeddings projected out of the MCP payload
    mock_memory.pin.assert_awaited_once_with("trace-123")


async def test_unpin_found(ctx: MagicMock, mock_memory: AsyncMock) -> None:
    result = await unpin("fact-123", ctx)
    assert "archived" in result.lower()


async def test_unpin_not_found(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    mock_memory.unpin.return_value = False
    result = await unpin("fact-123", ctx)
    assert "not found" in result.lower()


async def test_forget(ctx: MagicMock, mock_memory: AsyncMock) -> None:
    result = await forget("trace-123", ctx)
    assert "forgotten" in result.lower()
    mock_memory.forget.assert_awaited_once_with("trace-123", force=False)


async def test_forget_reports_retained_facts(
    ctx: MagicMock, mock_memory: AsyncMock
) -> None:
    mock_memory.forget.return_value = ForgetResult(
        trace_id="trace-123",
        archived_fact_ids=["f1"],
        retained_facts=[
            PersonaFact(
                subject="user",
                predicate="is_allergic_to",
                object="shellfish",
                category="health",
                content="user is allergic to shellfish",
                status="promoted",
            )
        ],
    )
    result = await forget("trace-123", ctx)
    assert "1 derived facts archived" in result
    assert "retained (health)" in result
    assert "force=true" in result


async def test_forget_not_found(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    mock_memory.forget.side_effect = TraceNotFoundError("not found")
    result = await forget("bad-id", ctx)
    assert "not found" in result.lower()


async def test_remember_extraction_error(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    mock_memory.experience.side_effect = ExtractionError("LLM failed")
    result = await remember("test", ctx)
    assert "LLM failed" in result
    assert isinstance(result, str)


async def test_remember_storage_error(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    mock_memory.experience.side_effect = StorageError("db down")
    result = await remember("test", ctx)
    assert "db down" in result
    assert isinstance(result, str)


async def test_recall_storage_error(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    mock_memory.think_about.side_effect = StorageError("db down")
    result = await recall("query", ctx)
    assert "db down" in result
    assert isinstance(result, str)


async def test_pin_not_found(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    mock_memory.pin.side_effect = TraceNotFoundError("no such trace")
    with pytest.raises(TraceNotFoundError):
        await pin("bad-id", ctx)


async def test_unpin_sdk_error(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    mock_memory.unpin.side_effect = MemorySDKError("unexpected")
    result = await unpin("fact-id", ctx)
    assert "unexpected" in result
    assert isinstance(result, str)


async def test_forget_sdk_error(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    mock_memory.forget.side_effect = MemorySDKError("unexpected")
    result = await forget("trace-id", ctx)
    assert "unexpected" in result
    assert isinstance(result, str)


# -- Session management --


async def test_ensure_session_creates_on_first_call(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    session_id = await _ensure_session(ctx)
    assert session_id == "sess-123"
    mock_memory.start_session.assert_awaited_once()


async def test_ensure_session_reuses_existing(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    first = await _ensure_session(ctx)
    second = await _ensure_session(ctx)
    assert first == second
    mock_memory.start_session.assert_awaited_once()


async def test_ensure_session_no_user_id(
    ctx_no_user: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    result = await _ensure_session(ctx_no_user)
    assert result is None
    mock_memory.start_session.assert_not_awaited()


# -- Primer --


async def test_generate_primer_with_facts(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    app = ctx.request_context.lifespan_context
    result = await _generate_primer(app)
    assert "KNOWN FACTS" in result
    assert "Alex" in result
    assert "mother_of" in result
    assert "Sarah" in result


async def test_generate_primer_empty(
    ctx: MagicMock,
    mock_memory: AsyncMock,
) -> None:
    mock_memory.facts.return_value = []
    app = ctx.request_context.lifespan_context
    result = await _generate_primer(app)
    assert "No known" in result


# -- Resources --


async def test_format_facts(mock_memory: AsyncMock) -> None:
    facts = [
        PersonaFact(
            subject="Alex",
            predicate="is_allergic_to",
            object="peanut",
            content="Severe peanut allergy",
            status="promoted",
            category="health",
            confidence=0.95,
        )
    ]
    result = _format_facts(facts)
    assert "PERSONA FACTS (1 promoted, 0 pinned)" in result
    assert "Alex is_allergic_to peanut" in result
    assert "health" in result
    assert "0.95" in result
    assert "promoted" in result
    assert "Severe peanut allergy" in result
    assert "ago" in result or "now" in result  # humanize relative time


async def test_format_facts_empty() -> None:
    result = _format_facts([])
    assert "No persona facts" in result


async def test_health_via_app_context(ctx: MagicMock) -> None:
    app = ctx.request_context.lifespan_context
    result = app.memory.health()
    assert isinstance(result, HealthStatus)
    assert result.status == "ok"
