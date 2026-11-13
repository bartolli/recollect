"""Safety-net priming is per connection, not per process.

Streamable HTTP shares the lifespan state across every client, so the
one-shot is keyed on the connection: the stdio process, or the HTTP
`Mcp-Session-Id`. A request that carries no session id has no connection
to prime and always gets the net.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from recollect.config import config
from recollect.models import MemoryTrace, PersonaFact, Thought
from recollect_mcp import server
from recollect_mcp.server import AppContext, recall, reflect


def _ctx(app: AppContext, headers: dict[str, str] | None) -> MagicMock:
    fake = MagicMock()
    fake.request_context.lifespan_context = app
    fake.headers = headers
    return fake


@pytest.fixture
def app() -> AppContext:
    memory = AsyncMock()
    memory.think_about.return_value = [
        Thought(
            trace=MemoryTrace(content="weekend plans"),
            relevance=0.7,
            token_count=8,
            reconstruction="weekend plans",
        )
    ]
    memory.facts.return_value = [
        PersonaFact(
            subject="Angel",
            predicate="is_allergic_to",
            object="penicillin",
            content="Angel is allergic to penicillin",
            status="pinned",
            category="health",
        )
    ]
    return AppContext(memory=memory, worker=MagicMock(), user_id="alex")


async def test_each_http_session_gets_the_safety_net_once(app: AppContext) -> None:
    a = _ctx(app, {"mcp-session-id": "session-a"})
    b = _ctx(app, {"mcp-session-id": "session-b"})
    assert "penicillin" in await recall("one", a)
    assert "penicillin" not in await recall("two", a)
    assert "penicillin" in await recall("one", b)


async def test_reflect_primes_only_its_own_http_session(app: AppContext) -> None:
    a = _ctx(app, {"mcp-session-id": "session-a"})
    b = _ctx(app, {"mcp-session-id": "session-b"})
    await reflect(a)
    assert "penicillin" not in await recall("one", a)
    assert "penicillin" in await recall("one", b)


async def test_request_without_a_session_id_always_gets_the_safety_net(
    app: AppContext,
) -> None:
    ctx = _ctx(app, {})
    await reflect(ctx)
    assert "penicillin" in await recall("one", ctx)
    assert "penicillin" in await recall("two", ctx)


async def test_stdio_process_is_one_connection(app: AppContext) -> None:
    assert "penicillin" in await recall("one", _ctx(app, None))
    assert "penicillin" not in await recall("two", _ctx(app, None))


def test_streamable_http_binds_the_configured_host_and_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = MagicMock()
    monkeypatch.setattr(
        sys, "argv", ["recollect-mcp", "--transport", "streamable-http"]
    )
    monkeypatch.setattr(server.mcp, "run", run)
    monkeypatch.setattr(server, "configure_logging", MagicMock())
    server.main()
    run.assert_called_once_with(
        transport="streamable-http",
        host=str(config.get("server.host")),
        port=int(config.get("server.port")),
    )
