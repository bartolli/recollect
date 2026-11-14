"""Authoring provider errors: recorded per session, the next session still runs."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp.client.transports import FastMCPTransport
from mcp.server.mcpserver import MCPServer
from probe_cli.authoring import (
    AuthoringPlan,
    Conversation,
    GuidanceRecord,
    GuidanceVariant,
    SessionRecord,
    SessionSpec,
    Target,
    ToolCallRecord,
    run_authoring,
    run_session,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

if TYPE_CHECKING:
    from pathlib import Path

REMEMBER_DOC = "Store it."
CONVERSATION = Conversation(
    id="c1",
    case="correction",
    today=date(2026, 10, 5),
    turns=["save the first", "second", "third"],
    targets=[Target(turn=0, must_mention=["first"], reader_checks=["?"])],
)


def _factory(user_id: str) -> MCPToolset[None]:
    server = MCPServer(name="stand-in", instructions="Stand-in server.")

    async def remember(content: str, context: dict[str, Any] | None = None) -> str:
        return f"Stored: {content}"

    server.add_tool(remember, description=REMEMBER_DOC)
    return MCPToolset(FastMCPTransport(server))


def _failing_on(*failing: int) -> FunctionModel:
    requests = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests in failing:
            raise ModelHTTPError(status_code=503, model_name="m")
        if requests == 1:
            call = ToolCallPart("remember", {"content": "the first"})
            return ModelResponse(parts=[call])
        return ModelResponse(parts=[TextPart("ok")])

    return FunctionModel(respond)


async def test_provider_error_recorded_and_next_session_runs(tmp_path: Path) -> None:
    plan = AuthoringPlan(
        name="t",
        model_id="m",
        runs=2,
        variants=[GuidanceVariant(name="served")],
        conversations=[CONVERSATION],
    )
    out = tmp_path / "authoring-t"
    model = _failing_on(1)
    summary = await run_authoring(
        plan, model=model, toolset_factory=_factory, out_dir=out
    )
    lines = (out / "served.jsonl").read_text(encoding="utf-8").splitlines()
    first, second = (SessionRecord.model_validate_json(line) for line in lines)
    assert first.error is not None
    assert first.error.startswith("ModelHTTPError")
    assert (first.turns, first.tool_calls, first.run) == ([], [], 0)
    assert (second.error, len(second.turns), second.run) == (None, 3, 1)
    assert [s.error is None for s in summary.sessions] == [False, True]
    guidance = GuidanceRecord.model_validate_json(
        (out / "guidance.json").read_text(encoding="utf-8")
    )
    (described,) = guidance.variants
    assert described.served_remember_description == REMEMBER_DOC
    assert described.remember_description == REMEMBER_DOC
    assert described.instructions.startswith(
        "You are a personal assistant for one user.\nToday's date is {today}."
    )
    assert (out / "summary.json").exists()


async def test_server_that_cannot_start_leaves_no_evidence_dir(tmp_path: Path) -> None:
    def unstartable(user_id: str) -> MCPToolset[None]:
        raise OSError("server did not start")

    plan = AuthoringPlan(
        name="t",
        model_id="m",
        runs=1,
        variants=[GuidanceVariant(name="served")],
        conversations=[CONVERSATION],
    )
    out = tmp_path / "authoring-t"
    with pytest.raises(OSError, match="did not start"):
        await run_authoring(
            plan, model=_failing_on(), toolset_factory=unstartable, out_dir=out
        )
    assert not out.exists()


async def test_write_before_provider_error_stays_in_the_record() -> None:
    spec = SessionSpec(
        name="t",
        execution_id="x1",
        model_id="m",
        variant=GuidanceVariant(name="served"),
        conversation=CONVERSATION,
        run=0,
    )
    record = await run_session(spec, model=_failing_on(2), toolset_factory=_factory)
    assert record.error is not None
    assert record.turns == []
    assert record.tool_calls == [
        ToolCallRecord(
            turn=0,
            tool="remember",
            args={"content": "the first"},
            returned="Stored: the first",
        )
    ]
