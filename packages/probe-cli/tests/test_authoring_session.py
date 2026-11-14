"""Authoring sessions: what reaches the model, turn order, tool-call capture.

The stand-in is an in-process MCP server behind the same MCPToolset the production
factory builds over stdio, so instructions and tool definitions take the real path.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastmcp.client.transports import FastMCPTransport
from mcp.server.mcpserver import MCPServer
from probe_cli.authoring import (
    Conversation,
    GuidanceVariant,
    SessionSpec,
    Target,
    ToolCallRecord,
    run_session,
)
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

SERVED_INSTRUCTIONS = "Stand-in server.\n\nREMEMBER -- store a statement."
REMEMBER_DOC = "Store it.\n\n    Args:\n        content: The statement.\n    "
RECALL_DOC = "Recall it."
HEAD = "You are a personal assistant for one user.\nToday's date is 2026-06-02."
CONVERSATION = Conversation(
    id="c1",
    case="elliptical-referent",
    today=date(2026, 6, 2),
    turns=["first", "second", "third, save it"],
    targets=[Target(turn=2, must_mention=["third"], reader_checks=["ok?"])],
)
Log = list[tuple[list[ModelMessage], AgentInfo]]


def _server() -> MCPServer:
    server = MCPServer(name="stand-in", instructions=SERVED_INSTRUCTIONS)

    async def remember(content: str, context: dict[str, Any] | None = None) -> str:
        return f"Stored: {content}"

    async def recall(query: str) -> str:
        return "No relevant memories found."

    server.add_tool(remember, description=REMEMBER_DOC)
    server.add_tool(recall, description=RECALL_DOC)
    return server


def _factory(user_id: str) -> MCPToolset[None]:
    return MCPToolset(FastMCPTransport(_server()))


def _scripted(log: Log) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        log.append((messages, info))
        last = messages[-1].parts[-1]
        if isinstance(last, ToolReturnPart):
            return ModelResponse(parts=[TextPart("Saved.")])
        if isinstance(last, UserPromptPart) and "save" in str(last.content):
            call = ToolCallPart("remember", {"content": f"wrote {last.content}"})
            return ModelResponse(parts=[TextPart("On it."), call])
        return ModelResponse(parts=[TextPart("Noted.")])

    return FunctionModel(respond)


def _spec(variant: GuidanceVariant) -> SessionSpec:
    return SessionSpec(
        name="t",
        execution_id="x1",
        model_id="fn",
        variant=variant,
        conversation=CONVERSATION,
        run=0,
    )


async def _seen(variant: GuidanceVariant) -> AgentInfo:
    log: Log = []
    await run_session(_spec(variant), model=_scripted(log), toolset_factory=_factory)
    return log[0][1]


async def test_served_variant_reaches_model_unchanged() -> None:
    info = await _seen(GuidanceVariant(name="served"))
    tools = {t.name: t for t in info.function_tools}
    assert tools["remember"].description == REMEMBER_DOC
    assert tools["recall"].description == RECALL_DOC
    assert info.instructions == f"{HEAD}\n\n{SERVED_INSTRUCTIONS}"


async def test_remember_override_changes_only_that_description() -> None:
    served = await _seen(GuidanceVariant(name="served"))
    variant = GuidanceVariant(name="v", remember_description="Write it whole.")
    override = await _seen(variant)
    before = {t.name: t for t in served.function_tools}
    after = {t.name: t for t in override.function_tools}
    assert after["remember"].description == "Write it whole."
    schema = after["remember"].parameters_json_schema
    assert schema == before["remember"].parameters_json_schema
    assert after["recall"] == before["recall"]
    assert override.instructions == served.instructions


async def test_instructions_override_replaces_served_text() -> None:
    variant = GuidanceVariant(name="v", instructions="OVERRIDE", system_prompt="TAIL")
    info = await _seen(variant)
    assert info.instructions == f"{HEAD}\n\nOVERRIDE\n\nTAIL"


async def test_system_prompt_follows_served_instructions() -> None:
    info = await _seen(GuidanceVariant(name="v", system_prompt="TAIL"))
    assert info.instructions == f"{HEAD}\n\n{SERVED_INSTRUCTIONS}\n\nTAIL"


async def test_turns_play_in_order_with_history_carried() -> None:
    log: Log = []
    spec = _spec(GuidanceVariant(name="served"))
    record = await run_session(spec, model=_scripted(log), toolset_factory=_factory)
    turn_openers = [m for m, _ in log if isinstance(m[-1].parts[-1], UserPromptPart)]
    prompts = [
        [p.content for m in msgs for p in m.parts if isinstance(p, UserPromptPart)]
        for msgs in turn_openers
    ]
    assert prompts == [["first"], ["first", "second"], CONVERSATION.turns]
    assert [t.user for t in record.turns] == CONVERSATION.turns
    assert [t.assistant for t in record.turns] == ["Noted.", "Noted.", "On it.\nSaved."]


async def test_tool_calls_captured_with_turn_args_and_return() -> None:
    spec = _spec(GuidanceVariant(name="served"))
    record = await run_session(spec, model=_scripted([]), toolset_factory=_factory)
    assert record.tool_calls == [
        ToolCallRecord(
            turn=2,
            tool="remember",
            args={"content": "wrote third, save it"},
            returned="Stored: wrote third, save it",
        )
    ]
    assert record.error is None
    assert record.user_id == "probe-authoring-t-x1-served-c1-r0"


async def test_writes_on_two_turns_each_recorded_once_on_their_own_turn() -> None:
    conversation = CONVERSATION.model_copy(
        update={"turns": ["first, save it", "second", "third, save it"]}
    )
    spec = _spec(GuidanceVariant(name="served")).model_copy(
        update={"conversation": conversation}
    )
    record = await run_session(spec, model=_scripted([]), toolset_factory=_factory)
    assert [(c.turn, c.args["content"]) for c in record.tool_calls] == [
        (0, "wrote first, save it"),
        (2, "wrote third, save it"),
    ]
