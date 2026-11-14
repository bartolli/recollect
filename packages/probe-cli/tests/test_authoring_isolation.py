"""Authoring isolation: only the scratch database is ever started against, and a
rerun under one name starts on empty memory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp.client.transports import FastMCPTransport
from mcp.server.mcpserver import MCPServer
from probe_cli import authoring
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

if TYPE_CHECKING:
    from collections.abc import Mapping

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "authoring"
DB_ENV = "AUTHORING_TEST_DB_URL"
Store = dict[str, list[str]]


def _argv(out: Path, *extra: str) -> list[str]:
    return [
        "--name",
        "t",
        "--model",
        "test",
        "--conversations",
        str(FIXTURES / "conversations.jsonl"),
        "--guidance",
        str(FIXTURES / "guidance" / "served.toml"),
        "--db-url-env",
        DB_ENV,
        "--out",
        str(out),
        *extra,
    ]


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Store:
    """Writes per server user id, kept across runs like the real database."""
    writes: Store = {}

    def factory_builder(
        base_env: Mapping[str, str], *, db_url: str
    ) -> authoring.ToolsetFactory:
        def build(user_id: str) -> MCPToolset[None]:
            server = MCPServer(name="stand-in", instructions="Served.")

            async def remember(
                content: str, context: dict[str, Any] | None = None
            ) -> str:
                writes.setdefault(user_id, []).append(content)
                return "stored"

            server.add_tool(remember, description="Store it.")
            return MCPToolset(FastMCPTransport(server))

        return build

    monkeypatch.setattr(authoring, "stdio_toolset_factory", factory_builder)
    monkeypatch.setenv(DB_ENV, "unused-sentinel")
    return writes


def _connects_to(monkeypatch: pytest.MonkeyPatch, name: str) -> list[str]:
    started: list[str] = []

    async def connected(db_url: str) -> str:
        return name

    def no_server(base_env: Mapping[str, str], *, db_url: str) -> None:
        started.append(db_url)

    monkeypatch.setattr(authoring, "_connected_database", connected)
    if name != authoring.SCRATCH_DB_NAME:
        monkeypatch.setattr(authoring, "stdio_toolset_factory", no_server)
    return started


@pytest.mark.parametrize("database", ["probe_task", "recollect_bartolli", "memory_v3"])
@pytest.mark.parametrize("extra", [(), ("--dry-run",)])
def test_other_database_refused_before_any_server(
    tmp_path: Path,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    database: str,
    extra: tuple[str, ...],
) -> None:
    started = _connects_to(monkeypatch, database)
    assert authoring.main(_argv(tmp_path, *extra)) == 2
    assert started == []
    assert not (tmp_path / "authoring-t").exists()


def test_unreachable_database_exits_2(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unreachable(db_url: str) -> str:
        raise OSError("connection refused")

    monkeypatch.setattr(authoring, "_connected_database", unreachable)
    assert authoring.main(_argv(tmp_path)) == 2
    assert store == {}


def _writes_once(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    prompts = [p for m in messages for p in m.parts if isinstance(p, UserPromptPart)]
    if len(prompts) == 1 and isinstance(messages[-1].parts[-1], UserPromptPart):
        return ModelResponse(parts=[ToolCallPart("remember", {"content": "a fact"})])
    return ModelResponse(parts=[TextPart("ok")])


def test_rerun_into_another_out_dir_starts_on_empty_memory(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    _connects_to(monkeypatch, authoring.SCRATCH_DB_NAME)
    model = FunctionModel(_writes_once)
    monkeypatch.setattr(authoring, "infer_model", lambda model_id: model)
    first, second = tmp_path / "first", tmp_path / "second"
    assert authoring.main(_argv(first)) == 0
    assert authoring.main(_argv(second)) == 0
    # Five sessions per run, one write each: a user id reused by the second
    # run would hold two writes.
    assert len(store) == 10
    assert all(len(written) == 1 for written in store.values())
    ids = [
        authoring.GuidanceRecord.model_validate_json(
            (out / "authoring-t" / "guidance.json").read_text(encoding="utf-8")
        ).execution_id
        for out in (first, second)
    ]
    assert ids[0] != ids[1]
    prefixes = tuple(f"probe-authoring-t-{execution_id}-" for execution_id in ids)
    assert all(user_id.startswith(prefixes) for user_id in store)
