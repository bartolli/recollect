"""Authoring CLI: refusals exit 2 before any session; dry-run writes nothing;
a run writes its evidence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp.client.transports import FastMCPTransport
from mcp.server.mcpserver import MCPServer
from probe_cli import authoring
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

if TYPE_CHECKING:
    from collections.abc import Mapping

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "authoring"
DB_ENV = "AUTHORING_TEST_DB_URL"
SERVED_DOC = "Store it.\n\n    Args:\n        content: The statement.\n    "


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
        "--guidance",
        str(FIXTURES / "guidance" / "proposed.toml"),
        "--out",
        str(out),
        *extra,
    ]


def _stand_in(user_id: str) -> MCPToolset[None]:
    server = MCPServer(name="stand-in", instructions="Served instructions.")

    async def remember(content: str, context: dict[str, Any] | None = None) -> str:
        return content

    server.add_tool(remember, description=SERVED_DOC)
    return MCPToolset(FastMCPTransport(server))


@pytest.fixture
def factory_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def factory_builder(
        base_env: Mapping[str, str], *, db_url: str
    ) -> authoring.ToolsetFactory:
        def build(user_id: str) -> MCPToolset[None]:
            calls.append(user_id)
            return _stand_in(user_id)

        return build

    async def scratch(db_url: str) -> str:
        return authoring.SCRATCH_DB_NAME

    monkeypatch.setattr(authoring, "stdio_toolset_factory", factory_builder)
    monkeypatch.setattr(authoring, "_connected_database", scratch)
    monkeypatch.setattr(authoring, "new_execution_id", lambda: "e1")
    monkeypatch.setenv(DB_ENV, "unused-sentinel")
    return calls


@pytest.fixture
def no_sessions(factory_calls: list[str], monkeypatch: pytest.MonkeyPatch) -> list[str]:
    async def no_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("a session started")

    monkeypatch.setattr(authoring, "run_authoring", no_run)
    return factory_calls


def test_existing_evidence_dir_exits_2(tmp_path: Path, no_sessions: list[str]) -> None:
    (tmp_path / "authoring-t").mkdir()
    assert authoring.main(_argv(tmp_path, "--db-url-env", DB_ENV)) == 2
    assert no_sessions == []


def test_unset_db_env_exits_2(
    tmp_path: Path, no_sessions: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DB_ENV)
    assert authoring.main(_argv(tmp_path, "--db-url-env", DB_ENV)) == 2
    assert no_sessions == []
    assert not (tmp_path / "authoring-t").exists()


def test_dry_run_prints_served_and_effective_and_writes_nothing(
    tmp_path: Path, no_sessions: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    assert authoring.main(_argv(out, "--db-url-env", DB_ENV, "--dry-run")) == 0
    assert no_sessions == [
        "probe-authoring-t-e1-served-describe",
        "probe-authoring-t-e1-proposed-describe",
    ]
    assert not out.exists()
    printed = capsys.readouterr().out
    served, proposed = printed.split("=== variant: proposed ===")
    assert served.count(SERVED_DOC) == 2
    assert "Served instructions." in served
    assert "Decontextualize the statement" in proposed
    assert proposed.count(SERVED_DOC) == 1
    assert "Today's date is {today}." in proposed


def test_run_writes_evidence_and_refuses_a_rerun(
    tmp_path: Path,
    factory_calls: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ok")])

    model = FunctionModel(reply)
    monkeypatch.setattr(authoring, "infer_model", lambda model_id: model)
    monkeypatch.setenv("PYDANTIC_AI_MODEL", "provider:extractor")
    monkeypatch.delenv("MEMORY_CONFIG", raising=False)
    argv = _argv(tmp_path, "--db-url-env", DB_ENV)
    assert authoring.main(argv) == 0
    out = tmp_path / "authoring-t"
    assert sorted(p.name for p in out.iterdir()) == [
        "guidance.json",
        "proposed.jsonl",
        "served.jsonl",
        "summary.json",
    ]
    assert len((out / "served.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    guidance = authoring.GuidanceRecord.model_validate_json(
        (out / "guidance.json").read_text(encoding="utf-8")
    )
    assert (guidance.extraction_model, guidance.memory_config) == (
        "provider:extractor",
        "",
    )
    assert len(factory_calls) == 2 + 2 * 5
    table = capsys.readouterr().out.splitlines()
    assert table[0].split()[:5] == [
        "variant",
        "conversation",
        "run",
        "turn",
        "remember",
    ]
    assert len(table) == 1 + 2 * 6
    assert authoring.main(argv) == 2
    assert len(factory_calls) == 2 + 2 * 5
