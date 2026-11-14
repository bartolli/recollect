"""Guidance authoring comparison: scripted conversations played to a model holding
the real recollect MCP tools, one guidance variant at a time; records its remember
calls."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tomllib
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import asyncpg
from fastmcp.client.transports import StdioTransport
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import (
    ModelAPIError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import infer_model
from pydantic_ai.models.function import AgentInfo, FunctionModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic_ai import AgentRunResult, RunContext
    from pydantic_ai.models import Model
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.toolsets import AbstractToolset

logger = logging.getLogger(__name__)

REMEMBER = "remember"
ROLE_LINE = "You are a personal assistant for one user."
TODAY_PLACEHOLDER = "{today}"
# Server start bootstraps the schema, also under --dry-run, so the connected
# database name is an invariant, not a knob: no other store is ever migrated.
SCRATCH_DB_NAME = "probe_eval"
# Cold-start headroom: the server opens storage and loads the embedding model
# in its lifespan, before it answers initialize; the MCPToolset default is 5 s.
_SERVER_INIT_TIMEOUT = 180.0
_PROVIDER_ERRORS = (ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded)
# Variant names and conversation ids key evidence files and server user ids.
_SLUG = r"^[a-z0-9][a-z0-9-]*$"

_Turn = Annotated[str, Field(pattern=r"\S")]
ToolsetFactory = Callable[[str], MCPToolset[None]]


class GuidanceVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_SLUG)
    remember_description: str | None = None
    instructions: str | None = None
    system_prompt: str | None = None


class Target(BaseModel):
    turn: int = Field(ge=0)
    must_mention: list[str] = Field(min_length=1)
    reader_checks: list[str] = Field(min_length=1)


class Conversation(BaseModel):
    id: str = Field(pattern=_SLUG)
    case: str
    today: date
    turns: list[_Turn] = Field(min_length=1)
    targets: list[Target]

    @model_validator(mode="after")
    def _targets_in_range(self) -> Conversation:
        for target in self.targets:
            if target.turn >= len(self.turns):
                raise ValueError(
                    f"target turn {target.turn} is past the last turn of {self.id}"
                )
        return self


class ToolCallRecord(BaseModel):
    turn: int = Field(ge=0)
    tool: str
    args: dict[str, Any]
    returned: str | None


class TurnRecord(BaseModel):
    turn: int = Field(ge=0)
    user: str
    assistant: str


class SessionRecord(BaseModel):
    variant: str
    conversation_id: str
    run: int = Field(ge=0)
    model: str
    user_id: str
    turns: list[TurnRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    error: str | None = None


class VariantGuidance(BaseModel):
    variant: str
    served_remember_description: str
    served_instructions: str
    # As the model receives them; instructions keep the {today} placeholder.
    remember_description: str
    instructions: str


class GuidanceRecord(BaseModel):
    execution_id: str
    extraction_model: str
    memory_config: str
    git_head: str
    git_dirty: bool
    variants: list[VariantGuidance]


class TargetScore(BaseModel):
    turn: int = Field(ge=0)
    remember_calls: int = Field(ge=0)
    mentions: dict[str, bool]


class SessionScore(BaseModel):
    variant: str
    conversation_id: str
    run: int = Field(ge=0)
    error: str | None
    targets: list[TargetScore]


class AuthoringSummary(BaseModel):
    name: str
    execution_id: str
    model: str
    sessions: list[SessionScore]


def new_execution_id() -> str:
    return uuid.uuid4().hex[:8]


def user_namespace(name: str, execution_id: str) -> str:
    # Server user ids are the only isolation between sessions: without the
    # execution id, a rerun under the same name starts on the memory an
    # earlier execution wrote.
    return f"probe-authoring-{name}-{execution_id}"


class SessionSpec(BaseModel):
    name: str
    execution_id: str
    model_id: str
    variant: GuidanceVariant
    conversation: Conversation
    run: int = Field(ge=0)

    @property
    def user_id(self) -> str:
        namespace = user_namespace(self.name, self.execution_id)
        return f"{namespace}-{self.variant.name}-{self.conversation.id}-r{self.run}"


class AuthoringPlan(BaseModel):
    name: str
    execution_id: str = Field(default_factory=new_execution_id)
    model_id: str
    # PYDANTIC_AI_MODEL and MEMORY_CONFIG as the server child receives them:
    # they set the extraction and assessment model, which model_id does not.
    extraction_model: str = ""
    memory_config: str = ""
    runs: int = Field(ge=1)
    variants: list[GuidanceVariant]
    conversations: list[Conversation]

    @property
    def namespace(self) -> str:
        return user_namespace(self.name, self.execution_id)

    def specs(self) -> list[SessionSpec]:
        return [
            SessionSpec(
                name=self.name,
                execution_id=self.execution_id,
                model_id=self.model_id,
                variant=variant,
                conversation=conversation,
                run=run,
            )
            for variant in self.variants
            for conversation in self.conversations
            for run in range(self.runs)
        ]


def _require_unique(names: Sequence[str], *, kind: str) -> None:
    duplicates = sorted(n for n, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {kind}: {', '.join(duplicates)}")


def load_conversations(path: Path) -> list[Conversation]:
    lines = path.read_text(encoding="utf-8").splitlines()
    conversations = [
        Conversation.model_validate_json(line)
        for line in lines
        if line.strip() and not line.startswith("#")
    ]
    _require_unique([c.id for c in conversations], kind="conversation id")
    return conversations


def load_variants(paths: Sequence[Path]) -> list[GuidanceVariant]:
    variants = [
        GuidanceVariant.model_validate(tomllib.loads(p.read_text(encoding="utf-8")))
        for p in paths
    ]
    # A repeated name would share a jsonl file and a server user id.
    _require_unique([v.name for v in variants], kind="guidance variant name")
    return variants


def stdio_toolset_factory(
    base_env: Mapping[str, str], *, db_url: str
) -> ToolsetFactory:
    def build(user_id: str) -> MCPToolset[None]:
        # The stdio client passes the child only a minimal default environment
        # plus env: the provider model and API keys come from base_env.
        env = {**base_env, "MEMORY_USER_ID": user_id, "DATABASE_URL": db_url}
        transport = StdioTransport(
            command=sys.executable,
            args=["-m", "recollect_mcp.server"],
            env=env,
            # The child exits with its session: one server per user id.
            keep_alive=False,
        )
        return MCPToolset(transport, init_timeout=_SERVER_INIT_TIMEOUT)

    return build


def agent_instructions(
    *, today: str, server_instructions: str, system_prompt: str | None
) -> str:
    parts = [f"{ROLE_LINE}\nToday's date is {today}.", server_instructions]
    if system_prompt:
        parts.append(system_prompt)
    return "\n\n".join(parts)


def _variant_toolset(
    toolset: MCPToolset[None], variant: GuidanceVariant
) -> AbstractToolset[None]:
    description = variant.remember_description
    if description is None:
        return toolset

    def prepare(
        ctx: RunContext[None], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [
            replace(d, description=description) if d.name == REMEMBER else d
            for d in tool_defs
        ]

    return toolset.prepared(prepare)


def _agent(
    toolset: MCPToolset[None], *, variant: GuidanceVariant, today: str, model: Model
) -> Agent[None, str]:
    server_instructions = variant.instructions
    if server_instructions is None:
        server_instructions = toolset.instructions or ""
    instructions = agent_instructions(
        today=today,
        server_instructions=server_instructions,
        system_prompt=variant.system_prompt,
    )
    return Agent(
        model, instructions=instructions, toolsets=[_variant_toolset(toolset, variant)]
    )


def _assistant_text(messages: Sequence[ModelMessage]) -> str:
    return "\n".join(
        part.content
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, TextPart)
    )


def _tool_calls(messages: Sequence[ModelMessage], *, turn: int) -> list[ToolCallRecord]:
    parts = [part for message in messages for part in message.parts]
    returned = {
        p.tool_call_id: p.model_response_str()
        for p in parts
        if isinstance(p, ToolReturnPart)
    }
    return [
        ToolCallRecord(
            turn=turn,
            tool=p.tool_name,
            args=p.args_as_dict(),
            returned=returned.get(p.tool_call_id),
        )
        for p in parts
        if isinstance(p, ToolCallPart)
    ]


async def _run_captured(
    agent: Agent[None, str],
    *,
    user_turn: str,
    history: list[ModelMessage],
    new: list[ModelMessage],
) -> AgentRunResult[str]:
    with capture_run_messages() as messages:
        try:
            return await agent.run(user_turn, message_history=history)
        finally:
            new.extend(messages[len(history) :])


async def _play(
    agent: Agent[None, str], *, conversation: Conversation, record: SessionRecord
) -> None:
    history: list[ModelMessage] = []
    for index, user_turn in enumerate(conversation.turns):
        new: list[ModelMessage] = []
        try:
            result = await _run_captured(
                agent, user_turn=user_turn, history=history, new=new
            )
        finally:
            # Also on a provider error: a write the server already made stays
            # in the evidence.
            record.tool_calls.extend(_tool_calls(new, turn=index))
        record.turns.append(
            TurnRecord(turn=index, user=user_turn, assistant=_assistant_text(new))
        )
        history = result.all_messages()


async def run_session(
    spec: SessionSpec, *, model: Model, toolset_factory: ToolsetFactory
) -> SessionRecord:
    record = SessionRecord(
        variant=spec.variant.name,
        conversation_id=spec.conversation.id,
        run=spec.run,
        model=spec.model_id,
        user_id=spec.user_id,
    )
    toolset = toolset_factory(spec.user_id)
    # One context for every turn: re-entering per run would respawn the server.
    async with toolset:
        today = spec.conversation.today.isoformat()
        agent = _agent(toolset, variant=spec.variant, today=today, model=model)
        try:
            await _play(agent, conversation=spec.conversation, record=record)
        except _PROVIDER_ERRORS as exc:
            logger.exception("session %s ended by a provider error", spec.user_id)
            record.error = f"{type(exc).__name__}: {exc}"
    return record


async def _model_view(toolset: MCPToolset[None], variant: GuidanceVariant) -> AgentInfo:
    # A local FunctionModel observes what the session agent sends; no provider
    # call, and no tool call because it answers with text.
    seen: list[AgentInfo] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart("described")])

    model = FunctionModel(capture)
    agent = _agent(toolset, variant=variant, today=TODAY_PLACEHOLDER, model=model)
    await agent.run("describe")
    return seen[0]


async def describe_variant(
    variant: GuidanceVariant, *, namespace: str, toolset_factory: ToolsetFactory
) -> VariantGuidance:
    toolset = toolset_factory(f"{namespace}-{variant.name}-describe")
    async with toolset:
        served = {t.name: t.description or "" for t in await toolset.list_tools()}
        view = await _model_view(toolset, variant)
        effective = {d.name: d.description or "" for d in view.function_tools}
        return VariantGuidance(
            variant=variant.name,
            served_remember_description=served[REMEMBER],
            served_instructions=toolset.instructions or "",
            remember_description=effective[REMEMBER],
            instructions=view.instructions or "",
        )


async def _git(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        "git", *args, stdout=asyncio.subprocess.PIPE
    )
    out, _ = await process.communicate()
    return out.decode().strip()


def _write_text(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


async def _guidance_record(
    plan: AuthoringPlan, *, toolset_factory: ToolsetFactory
) -> GuidanceRecord:
    variants = [
        await describe_variant(
            v, namespace=plan.namespace, toolset_factory=toolset_factory
        )
        for v in plan.variants
    ]
    return GuidanceRecord(
        execution_id=plan.execution_id,
        extraction_model=plan.extraction_model,
        memory_config=plan.memory_config,
        git_head=await _git("rev-parse", "HEAD"),
        git_dirty=bool(await _git("status", "--porcelain")),
        variants=variants,
    )


def _score_target(record: SessionRecord, target: Target) -> TargetScore:
    contents = [
        str(call.args.get("content", ""))
        for call in record.tool_calls
        if call.turn == target.turn and call.tool == REMEMBER
    ]
    written = "\n".join(contents).lower()
    return TargetScore(
        turn=target.turn,
        remember_calls=len(contents),
        mentions={term: term.lower() in written for term in target.must_mention},
    )


def score_session(record: SessionRecord, conversation: Conversation) -> SessionScore:
    return SessionScore(
        variant=record.variant,
        conversation_id=record.conversation_id,
        run=record.run,
        error=record.error,
        targets=[_score_target(record, t) for t in conversation.targets],
    )


async def run_authoring(
    plan: AuthoringPlan,
    *,
    model: Model,
    toolset_factory: ToolsetFactory,
    out_dir: Path,
) -> AuthoringSummary:
    # Before the evidence dir exists: a server that cannot start must not leave
    # an empty dir that refuses the next run under the same name.
    guidance = await _guidance_record(plan, toolset_factory=toolset_factory)
    await asyncio.to_thread(out_dir.mkdir, parents=True)
    await asyncio.to_thread(
        _write_text, out_dir / "guidance.json", guidance.model_dump_json(indent=2)
    )
    scores: list[SessionScore] = []
    for spec in plan.specs():
        record = await run_session(spec, model=model, toolset_factory=toolset_factory)
        path = out_dir / f"{spec.variant.name}.jsonl"
        await asyncio.to_thread(_append_line, path, record.model_dump_json())
        scores.append(score_session(record, spec.conversation))
    summary = AuthoringSummary(
        name=plan.name,
        execution_id=plan.execution_id,
        model=plan.model_id,
        sessions=scores,
    )
    path = out_dir / "summary.json"
    await asyncio.to_thread(_write_text, path, summary.model_dump_json(indent=2))
    return summary


_TABLE_HEADER = ("variant", "conversation", "run", "turn", "remember")


def _score_rows(score: SessionScore) -> list[tuple[list[str], str]]:
    rows: list[tuple[list[str], str]] = []
    for target in score.targets:
        cells = [
            score.variant,
            score.conversation_id,
            str(score.run),
            str(target.turn),
            str(target.remember_calls),
        ]
        hits = " ".join(
            f"{term}={'hit' if hit else 'miss'}"
            for term, hit in target.mentions.items()
        )
        if score.error:
            hits += f"  error: {score.error.splitlines()[0]}"
        rows.append((cells, hits))
    return rows


def summary_table(summary: AuthoringSummary) -> str:
    rows = [(list(_TABLE_HEADER), "must_mention")]
    rows.extend(row for score in summary.sessions for row in _score_rows(score))
    columns = range(len(_TABLE_HEADER))
    widths = [max(len(cells[i]) for cells, _ in rows) for i in columns]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + "  " + tail
        for cells, tail in rows
    )


def guidance_text(guidance: VariantGuidance) -> str:
    sections = [
        ("served remember description", guidance.served_remember_description),
        ("effective remember description", guidance.remember_description),
        ("served instructions", guidance.served_instructions),
        ("effective instructions", guidance.instructions),
    ]
    body = "\n".join(f"--- {title} ---\n{text}" for title, text in sections)
    return f"=== variant: {guidance.variant} ===\n{body}\n"


def _emit(text: str) -> None:
    # sys.stdout rather than print: T201 binds this package outside __main__.
    sys.stdout.write(text + "\n")


async def _dry_run(
    variants: Sequence[GuidanceVariant],
    *,
    namespace: str,
    toolset_factory: ToolsetFactory,
) -> None:
    for variant in variants:
        guidance = await describe_variant(
            variant, namespace=namespace, toolset_factory=toolset_factory
        )
        _emit(guidance_text(guidance))


async def _connected_database(db_url: str) -> str:
    conn = await asyncpg.connect(db_url)
    try:
        return str(await conn.fetchval("SELECT current_database()"))
    finally:
        await conn.close()


def _scratch_db_url(env_name: str, environ: Mapping[str, str]) -> str | None:
    url = environ.get(env_name)
    if not url:
        logger.error("%s is unset", env_name)
        return None
    # Read from a live connection, not the URL: the env var name says nothing
    # about its target, and a URL without a database resolves elsewhere.
    try:
        name = asyncio.run(_connected_database(url))
    except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, ValueError):
        logger.exception("cannot connect through %s", env_name)
        return None
    if name != SCRATCH_DB_NAME:
        logger.error(
            "%s connects to %r; authoring runs only on %r",
            env_name,
            name,
            SCRATCH_DB_NAME,
        )
        return None
    return url


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m probe_cli.authoring",
        description="Compare remember guidance variants over scripted conversations.",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True, help="pydantic-ai model id")
    parser.add_argument("--conversations", required=True, type=Path)
    parser.add_argument("--guidance", required=True, type=Path, action="append")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--db-url-env", default="RECOLLECT_PROBE_DB_URL")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _parser().parse_args(argv)
    db_url = _scratch_db_url(args.db_url_env, os.environ)
    if db_url is None:
        return 2
    try:
        variants = load_variants(args.guidance)
        conversations = load_conversations(args.conversations)
    except (OSError, ValueError):
        logger.exception("cannot load guidance or conversations")
        return 2
    factory = stdio_toolset_factory(os.environ, db_url=db_url)
    execution_id = new_execution_id()
    if args.dry_run:
        namespace = user_namespace(args.name, execution_id)
        asyncio.run(_dry_run(variants, namespace=namespace, toolset_factory=factory))
        return 0
    out_dir = args.out / f"authoring-{args.name}"
    if out_dir.exists():
        logger.error("evidence exists at %s; move it or use a new --name", out_dir)
        return 2
    plan = AuthoringPlan(
        name=args.name,
        execution_id=execution_id,
        model_id=args.model,
        extraction_model=os.environ.get("PYDANTIC_AI_MODEL", ""),
        memory_config=os.environ.get("MEMORY_CONFIG", ""),
        runs=args.runs,
        variants=variants,
        conversations=conversations,
    )
    model = infer_model(args.model)
    summary = asyncio.run(
        run_authoring(plan, model=model, toolset_factory=factory, out_dir=out_dir)
    )
    _emit(summary_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
