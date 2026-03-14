"""MCP server for Memory SDK.

Exposes CognitiveMemory as MCP tools and resources.
Supports stdio and streamable-http transports.

6 tools: remember, recall, reflect, pin, unpin, forget
3 resources: primer, facts, health
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import humanize
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.resources import FunctionResource
from mcp.server.session import ServerSession
from pydantic import AnyUrl
from recollect.config import config
from recollect.core import CognitiveMemory
from recollect.datetime_utils import now_utc
from recollect.exceptions import (
    ExtractionError,
    MemorySDKError,
    StorageError,
    TraceNotFoundError,
)
from recollect.extraction import PatternExtractor
from recollect.log_setup import configure_logging
from recollect.models import (
    HealthStatus,
    MemoryTrace,
    PersonaFact,
    Thought,
)
from recollect.worker import ConsolidationWorker

logger = logging.getLogger(__name__)

# Type alias for our Context
Ctx = Context[ServerSession, "AppContext", Any]


@dataclass
class AppContext:
    """Lifespan context holding memory and worker instances."""

    memory: CognitiveMemory
    worker: ConsolidationWorker
    user_id: str
    _sessions: dict[str, str] = field(default_factory=dict)
    primed: bool = False


def _create_extractor() -> PatternExtractor:
    """Create the LLM-backed PatternExtractor using pydantic-ai.

    pydantic-ai supports 20+ model backends via model string format:
    "anthropic:claude-sonnet-4-6", "openai:gpt-4o", "ollama:llama3", etc.
    Configure via PYDANTIC_AI_MODEL environment variable.

    Raises ExtractionError if the provider cannot be initialized.
    """
    from recollect.llm.pydantic_ai import PydanticAIProvider

    try:
        provider = PydanticAIProvider()
        logger.info("LLM extraction enabled (pydantic-ai, %s)", provider.model_name)
        return PatternExtractor(provider)
    except (ExtractionError, ImportError, ValueError) as exc:
        raise ExtractionError(
            "Failed to initialize pydantic-ai provider. Set PYDANTIC_AI_MODEL "
            "environment variable (e.g., 'anthropic:claude-sonnet-4-6', "
            "'ollama:llama3')."
        ) from exc


@asynccontextmanager
async def app_lifespan(
    server: FastMCP[AppContext],
) -> AsyncIterator[AppContext]:
    """Manage CognitiveMemory and worker lifecycle.

    LLM extraction is mandatory -- concept attention relies on
    LLM-extracted tags for retrieval.
    """
    try:
        extractor = _create_extractor()
    except (ExtractionError, ImportError, ValueError):
        logger.exception("Server cannot start without an LLM provider")
        raise
    memory = CognitiveMemory(
        extractor=extractor,
    )
    await memory.connect()
    await memory._embeddings.warm()  # pre-load ONNX model at startup
    worker = ConsolidationWorker(memory)
    worker.start()
    user_id = config.server_user_id
    if not user_id:
        raise RuntimeError(
            "MEMORY_USER_ID is required. "
            "Set it via environment variable or server.user_id in config."
        )
    logger.info("Memory SDK server started, user_id=%s", user_id)
    app = AppContext(memory=memory, worker=worker, user_id=user_id)

    async def _primer() -> str:
        return await _generate_primer(app)

    async def _facts() -> str:
        all_facts = await app.memory.facts()
        active = [f for f in all_facts if f.status in ("promoted", "pinned")]
        return _format_facts(active)

    def _health() -> HealthStatus:
        return app.memory.health()

    mcp.add_resource(
        FunctionResource(
            uri=AnyUrl("memory://primer"),
            name="primer",
            description="Known relationships and key facts as a relational graph.",
            fn=_primer,
        )
    )
    mcp.add_resource(
        FunctionResource(
            uri=AnyUrl("memory://facts"),
            name="facts",
            description="Promoted and pinned persona facts.",
            fn=_facts,
        )
    )
    mcp.add_resource(
        FunctionResource(
            uri=AnyUrl("memory://health"),
            name="health",
            description="System health and statistics.",
            fn=_health,
        )
    )

    try:
        yield app
    finally:
        worker.stop()
        await memory.close()
        logger.info("Memory SDK server stopped")


def _get_ctx(ctx: Ctx) -> AppContext:
    return ctx.request_context.lifespan_context


def _get_memory(ctx: Ctx) -> CognitiveMemory:
    return _get_ctx(ctx).memory


async def _ensure_session(ctx: Ctx) -> str | None:
    """Lazy-init a session for the current connection.

    Returns session_id or None if no user_id is configured.
    """
    app = _get_ctx(ctx)
    if not app.user_id:
        return None
    # Use a fixed key for stdio (single connection).
    # For HTTP, could use MCP session ID.
    connection_key = "default"
    if connection_key not in app._sessions:
        session = await app.memory.start_session(user_id=app.user_id)
        app._sessions[connection_key] = session.id
        logger.info(
            "Auto-created session %s for user %s",
            session.id,
            app.user_id,
        )
    return app._sessions[connection_key]


_INSTRUCTIONS = """\
Cognitive memory system for personal AI assistants.

You have 6 tools:

REMEMBER -- Store an experience in memory.
  Write complete, contextually-anchored statements that include relational
  identifiers. Use "Alex's mother Sarah" not just "Sarah" when names could
  refer to different people. What you write is what the system indexes --
  ambiguous input produces ambiguous recall.

RECALL -- Retrieve relevant memories for a query.
  Returns persona facts as IMPORTANT CONTEXT followed by related memories.
  Be specific in your queries. The more precise the query, the better the
  results. You do not need to pass user_id or session_id -- the server
  manages these automatically.

PIN -- Promote a memory to a permanent persona fact.
  Use this for stable truths about the user: allergies, relationships,
  preferences. Pinned facts are always surfaced during relevant recall.

UNPIN -- Remove a persona fact that is no longer accurate.

FORGET -- Remove an incorrect or irrelevant memory.

REFLECT -- Load persona context before responding to the user.
  Call this at the start of every session before your first response.
  Returns known relationships and promoted facts. Without this context,
  you risk missing safety-critical information like allergies, medical
  conditions, and relationship constraints. No parameters needed.

RESOURCES:
  memory://primer -- Read this at the start of every conversation.
    Contains the user's known relationships and key facts as a compact
    relational graph. This is your context foundation.

  memory://facts -- All promoted and pinned persona facts.

  memory://health -- System health and statistics.\
"""

mcp = FastMCP(
    name="memory-sdk",
    instructions=_INSTRUCTIONS,
    host=str(config.get("server.host", "127.0.0.1")),
    port=int(config.get("server.port", 8000)),
    lifespan=app_lifespan,
)


# -- Tools --


@mcp.tool()
async def remember(
    content: str,
    ctx: Ctx,
    context: dict[str, Any] | None = None,
) -> str:
    """Store an experience in memory.

    Write complete, contextually-anchored statements. Include relational
    identifiers like "Alex's mother Sarah" rather than bare names when
    the same name could refer to different people.

    If you haven't called reflect this session, do so before using this tool.

    Args:
        content: The experience to remember. Be specific and complete.
        context: Optional metadata dict attached to the memory trace.
    """
    app = _get_ctx(ctx)
    try:
        session_id = await _ensure_session(ctx)
        trace = await app.memory.experience(
            content,
            context=context,
            session_id=session_id,
            user_id=app.user_id or None,
        )
        return _format_remember_result(trace)
    except ExtractionError as exc:
        logger.exception("Extraction failed for remember")
        return f"Extraction failed: {exc}"
    except StorageError as exc:
        logger.exception("Storage error in remember")
        return f"Storage failed: {exc}"
    except MemorySDKError as exc:
        logger.exception("Unexpected SDK error in remember")
        return f"Memory error: {exc}"


@mcp.tool()
async def recall(
    query: str,
    ctx: Ctx,
    token_budget: int = 2000,
) -> str:
    """Retrieve relevant memories for a query.

    Returns persona facts as IMPORTANT CONTEXT followed by related
    memories. The more specific the query, the more precise the results.

    If you haven't called reflect this session, do so before using this tool.

    Args:
        query: What to recall. Be specific.
        token_budget: Maximum tokens in the response (default 2000).
    """
    app = _get_ctx(ctx)
    try:
        if not app.primed:
            app.primed = True
            primer = await _generate_primer(app)
            primer_tokens = len(primer) // 4
            adjusted_budget = max(token_budget - primer_tokens, 500)
            thoughts = await app.memory.think_about(
                query,
                token_budget=adjusted_budget,
                user_id=app.user_id or None,
            )
            return f"{primer}\n\n{_format_thoughts(thoughts)}"
        thoughts = await app.memory.think_about(
            query,
            token_budget=token_budget,
            user_id=app.user_id or None,
        )
        return _format_thoughts(thoughts)
    except StorageError as exc:
        logger.exception("Storage error in recall")
        return f"Recall failed: {exc}"
    except MemorySDKError as exc:
        logger.exception("Unexpected SDK error in recall")
        return f"Recall failed: {exc}"


@mcp.tool()
async def reflect(ctx: Ctx) -> str:
    """Load persona context and known facts for the current session.

    Call this BEFORE responding to any user message if you haven't
    already this session. Without this context, you risk missing
    safety-critical information about the user's relationships,
    health conditions, and active constraints.

    Returns the user's identity, relationships, promoted facts,
    and safety-critical context needed for informed responses.
    No parameters required.
    """
    app = _get_ctx(ctx)
    app.primed = True
    primer = await _generate_primer(app)
    all_facts = await app.memory.facts()
    active = [f for f in all_facts if f.status in ("promoted", "pinned")]
    facts_text = _format_facts(active)
    return f"{primer}\n\n{facts_text}"


@mcp.tool()
async def pin(
    trace_id: str,
    ctx: Ctx,
) -> PersonaFact:
    """Promote a memory to a permanent persona fact.

    Use for stable truths: allergies, relationships, preferences.
    Pinned facts are always surfaced during relevant recall.

    Args:
        trace_id: ID of the memory trace to pin.
    """
    try:
        return await _get_memory(ctx).pin(trace_id)
    except TraceNotFoundError:
        logger.exception("Trace not found for pin")
        raise
    except MemorySDKError:
        logger.exception("SDK error in pin")
        raise


@mcp.tool()
async def unpin(
    fact_id: str,
    ctx: Ctx,
) -> str:
    """Remove a persona fact that is no longer accurate.

    Args:
        fact_id: ID of the persona fact to remove.
    """
    try:
        deleted = await _get_memory(ctx).unpin(fact_id)
        if deleted:
            return f"Persona fact {fact_id} unpinned."
        return f"Persona fact {fact_id} not found."
    except MemorySDKError as exc:
        logger.exception("SDK error in unpin")
        return f"Failed to unpin fact {fact_id}: {exc}"


@mcp.tool()
async def forget(
    trace_id: str,
    ctx: Ctx,
) -> str:
    """Remove an incorrect or irrelevant memory.

    Args:
        trace_id: ID of the memory trace to forget.
    """
    try:
        await _get_memory(ctx).forget(trace_id)
        return f"Trace {trace_id} forgotten."
    except TraceNotFoundError:
        return f"Trace {trace_id} not found."
    except MemorySDKError as exc:
        logger.exception("SDK error in forget")
        return f"Failed to forget trace {trace_id}: {exc}"


def _format_remember_result(trace: MemoryTrace) -> str:
    """Format a stored trace as a compact confirmation.

    Returns the full trace ID so pin/forget can reference it.
    Previous short-ID format (8 chars) broke downstream lookups.
    """
    entities = trace.pattern.get("entities", [])
    concepts = trace.pattern.get("concepts", [])
    sig = trace.significance
    parts = [f"Remembered [{trace.id}] (significance {sig:.2f}"]
    if entities:
        parts.append(f", {len(entities)} entities")
    if concepts:
        parts.append(f", {len(concepts)} concepts")
    else:
        parts.append(", extraction degraded")
    parts.append(")")
    return "".join(parts)


def _format_thoughts(thoughts: list[Thought]) -> str:
    """Format thoughts with IMPORTANT CONTEXT section for persona facts."""
    important: list[str] = []
    regular: list[str] = []

    for thought in thoughts:
        score = f"{thought.relevance:.2f}"
        date = thought.trace.created_at.strftime("%Y-%m-%d")
        line = f"[score: {score}, {date}] {thought.reconstruction}"
        if thought.trace.pattern.get("persona_fact"):
            important.append(line)
        else:
            regular.append(line)

    parts: list[str] = []
    if important:
        parts.append("IMPORTANT CONTEXT:")
        parts.extend(f"- {line}" for line in important)
        parts.append("")
    if regular:
        parts.append("Related memories:")
        parts.extend(f"- {mem}" for mem in regular)

    return "\n".join(parts) if parts else "No relevant memories found."


def _format_facts(facts: list[PersonaFact]) -> str:
    """Format persona facts as compact text with temporal weight."""
    promoted = sum(1 for f in facts if f.status == "promoted")
    pinned = sum(1 for f in facts if f.status == "pinned")
    if not facts:
        return "No persona facts yet."
    lines = [f"PERSONA FACTS ({promoted} promoted, {pinned} pinned):"]
    for fact in facts:
        lines.append("")
        lines.append(_format_single_fact(fact))
    return "\n".join(lines)


def _format_single_fact(fact: PersonaFact) -> str:
    """Format one persona fact as a compact multi-line block."""
    short_id = fact.id[:8]
    age = humanize.naturaltime(now_utc() - fact.created_at)
    meta = f"{fact.category}, {fact.confidence:.2f}, {fact.status}, {age}"
    header = f"[{short_id}] {fact.subject} {fact.predicate} {fact.object}  ({meta})"
    if fact.content:
        return f"{header}\n  {fact.content}"
    return header


# -- Resources --


async def _generate_primer(app: AppContext) -> str:
    """Generate a two-level relational graph from persona facts."""
    facts = await app.memory.facts()
    active_facts = [f for f in facts if f.status in ("promoted", "pinned")]

    if not active_facts:
        return "No known relationships or facts yet."

    # Group facts by subject
    by_subject: dict[str, list[PersonaFact]] = {}
    for fact in active_facts:
        by_subject.setdefault(fact.subject, []).append(fact)

    lines: list[str] = ["KNOWN FACTS AND RELATIONSHIPS:"]
    for subject in sorted(by_subject):
        lines.append(f"  {subject}")
        for fact in by_subject[subject]:
            lines.append(f"    {fact.predicate}: {fact.object}")
            mechanical = f"{fact.subject} {fact.predicate} {fact.object}"
            if fact.content and fact.content != mechanical:
                lines.append(f"      ({fact.content})")

    return "\n".join(lines)


# -- Entrypoint --


def _load_prompt_files(
    extraction_path: str | None,
    token_path: str | None,
) -> None:
    """Load custom prompt files into config.

    Extraction prompt: entire file content replaces extraction.instructions.
    Token prompt: file split on '---' separator into system and user prompts.
    """
    if extraction_path:
        try:
            text = Path(extraction_path).read_text(encoding="utf-8").strip()
            config._set("extraction.instructions", text)
            logger.info("Loaded extraction prompt from %s", extraction_path)
        except OSError:
            logger.exception(
                "Failed to load extraction prompt from %s", extraction_path
            )
            raise

    if token_path:
        try:
            text = Path(token_path).read_text(encoding="utf-8").strip()
            parts = text.split("---")
            # Find the system and user prompt sections
            system_prompt = ""
            user_prompt = ""
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("## System Prompt"):
                    system_prompt = stripped.removeprefix("## System Prompt").strip()
                elif stripped.startswith("## User Prompt Template"):
                    user_prompt = stripped.removeprefix(
                        "## User Prompt Template"
                    ).strip()
            if system_prompt:
                config._set("recall_tokens.assessment_system_prompt", system_prompt)
            if user_prompt:
                config._set("recall_tokens.assessment_user_prompt", user_prompt)
            logger.info("Loaded token assessment prompt from %s", token_path)
        except OSError:
            logger.exception("Failed to load token prompt from %s", token_path)
            raise


def main() -> None:
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="recollect-mcp",
        description="Memory SDK MCP server",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--extraction-prompt",
        metavar="FILE",
        help="Path to custom extraction system prompt file",
    )
    parser.add_argument(
        "--token-prompt",
        metavar="FILE",
        help="Path to custom recall token assessment prompt file",
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        help="Path to JSON Lines log file (e.g. logs/mcp.jsonl)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging to stderr",
    )
    args = parser.parse_args()

    configure_logging(log_file=args.log_file, verbose=args.verbose)

    _load_prompt_files(args.extraction_prompt, args.token_prompt)

    if args.transport == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
