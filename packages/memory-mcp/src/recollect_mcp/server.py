"""MCP server for Recollect.

Exposes CognitiveMemory as MCP tools and resources.
Supports stdio and streamable-http transports.

6 tools: remember, recall, reflect, pin, unpin, forget
3 resources: primer, facts, health
"""

import logging
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import humanize
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resources import FunctionResource
from recollect.config import config
from recollect.core import CognitiveMemory, _estimate_tokens
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
    SURFACING_STATUSES,
    ExperienceResult,
    HealthReport,
    MemoryTrace,
    PersonaFact,
    SurfacedFacts,
    Thought,
    ThoughtWindow,
)
from recollect.worker import ConsolidationWorker

from recollect_mcp import __version__

logger = logging.getLogger(__name__)

# Type alias for our Context
Ctx = Context["AppContext", Any]

_STDIO_CONNECTION = "stdio"
_SESSION_HEADER = "mcp-session-id"
_SHORT_ID_LEN = 8
_ID_INPUT = re.compile(r"[0-9A-Za-z_-]{8,}")

_NO_MATCH = "No relevant memories found."
_BUDGET_EXHAUSTED = (
    "Memories matched, but none fits token_budget. Raise token_budget and recall again."
)
_SAFETY_PENDING = (
    "Safety context is pending and does not fit token_budget. "
    "Raise token_budget and recall again."
)
# Fixed text: an exception message is unbounded and stays in the server log.
_STORAGE_FAILED = "Recall failed: storage error. Details are in the server log."
_RECALL_FAILED = "Recall failed: memory error. Details are in the server log."
# The write succeeded: a failure reply would invite a second write of the memory.
_SURFACING_FAILED = (
    "Related persona facts could not be loaded: storage error. "
    "The memory is stored; do not store it again."
)
_SURFACING_PARTIAL = (
    "Other related persona facts could not be loaded: storage error. "
    "The memory is stored; do not store it again."
)
_MIN_RECALL_BUDGET = max(
    _estimate_tokens(n)
    for n in (
        _NO_MATCH,
        _BUDGET_EXHAUSTED,
        _SAFETY_PENDING,
        _STORAGE_FAILED,
        _RECALL_FAILED,
    )
)


@dataclass
class AppContext:
    """Lifespan context holding memory and worker instances."""

    memory: CognitiveMemory
    worker: ConsolidationWorker
    user_id: str
    _sessions: dict[str, str] = field(default_factory=dict)
    # Connection keys that had reflect or a delivered safety net. Keyed
    # because streamable-http shares this context across every client.
    primed: set[str] = field(default_factory=set)


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
    server: MCPServer[AppContext],
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
    worker = ConsolidationWorker(memory)
    worker.start()
    user_id = config.server_user_id
    if not user_id:
        raise RuntimeError(
            "MEMORY_USER_ID is required. "
            "Set it via environment variable or server.user_id in config."
        )
    logger.info("recollect-mcp server started, user_id=%s", user_id)
    app = AppContext(memory=memory, worker=worker, user_id=user_id)

    async def _primer() -> str:
        return await _generate_primer(app)

    async def _facts() -> str:
        all_facts = await _surfacing_facts(app)
        active = [f for f in all_facts if f.status in ("promoted", "pinned")]
        return _format_facts(active)

    async def _health() -> HealthReport:
        return await app.memory.health_report(user_id=app.user_id or None)

    mcp.add_resource(
        FunctionResource(
            uri="memory://primer",
            name="primer",
            description="Known relationships and key facts as a relational graph.",
            fn=_primer,
        )
    )
    mcp.add_resource(
        FunctionResource(
            uri="memory://facts",
            name="facts",
            description="Promoted and pinned persona facts.",
            fn=_facts,
        )
    )
    mcp.add_resource(
        FunctionResource(
            uri="memory://health",
            name="health",
            description="Lifecycle counts by status and violated invariants.",
            fn=_health,
        )
    )

    try:
        yield app
    finally:
        worker.stop()
        await memory.close()
        logger.info("recollect-mcp server stopped")


async def _surfacing_facts(app: AppContext) -> list[PersonaFact]:
    # Status-filtered in storage and unbounded: every always-available
    # surface reads the whole promoted+pinned graph, never a recency page.
    return await app.memory.facts(
        user_id=app.user_id or None, statuses=SURFACING_STATUSES, limit=None
    )


def _get_ctx(ctx: Ctx) -> AppContext:
    return ctx.request_context.lifespan_context


def _get_memory(ctx: Ctx) -> CognitiveMemory:
    return _get_ctx(ctx).memory


def _connection_key(ctx: Ctx) -> str | None:
    """Identity of the calling connection; None when the transport has none.

    stdio serves one client per process. Streamable HTTP carries
    `Mcp-Session-Id` on session-based connections only: a request without
    one (protocol 2026-07-28 single exchange) has no connection to prime.
    """
    headers = ctx.headers
    if headers is None:
        return _STDIO_CONNECTION
    return headers.get(_SESSION_HEADER)


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


def _short_id(full_id: str) -> str:
    return full_id[:_SHORT_ID_LEN]


def _id_prefix(raw: str) -> str | None:
    """A model-supplied id without pasted brackets; None under 8 characters."""
    candidate = raw.strip().strip("[]").strip()
    return candidate if _ID_INPUT.fullmatch(candidate) else None


def _opening(text: str | None, width: int = 60) -> str:
    text = text or ""
    return text if len(text) <= width else f"{text[:width].rstrip()}..."


async def _resolve_trace(app: AppContext, raw: str) -> MemoryTrace | str:
    """The one trace `raw` names for the server's user, else the refusal text.

    Unique or refuse: forget does not auto-revive, so an ambiguous id lists
    its candidates instead of choosing one.
    """
    prefix = _id_prefix(raw)
    if prefix is None:
        return (
            f"'{raw}' is not a memory id. Pass the 8-character id printed by "
            "recall or remember, or the full id."
        )
    matches = await app.memory.traces_by_id_prefix(
        prefix, user_id=app.user_id or None
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return f"No memory trace matches id {prefix}."
    lines = [f"Id {prefix} matches more than one memory. Repeat with one full id:"]
    lines.extend(
        f"[{t.id}] ({t.created_at:%Y-%m-%d}) {_opening(t.content)}" for t in matches
    )
    return "\n".join(lines)


async def _resolve_fact(app: AppContext, raw: str) -> PersonaFact | str:
    """The one persona fact `raw` names for the server's user, else the refusal."""
    prefix = _id_prefix(raw)
    if prefix is None:
        return (
            f"'{raw}' is not a persona fact id. Pass the 8-character id printed "
            "by reflect or pin, or the full id."
        )
    matches = await app.memory.facts_by_id_prefix(prefix, user_id=app.user_id or None)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return f"Persona fact {prefix} not found."
    lines = [f"Id {prefix} matches more than one fact. Repeat with one full id:"]
    lines.extend(f"[{f.id}] {f.subject} {f.predicate} {f.object}" for f in matches)
    return "\n".join(lines)


_INSTRUCTIONS = """\
Cognitive memory system for personal AI assistants.

You have 6 tools:

REMEMBER -- Store an experience in memory.
  Write complete, contextually-anchored statements that include relational
  identifiers. Use "Alex's mother Sarah" not just "Sarah" when names could
  refer to different people. What you write is what the system indexes --
  ambiguous input produces ambiguous recall. One memory holds one fact or
  one situation that is useful on its own, with the context needed to find
  it. Split unrelated topics into separate memories.

RECALL -- Retrieve relevant memories for a query.
  Returns persona facts as IMPORTANT CONTEXT followed by related memories.
  Be specific in your queries. The more precise the query, the better the
  results. You do not need to pass user_id or session_id -- the server
  manages these automatically.

PIN -- Promote a memory to a permanent persona fact.
  Use this for stable truths about the user: allergies, relationships,
  preferences. Pinned facts are always surfaced during relevant recall.

UNPIN -- Archive a persona fact that is no longer accurate.
  The fact stops surfacing in recall and reflect.

FORGET -- Forget an incorrect or irrelevant memory.
  The memory trace stops surfacing in recall and never auto-revives;
  the row is retained, not hard-deleted. Derived facts archive with
  it. Safety-critical (health/dietary/constraint) and pinned facts
  are retained unless force=true.

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

mcp = MCPServer(
    name="recollect-mcp",
    instructions=_INSTRUCTIONS,
    version=__version__,
    lifespan=app_lifespan,
)


# -- Tools --


@mcp.tool()
async def remember(
    content: str,
    ctx: Ctx,
    context: dict[str, Any] | None = None,
    supersedes: list[str] | None = None,
) -> str:
    """Store an experience in memory.

    Write complete, contextually-anchored statements. Include relational
    identifiers like "Alex's mother Sarah" rather than bare names when
    the same name could refer to different people.

    One memory holds one fact or one situation that is useful on its own,
    with the context needed to find it. Split unrelated topics into
    separate memories.

    If you haven't called reflect this session, do so before using this tool.

    Args:
        content: The experience to remember. Be specific and complete.
        context: Optional metadata dict attached to the memory trace.
        supersedes: Ids of earlier memories this one replaces: the 8-character
            ids printed by recall or remember, or full ids; each is forgotten
            as part of this write.
    """
    app = _get_ctx(ctx)
    try:
        # Resolved before any write: a refused id leaves the store untouched.
        originals = await _resolve_traces(app, supersedes or [])
        if isinstance(originals, str):
            return originals
        session_id = await _ensure_session(ctx)
        result = await app.memory.experience_result(
            content,
            context=context,
            session_id=session_id,
            user_id=app.user_id or None,
            supersedes=[t.id for t in originals],
        )
        echo = _write_echo(result, originals)
        surfaced = await _surfaced_facts(app, result.trace)
        if surfaced is None:
            stored = _format_remember_result(result.trace, echo=echo)
            return f"{stored}\n\n{_SURFACING_FAILED}"
        stored = _format_remember_result(result.trace, surfaced.facts, echo=echo)
        if surfaced.lookup_failed:
            return f"{stored}\n\n{_SURFACING_PARTIAL}"
        return stored
    except TraceNotFoundError as exc:
        return f"{exc}."
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
    Each related memory starts with its id in brackets; pass it to pin
    or forget.

    If you haven't called reflect this session, do so before using this tool.

    Args:
        query: What to recall. Be specific.
        token_budget: Maximum tokens in the response, at least 23 (default 2000).
    """
    # Rejected before the one-shot is claimed, as ToolError: its text reaches
    # the client, and a rejected call must not consume the safety net.
    if not query.strip():
        raise ToolError("query must be a non-empty string")
    if token_budget < _MIN_RECALL_BUDGET:
        raise ToolError(f"token_budget must be at least {_MIN_RECALL_BUDGET}")
    app = _get_ctx(ctx)
    # Persona facts surface through the recall-floored think_about output;
    # the full promoted+pinned graph is reflect's job. On the first
    # (unreflected) recall, prepend a narrow safety net -- pinned + recall
    # safety-bypass facts -- so safety-critical context is never dropped
    # when an agent skips reflect.
    key = _connection_key(ctx)
    surface_safety = key not in app.primed
    if surface_safety and key is not None:
        # Claimed before the await so concurrent first calls fire the net once;
        # restored on failure below so a failed delivery re-arms the one-shot.
        app.primed.add(key)
    try:
        safety_facts: Sequence[PersonaFact] = (
            await _safety_net_facts(app) if surface_safety else ()
        )
        if _safety_net_overflows(safety_facts, token_budget):
            # Re-armed, no retrieval: the next recall that holds the net prints it.
            _rearm_safety_net(app, key)
            return _SAFETY_PENDING
        # The budget bounds the printed text, safety net included.
        window = await app.memory.think_about_window(
            query,
            token_budget=token_budget,
            user_id=app.user_id or None,
            render=lambda w: _format_thoughts(w, safety_facts=safety_facts),
        )
        return _window_text(window, safety_facts)
    except StorageError:
        if surface_safety:
            _rearm_safety_net(app, key)
        logger.exception("Storage error in recall")
        return _STORAGE_FAILED
    except MemorySDKError:
        if surface_safety:
            _rearm_safety_net(app, key)
        logger.exception("Unexpected SDK error in recall")
        return _RECALL_FAILED


def _rearm_safety_net(app: AppContext, key: str | None) -> None:
    if key is not None:
        app.primed.discard(key)


def _safety_net_overflows(
    safety_facts: Sequence[PersonaFact], token_budget: int
) -> bool:
    if not safety_facts:
        return False
    net = _format_thoughts([], safety_facts=safety_facts)
    return _estimate_tokens(net) > token_budget


def _window_text(window: ThoughtWindow, safety_facts: Sequence[PersonaFact]) -> str:
    if window.thoughts or safety_facts:
        return _format_thoughts(window.thoughts, safety_facts=safety_facts)
    return _BUDGET_EXHAUSTED if window.withheld else _NO_MATCH


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
    key = _connection_key(ctx)
    if key is not None:
        app.primed.add(key)
    primer = await _generate_primer(app)
    all_facts = await _surfacing_facts(app)
    active = [f for f in all_facts if f.status in ("promoted", "pinned")]
    facts_text = _format_facts(active)
    return f"{primer}\n\n{facts_text}"


@mcp.tool()
async def pin(
    trace_id: str,
    ctx: Ctx,
) -> str:
    """Promote a memory's extracted relations to permanent persona facts.

    Use for stable truths: allergies, relationships, preferences.
    Pinned facts are always surfaced during relevant recall.

    Args:
        trace_id: Id of the memory trace: the 8-character id printed by
            recall or remember, or the full id.
    """
    app = _get_ctx(ctx)
    try:
        target = await _resolve_trace(app, trace_id)
        if isinstance(target, str):
            # Returned, not raised: the refusal quotes memory text, and a
            # raised refusal would reach the exception log below.
            return target
        return _format_pin_result(await app.memory.pin(target.id))
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
    """Archive a persona fact that is no longer accurate.

    The fact stops surfacing in recall and reflect; the row is
    retained for audit.

    Args:
        fact_id: Id of the persona fact: the 8-character id printed by
            reflect or pin, or the full id.
    """
    app = _get_ctx(ctx)
    try:
        target = await _resolve_fact(app, fact_id)
        if isinstance(target, str):
            return target
        if await app.memory.unpin(target.id):
            triple = f"{target.subject} {target.predicate} {target.object}"
            return (
                f"Persona fact {target.id} archived ({triple}); "
                "it will no longer surface."
            )
        return f"Persona fact {target.id} not found."
    except MemorySDKError as exc:
        logger.exception("SDK error in unpin")
        return f"Failed to unpin fact {fact_id}: {exc}"


@mcp.tool()
async def forget(
    trace_id: str,
    ctx: Ctx,
    force: bool = False,
) -> str:
    """Forget an incorrect or irrelevant memory.

    The trace stops surfacing in recall and never auto-revives; the
    row is retained, not hard-deleted. Derived facts archive with it,
    except safety-critical (health/dietary/constraint) and pinned
    facts, which are retained unless force=true.

    Args:
        trace_id: Id of the memory trace: the 8-character id printed by
            recall or remember, or the full id.
        force: Also archive safety-critical and pinned derived facts.
    """
    app = _get_ctx(ctx)
    try:
        target = await _resolve_trace(app, trace_id)
        if isinstance(target, str):
            return target
        result = await app.memory.forget(target.id, force=force)
    except TraceNotFoundError:
        return f"Trace {trace_id} not found."
    except MemorySDKError as exc:
        logger.exception("SDK error in forget")
        return f"Failed to forget trace {trace_id}: {exc}"
    parts = [f'Trace {target.id} forgotten ("{_opening(target.content)}").']
    if result.archived_fact_ids:
        parts.append(f"{len(result.archived_fact_ids)} derived facts archived.")
    if result.retained_facts:
        kinds = ", ".join(
            sorted({f.category for f in result.retained_facts})
        )
        parts.append(
            f"{len(result.retained_facts)} facts retained ({kinds}); "
            "repeat with force=true to also suppress them."
        )
    return " ".join(parts)


async def _surfaced_facts(
    app: AppContext, trace: MemoryTrace
) -> SurfacedFacts | None:
    """Write-time surfaced facts; None when the safety lookup failed after the write."""
    try:
        return await app.memory.surface_relevant_facts_result(
            trace, user_id=app.user_id or None
        )
    except StorageError:
        logger.exception(
            "Fact surfacing failed after remember stored %s", _short_id(trace.id)
        )
        return None


async def _resolve_traces(
    app: AppContext, raws: Sequence[str]
) -> list[MemoryTrace] | str:
    """Every trace `raws` names, else the first refusal text."""
    traces: list[MemoryTrace] = []
    for raw in raws:
        target = await _resolve_trace(app, raw)
        if isinstance(target, str):
            return target
        if target.status == "forgotten":
            # A retracted memory has nothing to replace: the SDK's refusal,
            # ahead of the session write.
            return f"Trace {raw} not found."
        traces.append(target)
    return traces


_RETAINED_HINT = "Retained facts stay live; remove one with unpin and its id."
# forget resumes on either side of the trace flip, so one follow-up serves both.
_UNSUPERSEDED_HINT = (
    "This memory is stored; do not store it again. "
    "Run forget on each memory listed as not superseded."
)


def _write_echo(
    result: ExperienceResult, originals: Sequence[MemoryTrace]
) -> list[str]:
    """What the write did beside storing the trace, one line each."""
    contents = {t.id: t.content for t in originals}
    lines: list[str] = []
    for outcome in result.superseded:
        lines.append(
            f"Superseded [{_short_id(outcome.trace_id)}] "
            f'("{_opening(contents.get(outcome.trace_id))}"): '
            f"{len(outcome.archived_fact_ids)} derived facts archived."
        )
        lines.extend(f"  {_retained_line(f)}" for f in outcome.retained_facts)
    if any(outcome.retained_facts for outcome in result.superseded):
        lines.append(_RETAINED_HINT)
    lines.extend(
        f"Not superseded [{_short_id(trace_id)}] "
        f'("{_opening(contents.get(trace_id))}"): storage error.'
        for trace_id in result.unsuperseded
    )
    if result.unsuperseded:
        lines.append(_UNSUPERSEDED_HINT)
    lines.extend(
        f"Persona fact {f.subject} {f.predicate} {f.object} was not stored: "
        "storage error. The memory itself is stored."
        for f in result.unstored_facts
    )
    return lines


def _retained_line(fact: PersonaFact) -> str:
    tag = "pinned" if fact.status == "pinned" else fact.category
    triple = f"{fact.subject} {fact.predicate} {fact.object}"
    return f"[{_short_id(fact.id)}] [{tag}] {triple}"


def _format_remember_result(
    trace: MemoryTrace,
    surfaced_facts: list[PersonaFact] | None = None,
    *,
    echo: Sequence[str] = (),
) -> str:
    """Format a stored trace as a compact confirmation.

    Prints the short trace id; pin and forget resolve it. The echo lines
    follow the confirmation line. Write-time surfaced persona facts follow
    as an IMPORTANT CONTEXT block.
    """
    entities = trace.pattern.get("entities", [])
    concepts = trace.pattern.get("concepts", [])
    sig = trace.significance
    parts = [f"Remembered [{_short_id(trace.id)}] (significance {sig:.2f}"]
    if entities:
        parts.append(f", {len(entities)} entities")
    if concepts:
        parts.append(f", {len(concepts)} concepts")
    else:
        parts.append(", extraction degraded")
    parts.append(")")
    confirmation = "\n".join(["".join(parts), *echo])
    if not surfaced_facts:
        return confirmation
    block = ["IMPORTANT CONTEXT:"]
    block.extend(_format_single_fact(f) for f in surfaced_facts)
    return f"{confirmation}\n\n" + "\n".join(block)


_RECALL_SAFETY_NET_CATEGORIES = frozenset({"health", "dietary"})


async def _safety_net_facts(app: AppContext) -> list[PersonaFact]:
    """Pinned + recall-safety-bypass facts -- the skip-reflect always-on surface.

    Mirrors the recall floor's exemptions (pinned floor-exemption + {health,
    dietary} bypass) so the unreflected path never silently drops
    safety-critical context. Candidates and archived facts excluded.
    """
    facts = await _surfacing_facts(app)
    return [
        f
        for f in facts
        if f.status in ("promoted", "pinned")
        and (f.status == "pinned" or f.category in _RECALL_SAFETY_NET_CATEGORIES)
    ]


def _safety_net_line(fact: PersonaFact) -> str:
    triple = f"{fact.subject} {fact.predicate} {fact.object}"
    tag = "pinned" if fact.status == "pinned" else fact.category
    if fact.content and fact.content != triple:
        return f"[{tag}] {triple} -- {fact.content}"
    return f"[{tag}] {triple}"


def _format_thoughts(
    thoughts: list[Thought],
    *,
    safety_facts: Sequence[PersonaFact] = (),
) -> str:
    """Format thoughts with IMPORTANT CONTEXT section for persona facts.

    safety_facts are the skip-reflect safety net (pinned + recall safety-bypass);
    surfaced under IMPORTANT CONTEXT ahead of the gated persona facts.
    """
    # Dedup the safety net against facts the gate already surfaced: a persona
    # thought's reconstruction embeds its "subject predicate object" triple.
    gated_recons = [
        t.reconstruction for t in thoughts if t.trace.pattern.get("persona_fact")
    ]
    important: list[str] = [
        _safety_net_line(f)
        for f in safety_facts
        if not any(
            f"{f.subject} {f.predicate} {f.object}" in recon for recon in gated_recons
        )
    ]
    regular: list[str] = []

    for thought in thoughts:
        score = f"{thought.relevance:.2f}"
        date = thought.trace.created_at.strftime("%Y-%m-%d")
        line = f"[score: {score}, {date}] {thought.reconstruction}"
        if thought.trace.pattern.get("persona_fact"):
            # A fact thought rides a synthetic trace: its id names no row.
            important.append(line)
        else:
            regular.append(f"[{_short_id(thought.trace.id)}] {line}")

    parts: list[str] = []
    if important:
        parts.append("IMPORTANT CONTEXT:")
        parts.extend(f"- {line}" for line in important)
        parts.append("")
    if regular:
        parts.append("Related memories:")
        parts.extend(f"- {mem}" for mem in regular)

    return "\n".join(parts) if parts else _NO_MATCH


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


def _format_pin_result(facts: list[PersonaFact]) -> str:
    """Confirm pinned facts as compact text -- never serialize embeddings.

    pin() returns PersonaFact models carrying 768-float embeddings; returning
    them raw overflowed the MCP result cap (adr-pin-upsert-semantics).
    """
    if not facts:
        return "Nothing to pin."
    lines = [f"Pinned {len(facts)} persona fact(s):"]
    for fact in facts:
        lines.append("")
        lines.append(_format_single_fact(fact))
    return "\n".join(lines)


def _format_single_fact(fact: PersonaFact) -> str:
    """Format one persona fact as a compact multi-line block.

    Prints the short fact id; unpin resolves it.
    """
    age = humanize.naturaltime(now_utc() - fact.created_at)
    meta = f"{fact.category}, {fact.confidence:.2f}, {fact.status}, {age}"
    triple = f"{fact.subject} {fact.predicate} {fact.object}"
    header = f"[{_short_id(fact.id)}] {triple}  ({meta})"
    if fact.content:
        return f"{header}\n  {fact.content}"
    return header


# -- Resources --


async def _generate_primer(app: AppContext) -> str:
    """Generate a two-level relational graph from persona facts."""
    facts = await _surfacing_facts(app)
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
    extraction_template_path: str | None,
    extraction_instructions_path: str | None,
    situational_path: str | None,
) -> None:
    """Load custom prompt files into config.

    extraction-template: full template override (applies-to: extraction-template).
        Sets extraction.template_path; PatternExtractor reloads at construction.
    extraction-instructions (legacy --extraction-prompt): raw text appended as
        suffix to template. No header required.
    situational: combined system + user override (applies-to: situational).
        Sets recall_tokens.assessment_*_prompt slots.
    """
    from recollect.exceptions import PromptValidationError
    from recollect.prompts import load_prompt_file

    if extraction_template_path:
        loaded = load_prompt_file(extraction_template_path)
        if loaded.applies_to != "extraction-template":
            raise PromptValidationError(
                f"{extraction_template_path}: applies-to must be "
                f"'extraction-template', got '{loaded.applies_to}'"
            )
        config._set("extraction.template_path", extraction_template_path)
        logger.info(
            "Extraction template override: %s (version %s)",
            extraction_template_path,
            loaded.version,
        )

    if extraction_instructions_path:
        try:
            text = (
                Path(extraction_instructions_path).read_text(encoding="utf-8").strip()
            )
        except OSError:
            logger.exception(
                "Failed to load extraction instructions from %s",
                extraction_instructions_path,
            )
            raise
        config._set("extraction.instructions", text)
        logger.info(
            "Extraction instructions suffix: %s", extraction_instructions_path
        )

    if situational_path:
        loaded = load_prompt_file(situational_path)
        if loaded.applies_to != "situational":
            raise PromptValidationError(
                f"{situational_path}: applies-to must be 'situational', "
                f"got '{loaded.applies_to}'"
            )
        config._set("recall_tokens.assessment_system_prompt", loaded.system)
        config._set("recall_tokens.assessment_user_prompt", loaded.user)
        logger.info(
            "Situational prompt override: %s (version %s)",
            situational_path,
            loaded.version,
        )


def main() -> None:
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="recollect-mcp",
        description="MCP server for Recollect",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--extraction-template",
        metavar="FILE",
        help=(
            "Path to full extraction template override (.md with header, "
            "applies-to: extraction-template). Replaces packaged default."
        ),
    )
    parser.add_argument(
        "--extraction-prompt",
        metavar="FILE",
        help=(
            "Path to extraction instructions suffix (raw text, appended to "
            "template). Legacy; prefer --extraction-template for full overrides."
        ),
    )
    parser.add_argument(
        "--token-prompt",
        metavar="FILE",
        help=(
            "Path to situational prompt override (.md with header, applies-to: "
            "situational, ## System Prompt + ## User Prompt sections)."
        ),
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

    _load_prompt_files(
        args.extraction_template,
        args.extraction_prompt,
        args.token_prompt,
    )

    if args.transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=str(config.get("server.host", "127.0.0.1")),
            port=int(config.get("server.port", 8000)),
        )
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
