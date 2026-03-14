"""Integration test: PydanticAIProvider full lifecycle with rich output.

Run directly:  uv run --env-file .env python tests/integration/test_v3_pydantic_ai.py
Run as pytest:
  uv run --env-file .env pytest tests/integration/test_v3_pydantic_ai.py -v -m slow

Requires ANTHROPIC_API_KEY for Anthropic tests.
Requires OLLAMA_BASE_URL for Ollama tests.
"""

from __future__ import annotations

import asyncio
import os
import time
import urllib.error
import urllib.request
from typing import Any, cast

import pytest
from recollect.exceptions import ExtractionError  # noqa: F401
from recollect.extraction import PatternExtractor
from recollect.llm.types import CompletionParams, ExtractionResult, Message
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

try:
    import pydantic_ai as _pydantic_ai  # noqa: F401

    _has_pydantic_ai = True
except ImportError:
    _has_pydantic_ai = False

console = Console()

# -- Config --

ANTHROPIC_MODEL = "anthropic:claude-haiku-4-5-20251001"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.87.34:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-5-local-ubuntu:latest")
PYDANTIC_AI_OLLAMA_MODEL = f"ollama:{OLLAMA_MODEL}"
OLLAMA_MAX_TOKENS = 60000

TEST_INPUT = (
    "My colleague Sarah mentioned she's severely allergic to shellfish. "
    "We need to keep this in mind when planning the team dinner next Friday "
    "at the new Thai restaurant downtown."
)

# -- Skip conditions --

skip_no_pydantic_ai = pytest.mark.skipif(
    not _has_pydantic_ai, reason="pydantic-ai not installed"
)
skip_no_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


def _ollama_available() -> bool:
    """Check if Ollama is running at the configured base URL."""
    base = OLLAMA_BASE_URL.removesuffix("/v1")
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=2)  # noqa: S310
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


skip_no_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason=f"Ollama not running at {OLLAMA_BASE_URL}",
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.asyncio,
    skip_no_pydantic_ai,
]


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, length: int = 120) -> str:
    """Truncate text to a maximum length with ellipsis."""
    if len(text) <= length:
        return text
    return text[:length] + "..."


def _print_provider_info(
    provider: Any,
    model_str: str,
    defaults: CompletionParams,
) -> None:
    """Step 1: Show provider construction details."""
    console.rule("[bold]Step 1: Provider Construction[/bold]")
    info_lines = [
        f"[bold]Model:[/bold]        {model_str}",
        "[bold]Provider:[/bold]     PydanticAIProvider",
        f"[bold]Max tokens:[/bold]   {defaults.max_tokens}",
        f"[bold]Temperature:[/bold]  {defaults.temperature}",
        f"[bold]Max retries:[/bold]  {provider._max_retries}",
        f"[bold]Timeout:[/bold]      {provider._timeout}s",
    ]
    console.print(Panel("\n".join(info_lines), title="Provider Info"))


def _print_message_mapping(
    messages: list[Message],
    system_prompt: str,
    user_prompt: str,
) -> None:
    """Step 2: Show how Message list maps to pydantic-ai format."""
    console.rule("[bold]Step 2: Message Mapping[/bold]")

    table = Table(title="Message List")
    table.add_column("Role", style="bold")
    table.add_column("Content", max_width=80)
    for msg in messages:
        table.add_row(msg.role, msg.content)
    console.print(table)

    console.print(f"\n[bold]System prompt:[/bold] {_truncate(system_prompt)}")
    console.print(f"[bold]User prompt:[/bold]   {_truncate(user_prompt)}")


def _print_agent_config(
    model_str: str,
    system_prompt: str,
    output_type: str,
    retries: int,
    model_settings: dict[str, Any],
) -> None:
    """Step 3: Show what pydantic-ai Agent receives."""
    console.rule("[bold]Step 3: Agent Configuration[/bold]")
    info_lines = [
        f"[bold]model:[/bold]          {model_str}",
        f"[bold]system_prompt:[/bold]  {_truncate(system_prompt, 300)}",
        f"[bold]output_type:[/bold]    {output_type}",
        f"[bold]retries:[/bold]        {retries}",
        f"[bold]model_settings:[/bold] {model_settings}",
    ]
    console.print(Panel("\n".join(info_lines), title="Agent Constructor Args"))


def _print_result(
    duration: float,
    usage: Any,
    output_type_name: str,
    output: str | None = None,
) -> None:
    """Step 4/5: Show timing, token usage, and output."""
    console.rule("[bold]Step 4: LLM Result[/bold]")

    console.print(f"[bold]Duration:[/bold] {duration:.2f}s")
    console.print(f"[bold]Output type:[/bold] {output_type_name}")

    table = Table(title="Token Usage")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Input tokens", str(usage.input_tokens))
    table.add_row("Output tokens", str(usage.output_tokens))
    table.add_row("Cache read tokens", str(usage.cache_read_tokens))
    table.add_row("Cache write tokens", str(usage.cache_write_tokens))
    table.add_row("Requests", str(usage.requests))
    console.print(table)

    if output is not None:
        console.print(Panel(output, title="Output"))


def _print_extraction_result(result: ExtractionResult) -> None:
    """Step 5: Show ExtractionResult as a hierarchical tree."""
    console.rule("[bold]Step 5: Structured Extraction Result[/bold]")

    tree = Tree("ExtractionResult")
    tree.add(f"fact_type: {result.fact_type}")
    tree.add(f"significance: {result.significance}")
    tree.add(f"emotional_valence: {result.emotional_valence}")

    concepts_branch = tree.add("Concepts")
    for concept in result.concepts:
        concepts_branch.add(concept)

    entities_branch = tree.add("Entities")
    for entity in result.entities:
        entities_branch.add(
            f"{entity.name} ({entity.entity_type}, {entity.confidence:.2f})"
        )

    relations_branch = tree.add("Relations")
    for rel in result.relations:
        relations_branch.add(
            f"{rel.source} -[{rel.relation}]-> {rel.target} "
            f"({rel.category}, {rel.confidence:.2f})"
        )

    console.print(tree)


# ---------------------------------------------------------------------------
# Lifecycle runner
# ---------------------------------------------------------------------------


async def _run_lifecycle(
    model_str: str,
    defaults: CompletionParams,
    label: str,
) -> None:
    """Run the full PydanticAIProvider lifecycle with rich output."""
    from pydantic_ai import Agent
    from pydantic_ai.settings import ModelSettings
    from recollect.llm.pydantic_ai import PydanticAIProvider

    max_tokens = defaults.max_tokens

    console.print()
    console.rule(f"[bold green]{label} Lifecycle[/bold green]")
    console.print()

    # -- Step 1: Provider construction --
    provider = PydanticAIProvider(model=model_str, defaults=defaults)
    _print_provider_info(provider, model_str, defaults)

    # -- Step 2: Basic complete() call --
    basic_messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Summarize in one sentence: " + TEST_INPUT),
    ]
    system_part = "\n".join(m.content for m in basic_messages if m.role == "system")
    user_part = "\n".join(m.content for m in basic_messages if m.role != "system")
    _print_message_mapping(basic_messages, system_part, user_part)

    # -- Step 3: Agent config for basic call --
    basic_settings = {
        "max_tokens": max_tokens,
        "temperature": defaults.temperature,
    }
    _print_agent_config(
        model_str, system_part, "str", provider._max_retries, basic_settings
    )

    # -- Step 4: Run basic complete() --
    start = time.perf_counter()
    basic_output = await provider.complete(basic_messages)
    basic_duration = time.perf_counter() - start

    # Run directly through Agent to capture usage for the basic call
    basic_agent: Agent[None, str] = Agent(
        model_str,
        system_prompt=system_part,
        output_type=str,
        retries=provider._max_retries,
    )
    basic_result = await basic_agent.run(
        user_part, model_settings=cast(ModelSettings, basic_settings)
    )
    _print_result(basic_duration, basic_result.usage(), "str", output=basic_output)

    # -- Step 5: Structured extraction --
    console.print()
    console.rule("[bold]Structured Extraction Pipeline[/bold]")
    console.print()

    extractor = PatternExtractor(provider, max_tokens=max_tokens)
    system_prompt_text = extractor._build_prompt()
    extraction_messages = [
        Message(role="system", content=system_prompt_text),
        Message(role="user", content=TEST_INPUT),
    ]

    ext_system = "\n".join(m.content for m in extraction_messages if m.role == "system")
    ext_user = "\n".join(m.content for m in extraction_messages if m.role != "system")
    _print_message_mapping(extraction_messages, ext_system, ext_user)

    settings = cast(
        ModelSettings,
        {"max_tokens": max_tokens, "temperature": 0.0},
    )
    _print_agent_config(
        model_str, ext_system, "ExtractionResult", provider._max_retries, dict(settings)
    )

    # Run directly through pydantic-ai Agent to get full result with usage
    agent: Agent[None, ExtractionResult] = Agent(
        model_str,
        system_prompt=ext_system,
        output_type=ExtractionResult,
        retries=provider._max_retries,
    )

    start = time.perf_counter()
    result = await agent.run(ext_user, model_settings=settings)
    duration = time.perf_counter() - start

    _print_result(duration, result.usage(), "ExtractionResult")
    _print_extraction_result(result.output)

    console.print()
    console.print(f"[green]{label} lifecycle complete.[/green]")
    console.print()


# ---------------------------------------------------------------------------
# Pytest test classes
# ---------------------------------------------------------------------------


@skip_no_anthropic_key
class TestPydanticAIAnthropic:
    """Full lifecycle through Anthropic via pydantic-ai."""

    async def test_full_lifecycle(self) -> None:
        await _run_lifecycle(
            ANTHROPIC_MODEL,
            CompletionParams(max_tokens=1024, temperature=0.0),
            "Anthropic",
        )


@skip_no_ollama
class TestPydanticAIOllama:
    """Full lifecycle through Ollama via pydantic-ai."""

    async def test_full_lifecycle(self) -> None:
        os.environ.setdefault("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        await _run_lifecycle(
            PYDANTIC_AI_OLLAMA_MODEL,
            CompletionParams(max_tokens=OLLAMA_MAX_TOKENS, temperature=0.0),
            "Ollama",
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


async def _main() -> None:
    """Run all available lifecycles."""
    if not _has_pydantic_ai:
        console.print("[bold red]pydantic-ai is not installed. Exiting.[/bold red]")
        return

    if os.environ.get("ANTHROPIC_API_KEY"):
        await _run_lifecycle(
            ANTHROPIC_MODEL,
            CompletionParams(max_tokens=1024, temperature=0.0),
            "Anthropic",
        )
    else:
        console.print("[dim]Skipping Anthropic (no ANTHROPIC_API_KEY)[/dim]")

    if _ollama_available():
        os.environ.setdefault("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        await _run_lifecycle(
            PYDANTIC_AI_OLLAMA_MODEL,
            CompletionParams(max_tokens=OLLAMA_MAX_TOKENS, temperature=0.0),
            "Ollama",
        )
    else:
        console.print(f"[dim]Skipping Ollama (not running at {OLLAMA_BASE_URL})[/dim]")


if __name__ == "__main__":
    asyncio.run(_main())
