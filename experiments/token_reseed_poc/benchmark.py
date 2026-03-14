"""Iterative Token Re-seeding POC benchmark runner.

Chain discovery: tests whether iterative token re-seeding discovers traces
at the end of multi-hop entity chains that one-hop token activation misses.

Usage:
    uv run python experiments/token_reseed_poc/benchmark.py
    uv run python experiments/token_reseed_poc/benchmark.py \
        --provider ollama --llm-model ministral-3
    uv run python experiments/token_reseed_poc/benchmark.py -v
"""

# ruff: noqa: T201 UP031

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

# Add project root to path so 'experiments' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import logging

import rich.box
from recollect.log_setup import configure_logging
from rich.console import Console
from rich.table import Table

from experiments.token_reseed_poc.config import PocConfig
from experiments.token_reseed_poc.engine import PocEngine
from experiments.token_reseed_poc.retrieval import (
    RetrievalResult,
    baseline_recall,
    iterative_token_recall,
    token_recall,
)
from experiments.token_reseed_poc.scenarios import SCENARIO, Query

logger = logging.getLogger(__name__)
console = Console()

SHOW_WRITE_TIME_GROUPS = {"anchor", "chain_a", "chain_b", "chain_c", "chain_d"}


def parse_args() -> PocConfig:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Iterative Token Re-seeding POC: Chain Discovery"
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Base URL for Ollama API (e.g. http://192.168.87.34:11434/v1)",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help="pydantic-ai model string (e.g. ollama:ministral-3)",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name (default: nomic-ai/nomic-embed-text-v1.5-Q)",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help="Max tokens for LLM responses (default: 1024)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results for recall queries (default: 20)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Path to prompt markdown file for token assessment",
    )
    parser.add_argument(
        "--hop-decay",
        type=float,
        default=None,
        help="Score propagation decay per hop (default: 0.85)",
    )
    parser.add_argument(
        "--iter-max-rounds",
        type=int,
        default=None,
        help="Max re-seeding rounds (default: 3)",
    )
    parser.add_argument(
        "--iter-stability-threshold",
        type=float,
        default=None,
        help="Stability threshold to stop iterating (default: 0.95)",
    )
    parser.add_argument(
        "--iter-top-seeds",
        type=int,
        default=None,
        help="How many top results become seeds each round (default: 3)",
    )
    parser.add_argument(
        "--propagation-blend",
        type=float,
        default=None,
        help="Propagated_sim blend factor for token scores (default: 0.5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    config = PocConfig()

    if args.database_url is not None:
        config.database_url = args.database_url
    if args.llm_base_url is not None:
        config.llm_base_url = args.llm_base_url
    if args.llm is not None:
        config.llm = args.llm
    if args.embedding_model is not None:
        config.embedding_model = args.embedding_model
    if args.llm_max_tokens is not None:
        config.llm_max_tokens = args.llm_max_tokens
    if args.top_k is not None:
        config.recall_top_k = args.top_k
    if args.prompt is not None:
        config.prompt_path = args.prompt
    if args.hop_decay is not None:
        config.hop_decay = args.hop_decay
    if args.iter_max_rounds is not None:
        config.iter_max_rounds = args.iter_max_rounds
    if args.iter_stability_threshold is not None:
        config.iter_stability_threshold = args.iter_stability_threshold
    if args.iter_top_seeds is not None:
        config.iter_top_seeds = args.iter_top_seeds
    if args.propagation_blend is not None:
        config.propagation_blend = args.propagation_blend

    configure_logging(
        log_file="logs/poc-token-reseed.jsonl",
        file_level=logging.WARNING,  # TODO: restore to DEBUG after focused run
        verbose=args.verbose,
    )

    return config


def truncate(text: str, length: int = 55) -> str:
    """Truncate text with ellipsis if over length."""
    if len(text) <= length:
        return text
    return text[:length] + "..."


def pct(num: int, den: int) -> str:
    """Format percentage, handling zero denominator."""
    if den == 0:
        return "N/A"
    return "%d%%" % (num * 100 // den)


def compute_metrics(
    results: list[RetrievalResult],
    query: Query,
    trace_id_to_index: dict[str, int],
) -> tuple[int, int, int, int]:
    """Return (expected_found, expected_total, unexpected_found, unexpected_total)."""
    retrieved_indices = {
        trace_id_to_index[r.trace_id]
        for r in results
        if r.trace_id in trace_id_to_index
    }
    expected_found = len(set(query.expected_indices) & retrieved_indices)
    expected_total = len(query.expected_indices)
    unexpected_found = len(set(query.unexpected_indices) & retrieved_indices)
    unexpected_total = len(query.unexpected_indices)
    return (expected_found, expected_total, unexpected_found, unexpected_total)


def chain_end_found(
    results: list[RetrievalResult],
    query: Query,
    trace_id_to_index: dict[str, int],
) -> str:
    """Check if the chain-end traces (expected_iterative) were found."""
    retrieved_indices = {
        trace_id_to_index[r.trace_id]
        for r in results
        if r.trace_id in trace_id_to_index
    }
    found = set(query.expected_iterative) & retrieved_indices
    if len(found) == len(query.expected_iterative):
        return "YES"
    return "%d/%d" % (len(found), len(query.expected_iterative))


def _chain_color(value: str) -> str:
    """Wrap chain-end value in green/red markup."""
    if value == "YES":
        return "[green]%s[/green]" % value
    return "[red]%s[/red]" % value


def _make_result_table(
    title: str,
    results: list[RetrievalResult],
    query: Query,
    trace_id_to_index: dict[str, int],
    index_to_group: dict[int, str],
) -> Table:
    """Build a rich table for retrieval results."""
    table = Table(
        title=title,
        show_header=True,
        box=rich.box.ROUNDED,
        expand=True,
    )
    table.add_column("", width=1)
    table.add_column("Score", justify="right", style="bold", width=5)
    table.add_column("Source", width=20, no_wrap=True)
    table.add_column("Group", width=8, no_wrap=True)
    table.add_column("T#", justify="right", width=3)
    table.add_column("Content", no_wrap=True)

    for r in results[:10]:
        idx = trace_id_to_index.get(r.trace_id)
        grp = index_to_group.get(idx, "?") if idx is not None else "?"
        idx_str = str(idx) if idx is not None else "?"

        # Format source column
        if r.source == "vector":
            src = "(vector)"
        elif r.source.startswith("token_hop_"):
            src = "(%s: %s)" % (r.source, r.token_label)
        elif r.source in ("token", "token+vector"):
            src = "(token: %s)" % r.token_label
        else:
            src = "(%s)" % r.source

        # Determine row style
        if idx is not None and idx in query.expected_iterative:
            row_style = "bold green"
            marker = "*"
        elif idx is not None and idx in query.expected_indices:
            row_style = "green"
            marker = "*"
        elif grp == "hay":
            row_style = "dim"
            marker = ""
        else:
            row_style = ""
            marker = ""

        table.add_row(
            marker,
            "%.3f" % r.score,
            src,
            grp,
            idx_str,
            truncate(r.content),
            style=row_style,
        )

    return table


async def main() -> None:  # noqa: C901
    config = parse_args()
    engine = PocEngine(config)
    scenario = SCENARIO
    index_to_group = {i: m.group for i, m in enumerate(scenario.memories)}

    console.rule("[bold]Iterative Token Re-seeding POC: Chain Discovery")
    console.print(
        "llm=[cyan]%s[/]  top_k=[cyan]%d[/]"
        "  rounds=[cyan]%d[/]  stability=[cyan]%.2f[/]"
        "  seeds=[cyan]%d[/]  hop_decay=[cyan]%.2f[/]"
        "  blend=[cyan]%.2f[/]"
        % (
            config.llm,
            config.recall_top_k,
            config.iter_max_rounds,
            config.iter_stability_threshold,
            config.iter_top_seeds,
            config.hop_decay,
            config.propagation_blend,
        )
    )

    try:
        await engine.setup()

        # Clean tables
        async with engine.pool.acquire() as conn:
            await conn.execute(
                "TRUNCATE poc_token_stamps, poc_recall_tokens, poc_traces CASCADE"
            )

        # --- Store all memories ---
        total = len(scenario.memories)
        hay_count = sum(1 for m in scenario.memories if m.group == "hay")
        print("\n--- Storing %d memories (%d haystack) ---" % (total, hay_count))
        trace_ids: list[str] = []
        hay_stored = 0
        for i, memory in enumerate(scenario.memories):
            result = await engine.experience(
                memory.content,
                significance=memory.significance,
            )
            trace_ids.append(result.trace_id)

            if memory.group in SHOW_WRITE_TIME_GROUPS:
                wtr = result.write_time_recall
                print(
                    '  [%-8s]  T%-2d  "%s"'
                    % (memory.group, i, truncate(memory.content))
                )
                # Show what LLM saw and decided
                if wtr.related_found:
                    for ri, rel in enumerate(wtr.related_found):
                        print("              %d. %s" % (ri + 1, truncate(rel, 65)))
                if wtr.token_created:
                    if wtr.token_revised:
                        action = "revised"
                    elif wtr.token_reused:
                        action = "extended"
                    else:
                        action = "created"
                    print(
                        "              -> group %s: [%s]"
                        " (linked: %s)" % (action, wtr.token_label, wtr.linked_indices)
                    )
            else:
                hay_stored += 1
                print(
                    "\r  haystack: %d/%d" % (hay_stored, hay_count),
                    end="",
                    flush=True,
                )
        if hay_count > 0:
            print()  # newline after progress

        # Build trace_id -> index mapping
        trace_id_to_index = {tid: i for i, tid in enumerate(trace_ids)}

        # --- Run queries ---
        summaries: list[dict[str, object]] = []

        for qi, query in enumerate(scenario.queries):
            console.rule("[bold]Query %d: %s" % (qi + 1, query.text))
            console.print("[dim]%s[/dim]" % query.description)
            console.print(
                "[dim]Expected: %s[/dim]"
                % ", ".join("T%d" % idx for idx in query.expected_indices)
            )
            console.print(
                "[dim]Expected (iterative-only): %s[/dim]"
                % ", ".join("T%d" % idx for idx in query.expected_iterative)
            )
            console.print(
                "[dim]Unexpected: %s[/dim]"
                % ", ".join("T%d" % idx for idx in query.unexpected_indices)
            )

            query_embedding = await engine.embed(query.text)

            # --- Baseline ---
            baseline_results = await baseline_recall(
                engine.pool,
                query_embedding,
                top_k=config.recall_top_k,
                significance_weight=config.significance_weight,
                valence_weight=config.valence_weight,
            )
            console.print(
                _make_result_table(
                    "Baseline (vector only)",
                    baseline_results,
                    query,
                    trace_id_to_index,
                    index_to_group,
                )
            )

            # --- One-hop token recall ---
            token_results = await token_recall(
                engine.pool,
                query_embedding,
                top_k=config.recall_top_k,
                strength_threshold=config.token_strength_threshold,
                reinforce_boost=config.token_reinforce_boost,
                hop_decay=config.hop_decay,
                significance_weight=config.significance_weight,
                valence_weight=config.valence_weight,
                propagation_blend=config.propagation_blend,
            )
            console.print(
                _make_result_table(
                    "One-hop token recall",
                    token_results,
                    query,
                    trace_id_to_index,
                    index_to_group,
                )
            )

            # --- Iterative token recall ---
            iter_results, rounds = await iterative_token_recall(
                engine.pool,
                query_embedding,
                top_k=config.recall_top_k,
                strength_threshold=config.token_strength_threshold,
                reinforce_boost=config.token_reinforce_boost,
                hop_decay=config.hop_decay,
                max_rounds=config.iter_max_rounds,
                stability_threshold=config.iter_stability_threshold,
                top_seeds=config.iter_top_seeds,
                significance_weight=config.significance_weight,
                valence_weight=config.valence_weight,
                propagation_blend=config.propagation_blend,
            )
            console.print(
                _make_result_table(
                    "Iterative token recall (%d rounds)" % rounds,
                    iter_results,
                    query,
                    trace_id_to_index,
                    index_to_group,
                )
            )

            # --- Metrics ---
            b_metrics = compute_metrics(baseline_results, query, trace_id_to_index)
            t_metrics = compute_metrics(token_results, query, trace_id_to_index)
            i_metrics = compute_metrics(iter_results, query, trace_id_to_index)

            b_chain = chain_end_found(baseline_results, query, trace_id_to_index)
            t_chain = chain_end_found(token_results, query, trace_id_to_index)
            i_chain = chain_end_found(iter_results, query, trace_id_to_index)

            metrics_table = Table(
                title="Metrics",
                show_header=True,
                box=rich.box.SIMPLE_HEAVY,
            )
            metrics_table.add_column("Metric")
            metrics_table.add_column("Baseline", justify="center")
            metrics_table.add_column("One-hop", justify="center")
            metrics_table.add_column("Iterative", justify="center")

            metrics_table.add_row(
                "Expected recall",
                "%d/%d" % (b_metrics[0], b_metrics[1]),
                "%d/%d" % (t_metrics[0], t_metrics[1]),
                "%d/%d" % (i_metrics[0], i_metrics[1]),
            )
            metrics_table.add_row(
                "Chain-end found",
                _chain_color(b_chain),
                _chain_color(t_chain),
                _chain_color(i_chain),
            )
            metrics_table.add_row(
                "Intrusion rate",
                "%d/%d" % (b_metrics[2], b_metrics[3]),
                "%d/%d" % (t_metrics[2], t_metrics[3]),
                "%d/%d" % (i_metrics[2], i_metrics[3]),
            )
            console.print(metrics_table)

            summaries.append(
                {
                    "query": truncate(query.text, 30),
                    "b_expected": "%d/%d" % (b_metrics[0], b_metrics[1]),
                    "t_expected": "%d/%d" % (t_metrics[0], t_metrics[1]),
                    "i_expected": "%d/%d" % (i_metrics[0], i_metrics[1]),
                    "b_chain": b_chain,
                    "t_chain": t_chain,
                    "i_chain": i_chain,
                    "rounds": rounds,
                }
            )

        # --- Summary table ---
        summary_table = Table(
            title="Summary",
            show_header=True,
            box=rich.box.DOUBLE_EDGE,
        )
        summary_table.add_column("Query", no_wrap=True)
        summary_table.add_column("Base", justify="center", no_wrap=True)
        summary_table.add_column("1-hop", justify="center", no_wrap=True)
        summary_table.add_column("Iter", justify="center", no_wrap=True)
        summary_table.add_column("B end", justify="center", no_wrap=True)
        summary_table.add_column("1h end", justify="center", no_wrap=True)
        summary_table.add_column("I end", justify="center", no_wrap=True)
        summary_table.add_column("Rnd", justify="center", no_wrap=True)

        for s in summaries:
            b_exp = str(s["b_expected"])
            t_exp = str(s["t_expected"])
            i_exp = str(s["i_expected"])

            # Color Iterative green when it beats One-hop
            i_exp_display = "[green]%s[/green]" % i_exp if i_exp > t_exp else i_exp

            summary_table.add_row(
                str(s["query"]),
                b_exp,
                t_exp,
                i_exp_display,
                _chain_color(str(s["b_chain"])),
                _chain_color(str(s["t_chain"])),
                _chain_color(str(s["i_chain"])),
                str(s["rounds"]),
            )

        console.print(summary_table)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        logger.exception("Benchmark failed")
        raise
    finally:
        await engine.teardown()


if __name__ == "__main__":
    asyncio.run(main())
