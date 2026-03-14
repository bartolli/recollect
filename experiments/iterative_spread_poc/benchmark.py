"""Iterative Spreading Activation POC benchmark runner.

Chain discovery: tests whether iterative re-seeding discovers traces
at the end of long association chains that fixed 2-hop spreading
activation misses.

Usage:
    uv run python experiments/iterative_spread_poc/benchmark.py
    uv run python experiments/iterative_spread_poc/benchmark.py \\
        --spread-decay 0.8 --iter-max-rounds 5
    uv run python experiments/iterative_spread_poc/benchmark.py -v
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

from experiments.iterative_spread_poc.config import PocConfig
from experiments.iterative_spread_poc.engine import PocEngine
from experiments.iterative_spread_poc.retrieval import (
    RetrievalResult,
    baseline_recall,
    fixed_spread_recall,
    iterative_recall,
)
from experiments.iterative_spread_poc.scenarios import SCENARIO, Query

logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> PocConfig:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Iterative Spreading Activation POC: Chain Discovery"
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name (default: nomic-ai/nomic-embed-text-v1.5-Q)",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=None,
        help="Embedding dimensions (default: 768)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results for recall queries (default: 10)",
    )
    parser.add_argument(
        "--spread-decay",
        type=float,
        default=None,
        help="Activation decay per hop (default: 0.7)",
    )
    parser.add_argument(
        "--spread-threshold",
        type=float,
        default=None,
        help="Stop spreading below this level (default: 0.1)",
    )
    parser.add_argument(
        "--spread-max-depth",
        type=int,
        default=None,
        help="Max hops for fixed CTE (default: 2)",
    )
    parser.add_argument(
        "--spread-weight",
        type=float,
        default=None,
        help="Score bonus multiplier: activation * weight (default: 0.1)",
    )
    parser.add_argument(
        "--iter-max-rounds",
        type=int,
        default=None,
        help="Max re-seeding rounds (default: 3)",
    )
    parser.add_argument(
        "--iter-top-seeds",
        type=int,
        default=None,
        help="How many top results become seeds each round (default: 3)",
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
    if args.embedding_model is not None:
        config.embedding_model = args.embedding_model
    if args.embedding_dimensions is not None:
        config.embedding_dimensions = args.embedding_dimensions
    if args.top_k is not None:
        config.recall_top_k = args.top_k
    if args.spread_decay is not None:
        config.spread_decay = args.spread_decay
    if args.spread_threshold is not None:
        config.spread_threshold = args.spread_threshold
    if args.spread_max_depth is not None:
        config.spread_max_depth = args.spread_max_depth
    if args.spread_weight is not None:
        config.spread_weight = args.spread_weight
    if args.iter_max_rounds is not None:
        config.iter_max_rounds = args.iter_max_rounds
    if args.iter_top_seeds is not None:
        config.iter_top_seeds = args.iter_top_seeds

    configure_logging(
        log_file="logs/poc-iterative-spread.jsonl",
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
    table.add_column("Source", width=18, no_wrap=True)
    table.add_column("Group", width=8, no_wrap=True)
    table.add_column("T#", justify="right", width=3)
    table.add_column("Content", no_wrap=True)

    for r in results[:10]:
        idx = trace_id_to_index.get(r.trace_id)
        grp = index_to_group.get(idx, "?") if idx is not None else "?"
        idx_str = str(idx) if idx is not None else "?"

        src = (
            "(%s, d=%d)" % (r.source, r.hop_depth)
            if r.source != "vector"
            else "(vector)"
        )

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


async def main() -> None:
    config = parse_args()
    engine = PocEngine(config)
    scenario = SCENARIO
    index_to_group = {i: m.group for i, m in enumerate(scenario.memories)}

    console.rule("[bold]Iterative Spreading Activation POC: Chain Discovery")
    console.print(
        "decay=[cyan]%.2f[/]  threshold=[cyan]%.2f[/]  depth=[cyan]%d[/]"
        "  weight=[cyan]%.2f[/]  rounds=[cyan]%d[/]  seeds=[cyan]%d[/]"
        % (
            config.spread_decay,
            config.spread_threshold,
            config.spread_max_depth,
            config.spread_weight,
            config.iter_max_rounds,
            config.iter_top_seeds,
        )
    )

    try:
        await engine.setup()

        # Clean tables
        async with engine.pool.acquire() as conn:
            await conn.execute(
                "TRUNCATE poc_iter_associations, poc_iter_traces CASCADE"
            )

        # --- Store all memories ---
        print("\n--- Storing %d memories ---" % len(scenario.memories))
        trace_ids: list[str] = []
        for i, memory in enumerate(scenario.memories):
            result = await engine.store(memory.content)
            trace_ids.append(result.id)
            # Only print chain memories, not hay
            if memory.group != "hay":
                print(
                    '  [%-8s]  T%-2d  "%s"'
                    % (memory.group, i, truncate(memory.content))
                )

        console.print(
            "  [dim]... and %d haystack memories[/dim]"
            % sum(1 for m in scenario.memories if m.group == "hay")
        )

        # --- Create associations ---
        print("\n--- Creating %d associations ---" % len(scenario.associations))
        assoc_table = Table(show_header=True, box=rich.box.SIMPLE)
        assoc_table.add_column("Source", style="cyan")
        assoc_table.add_column("Target", style="cyan")
        assoc_table.add_column("Fwd", justify="right")
        assoc_table.add_column("Bwd", justify="right")

        for assoc in scenario.associations:
            source_id = trace_ids[assoc.source_index]
            target_id = trace_ids[assoc.target_index]
            await engine.create_association(
                source_id,
                target_id,
                association_type=assoc.association_type,
                forward_strength=assoc.forward_strength,
                backward_strength=assoc.backward_strength,
            )
            src_group = scenario.memories[assoc.source_index].group
            tgt_group = scenario.memories[assoc.target_index].group
            assoc_table.add_row(
                "T%d (%s)" % (assoc.source_index, src_group),
                "T%d (%s)" % (assoc.target_index, tgt_group),
                "%.2f" % assoc.forward_strength,
                "%.2f" % assoc.backward_strength,
            )

        console.print(assoc_table)

        # Build trace_id -> index mapping
        trace_id_to_index = {tid: i for i, tid in enumerate(trace_ids)}

        # --- Run queries ---
        summaries: list[dict[str, object]] = []

        for qi, query in enumerate(scenario.queries):
            console.rule("[bold]Query %d: %s" % (qi + 1, query.text))
            console.print("[dim]%s[/dim]" % query.description)
            console.print(
                "[dim]Expected (all): %s[/dim]"
                % ", ".join("T%d" % idx for idx in query.expected_indices)
            )
            console.print(
                "[dim]Expected (2-hop): %s[/dim]"
                % ", ".join("T%d" % idx for idx in query.expected_2hop)
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

            # --- Fixed 2-hop ---
            fixed_results = await fixed_spread_recall(
                engine.pool,
                query_embedding,
                top_k=config.recall_top_k,
                spread_decay=config.spread_decay,
                spread_threshold=config.spread_threshold,
                spread_max_depth=config.spread_max_depth,
                num_seeds=config.iter_top_seeds,
                spread_weight=config.spread_weight,
            )
            console.print(
                _make_result_table(
                    "Fixed 2-hop spread",
                    fixed_results,
                    query,
                    trace_id_to_index,
                    index_to_group,
                )
            )

            # --- Iterative ---
            iter_results, rounds = await iterative_recall(
                engine.pool,
                query_embedding,
                top_k=config.recall_top_k,
                spread_decay=config.spread_decay,
                spread_threshold=config.spread_threshold,
                spread_max_depth=config.spread_max_depth,
                num_seeds=config.iter_top_seeds,
                max_rounds=config.iter_max_rounds,
                stability_threshold=config.iter_stability_threshold,
                spread_weight=config.spread_weight,
            )
            console.print(
                _make_result_table(
                    "Iterative spread (%d rounds)" % rounds,
                    iter_results,
                    query,
                    trace_id_to_index,
                    index_to_group,
                )
            )

            # --- Metrics ---
            b_metrics = compute_metrics(baseline_results, query, trace_id_to_index)
            f_metrics = compute_metrics(fixed_results, query, trace_id_to_index)
            i_metrics = compute_metrics(iter_results, query, trace_id_to_index)

            b_chain = chain_end_found(baseline_results, query, trace_id_to_index)
            f_chain = chain_end_found(fixed_results, query, trace_id_to_index)
            i_chain = chain_end_found(iter_results, query, trace_id_to_index)

            metrics_table = Table(
                title="Metrics",
                show_header=True,
                box=rich.box.SIMPLE_HEAVY,
            )
            metrics_table.add_column("Metric")
            metrics_table.add_column("Baseline", justify="center")
            metrics_table.add_column("Fixed 2-hop", justify="center")
            metrics_table.add_column("Iterative", justify="center")

            metrics_table.add_row(
                "Expected recall",
                "%d/%d" % (b_metrics[0], b_metrics[1]),
                "%d/%d" % (f_metrics[0], f_metrics[1]),
                "%d/%d" % (i_metrics[0], i_metrics[1]),
            )
            metrics_table.add_row(
                "Chain-end found",
                _chain_color(b_chain),
                _chain_color(f_chain),
                _chain_color(i_chain),
            )
            metrics_table.add_row(
                "Intrusion rate",
                "%d/%d" % (b_metrics[2], b_metrics[3]),
                "%d/%d" % (f_metrics[2], f_metrics[3]),
                "%d/%d" % (i_metrics[2], i_metrics[3]),
            )
            console.print(metrics_table)

            summaries.append(
                {
                    "query": truncate(query.text, 30),
                    "b_expected": "%d/%d" % (b_metrics[0], b_metrics[1]),
                    "f_expected": "%d/%d" % (f_metrics[0], f_metrics[1]),
                    "i_expected": "%d/%d" % (i_metrics[0], i_metrics[1]),
                    "b_chain": b_chain,
                    "f_chain": f_chain,
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
        summary_table.add_column("Fixed", justify="center", no_wrap=True)
        summary_table.add_column("Iter", justify="center", no_wrap=True)
        summary_table.add_column("B end", justify="center", no_wrap=True)
        summary_table.add_column("F end", justify="center", no_wrap=True)
        summary_table.add_column("I end", justify="center", no_wrap=True)
        summary_table.add_column("Rnd", justify="center", no_wrap=True)

        for s in summaries:
            b_exp = str(s["b_expected"])
            f_exp = str(s["f_expected"])
            i_exp = str(s["i_expected"])

            # Color Iterative green when it beats Fixed
            i_exp_display = "[green]%s[/green]" % i_exp if i_exp > f_exp else i_exp

            summary_table.add_row(
                str(s["query"]),
                b_exp,
                f_exp,
                i_exp_display,
                _chain_color(str(s["b_chain"])),
                _chain_color(str(s["f_chain"])),
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
