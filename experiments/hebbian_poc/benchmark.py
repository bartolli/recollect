"""Hebbian Recall Tokens POC benchmark runner.

Two-Sarahs entity disambiguation: tests whether recall tokens
create relational groups that disambiguate people with the same
name in different social contexts.

Usage:
    uv run python experiments/hebbian_poc/benchmark.py
    uv run python experiments/hebbian_poc/benchmark.py --llm ollama:ministral-3
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

from recollect.log_setup import configure_logging

from experiments.hebbian_poc.config import PocConfig
from experiments.hebbian_poc.engine import PocEngine
from experiments.hebbian_poc.retrieval import (
    RetrievalResult,
    baseline_recall,
    token_recall,
)
from experiments.hebbian_poc.scenarios import SCENARIO, Query

logger = logging.getLogger(__name__)

SHOW_WRITE_TIME_GROUPS = {"anchor", "mother", "kid"}


def parse_args() -> PocConfig:
    parser = argparse.ArgumentParser(
        description="Hebbian Recall Tokens POC: Two-Sarahs Entity Disambiguation"
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL connection URL",
    )
    parser.add_argument("--llm-base-url", default=None, help="Ollama API base URL")
    parser.add_argument(
        "--llm",
        default=None,
        help="pydantic-ai model string (e.g. ollama:ministral-3)",
    )
    parser.add_argument("--embedding-model", default=None, help="Embedding model name")
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help="Max tokens for LLM",
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
        help="Path to prompt markdown file (default: built-in relational prompt)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
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

    configure_logging(
        log_file="logs/poc-hebbian.jsonl",
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
    return f"{num * 100 // den}%"


def compute_metrics(
    results: list[RetrievalResult],
    query: Query,
    trace_id_to_index: dict[str, int],
) -> tuple[int, int, int, int, int, int]:
    """Return (expected, unexpected, relevant, total) counts."""
    retrieved_indices = {
        trace_id_to_index[r.trace_id]
        for r in results
        if r.trace_id in trace_id_to_index
    }

    expected_found = len(set(query.expected_indices) & retrieved_indices)
    expected_total = len(query.expected_indices)

    unexpected_found = len(set(query.unexpected_indices) & retrieved_indices)
    unexpected_total = len(query.unexpected_indices)

    # Precision: how many results are in the expected set
    relevant = 0
    for r in results:
        if r.trace_id in trace_id_to_index:
            idx = trace_id_to_index[r.trace_id]
            if idx in query.expected_indices:
                relevant += 1

    return (
        expected_found,
        expected_total,
        unexpected_found,
        unexpected_total,
        relevant,
        len(results),
    )


async def main() -> None:  # noqa: C901
    config = parse_args()
    engine = PocEngine(config)
    scenario = SCENARIO
    index_to_group = {i: m.group for i, m in enumerate(scenario.memories)}

    print("=== Hebbian Recall Tokens POC: Two-Sarahs ===")
    prompt_label = Path(config.prompt_path).stem if config.prompt_path else "built-in"
    print(
        "Config: %s, %s, prompt=%s"
        % (
            config.llm,
            config.embedding_model.split("/")[-1],
            prompt_label,
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
        print("\n--- Storing %d memories ---" % len(scenario.memories))
        trace_ids: list[str] = []

        for i, memory in enumerate(scenario.memories):
            result = await engine.experience(memory.content)
            trace_ids.append(result.trace_id)

            label = "[%-6s]" % memory.group
            content_short = truncate(memory.content)
            print('  %s  T%-2d  "%s"' % (label, i, content_short))

            # Show write-time recall detail for non-hay memories
            if memory.group in SHOW_WRITE_TIME_GROUPS:
                wtr = result.write_time_recall
                if wtr.related_found:
                    related_strs = []
                    for c in wtr.related_found:
                        related_strs.append('"%s"' % truncate(c, 40))
                    related_display = ", ".join(related_strs)
                    token_str = (
                        'TOKEN "%s"' % wtr.token_label
                        if wtr.token_created
                        else "no token"
                    )
                    print(
                        "            write-time: found [%s] -> %s"
                        % (related_display, token_str)
                    )
                else:
                    print("            write-time: found [] -> no token")

        # Build trace_id -> index mapping
        trace_id_to_index = {tid: i for i, tid in enumerate(trace_ids)}

        # --- Run queries ---
        recall_top_k = config.recall_top_k
        summaries: list[dict[str, str]] = []

        for qi, query in enumerate(scenario.queries):
            print('\n--- Query %d: "%s" ---' % (qi + 1, query.text))
            print("  %s" % query.description)
            print(
                "  Expected: %s"
                % ", ".join(
                    "T%d (%s)" % (idx, truncate(scenario.memories[idx].content, 30))
                    for idx in query.expected_indices
                )
            )
            print(
                "  Unexpected: %s"
                % ", ".join("T%d" % idx for idx in query.unexpected_indices)
            )

            # Generate query embedding
            query_embedding = await engine.embed(query.text)

            # Baseline recall (vector only)
            baseline_results = await baseline_recall(
                engine.pool, query_embedding, top_k=recall_top_k
            )
            print("\n  Baseline (vector only):")
            for r in baseline_results:
                idx = trace_id_to_index.get(r.trace_id)
                group = index_to_group.get(idx, "?") if idx is not None else "?"
                print("    [%.2f] [%-6s] %s" % (r.score, group, truncate(r.content)))

            # Token-augmented recall
            token_results = await token_recall(
                engine.pool,
                query_embedding,
                top_k=recall_top_k,
                strength_threshold=engine._config.token_strength_threshold,
                reinforce_boost=engine._config.token_reinforce_boost,
                token_bonus=engine._config.token_score_bonus,
            )
            print("\n  Token-augmented:")
            for r in token_results:
                idx = trace_id_to_index.get(r.trace_id)
                group = index_to_group.get(idx, "?") if idx is not None else "?"
                source_str = (
                    "(token: %s)" % r.token_label if r.source == "token" else "(vector)"
                )
                print(
                    "    [%.2f] %-30s [%-6s] %s"
                    % (r.score, source_str, group, truncate(r.content))
                )

            # Compute metrics
            b_ef, b_et, b_uf, b_ut, b_rel, b_tot = compute_metrics(
                baseline_results, query, trace_id_to_index
            )
            t_ef, t_et, t_uf, t_ut, t_rel, t_tot = compute_metrics(
                token_results, query, trace_id_to_index
            )

            print("\n  Metrics:")
            print(
                "    Expected recall  -- baseline: %d/%d  tokens: %d/%d"
                % (b_ef, b_et, t_ef, t_et)
            )
            print(
                "    Intrusion rate   -- baseline: %d/%d  tokens: %d/%d"
                % (b_uf, b_ut, t_uf, t_ut)
            )
            print(
                "    Precision        -- baseline: %s  tokens: %s"
                % (pct(b_rel, b_tot), pct(t_rel, t_tot))
            )

            summaries.append(
                {
                    "query": truncate(query.text, 25),
                    "base_expected": "%d/%d" % (b_ef, b_et),
                    "token_expected": "%d/%d" % (t_ef, t_et),
                    "base_intrusion": "%d/%d" % (b_uf, b_ut),
                    "token_intrusion": "%d/%d" % (t_uf, t_ut),
                    "base_precision": pct(b_rel, b_tot),
                    "token_precision": pct(t_rel, t_tot),
                }
            )

        # --- Summary table ---
        print("\n=== Summary ===")
        sep = "|-%-27s-|-%-16s-|-%-16s-|-%-16s-|-%-17s-|-%-16s-|-%-16s-|" % (
            "-" * 27,
            "-" * 16,
            "-" * 16,
            "-" * 16,
            "-" * 17,
            "-" * 16,
            "-" * 16,
        )
        print(sep)
        print(
            "| %-27s | %-16s | %-16s | %-16s | %-17s | %-16s | %-16s |"
            % (
                "Query",
                "Base Expected",
                "Token Expected",
                "Base Intrusion",
                "Token Intrusion",
                "Base Precision",
                "Token Precision",
            )
        )
        print(sep)
        for s in summaries:
            print(
                "| %-27s | %-16s | %-16s | %-16s | %-17s | %-16s | %-16s |"
                % (
                    s["query"],
                    s["base_expected"],
                    s["token_expected"],
                    s["base_intrusion"],
                    s["token_intrusion"],
                    s["base_precision"],
                    s["token_precision"],
                )
            )
        print(sep)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        logger.exception("Benchmark failed")
        raise
    finally:
        await engine.teardown()


if __name__ == "__main__":
    asyncio.run(main())
