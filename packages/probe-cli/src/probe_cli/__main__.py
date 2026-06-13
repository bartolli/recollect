"""probe CLI entry."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from recollect.llm.pydantic_ai import PydanticAIProvider
from recollect.llm.types import CompletionParams

from probe_cli.arm import Arm, load_arm
from probe_cli.corpus import load_corpus, load_query_corpus
from probe_cli.metrics import (
    AggregateMetrics,
    RetrievalAggregate,
    RetrievalRunReport,
    RunReport,
    aggregate_retrieval_runs,
    aggregate_runs,
)
from probe_cli.report import (
    write_retrieval_run_report,
    write_retrieval_summary,
    write_run_report,
    write_situational_run_report,
    write_situational_summary,
    write_summary,
    write_surfacing_run_report,
    write_surfacing_summary,
)
from probe_cli.runner import ArmRunner, RetrievalArmRunner
from probe_cli.situational_metrics import (
    SituationalAggregate,
    aggregate_situational_runs,
)
from probe_cli.situational_runner import SituationalArmRunner, SituationalRunReport
from probe_cli.surfacing_metrics import SurfacingMetrics, compute_surfacing_metrics
from probe_cli.surfacing_runner import SurfacingArmRunner, SurfacingRunReport

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(prog="probe", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run a single arm and write reports.")
    run.add_argument("--arm", required=True, type=Path, help="Path to arm TOML.")
    run.add_argument(
        "--model",
        help=(
            "Override arm.extraction.pydantic_ai_model "
            "(e.g. 'anthropic:claude-haiku-4-5-20251001', 'ollama:ministral-3')."
        ),
    )
    run.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: ./.env). Pass empty string to skip.",
    )
    run.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging to stderr."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.env_file and str(args.env_file):
        _load_dotenv(args.env_file)

    if args.cmd == "run":
        return asyncio.run(_run_arm(args.arm, model_override=args.model))
    return 1


async def _run_arm(arm_path: Path, *, model_override: str | None) -> int:
    arm = load_arm(arm_path)
    if model_override:
        arm.extraction.pydantic_ai_model = model_override
    if not arm.extraction.pydantic_ai_model:
        logger.error(
            "model required: set arm.extraction.pydantic_ai_model in TOML or "
            "pass --model"
        )
        return 2

    provider = PydanticAIProvider(
        model=arm.extraction.pydantic_ai_model,
        defaults=CompletionParams(**arm.extraction.model_settings),
        max_retries=arm.extraction.max_retries,
    )
    out_dir = Path(arm.output.dir) / arm.name

    if arm.surfacing.enabled:
        return await _run_surfacing_arm(arm, provider, out_dir)
    if arm.situational.enabled:
        return await _run_situational_arm(arm, provider, out_dir)
    if arm.retrieval.enabled:
        return await _run_retrieval_arm(arm, provider, out_dir)
    return await _run_extraction_arm(arm, provider, out_dir)


async def _run_extraction_arm(
    arm: Arm, provider: PydanticAIProvider, out_dir: Path
) -> int:
    corpus = load_corpus(arm.corpus.path)
    runner = ArmRunner(arm, provider)
    reports = await runner.run_all(corpus)

    for report in reports:
        write_run_report(report, out_dir)

    summary = aggregate_runs(
        arm_name=arm.name,
        model=runner.model,
        prompt_version=runner.template_version,
        reports=reports,
    )
    write_summary(summary, out_dir)
    _print_summary(arm, summary, reports)
    return 0


async def _run_situational_arm(
    arm: Arm, provider: PydanticAIProvider, out_dir: Path
) -> int:
    runner = SituationalArmRunner.from_arm(arm, provider)
    reports = await runner.run_all()

    for report in reports:
        write_situational_run_report(report, out_dir)

    summary = aggregate_situational_runs(arm.name, runner.model, reports)
    write_situational_summary(summary, out_dir)
    _print_situational_summary(arm, summary, reports)
    return 0


async def _run_surfacing_arm(
    arm: Arm, provider: PydanticAIProvider, out_dir: Path
) -> int:
    runner = SurfacingArmRunner.from_arm(arm, provider)
    report = await runner.run()
    write_surfacing_run_report(report, out_dir)

    metrics = compute_surfacing_metrics(report)
    write_surfacing_summary(metrics, out_dir)
    _print_surfacing_summary(arm, report, metrics)
    return 0


async def _run_retrieval_arm(
    arm: Arm, provider: PydanticAIProvider, out_dir: Path
) -> int:
    traces = load_corpus(arm.retrieval.traces_corpus_path)
    queries = load_query_corpus(arm.retrieval.query_corpus_path)
    runner = RetrievalArmRunner(arm, provider)
    reports = await runner.run_all(traces, queries)

    for report in reports:
        write_retrieval_run_report(report, out_dir)

    summary = aggregate_retrieval_runs(
        arm_name=arm.name,
        model=runner.model,
        prompt_version=runner.template_version,
        top_k=arm.retrieval.top_k,
        reports=reports,
    )
    write_retrieval_summary(summary, out_dir)
    _print_retrieval_summary(arm, summary, reports)
    return 0


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        logger.debug(".env not found at %s; skipping", path)
        return
    loaded: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    if loaded:
        logger.info("loaded %d vars from %s: %s", len(loaded), path, sorted(loaded))


def _print_summary(
    arm: Arm,
    summary: AggregateMetrics,
    reports: list[RunReport],
) -> None:
    print(f"\nArm: {arm.name}", file=sys.stderr)
    print(f"  prompt={summary.prompt_version}  model={summary.model}", file=sys.stderr)
    print(
        f"  runs={summary.runs}  entries/run={summary.total_entries_per_run}",
        file=sys.stderr,
    )
    print(
        f"  validity      = {summary.validity_rate_mean:.3f}"
        f" ± {summary.validity_rate_std:.3f}",
        file=sys.stderr,
    )
    print(
        f"  predicate_div = {summary.predicate_diversity_mean:.3f}"
        f" ± {summary.predicate_diversity_std:.3f}",
        file=sys.stderr,
    )
    print(
        f"  escape_valve  = {summary.escape_valve_rate_mean:.3f}"
        f" ± {summary.escape_valve_rate_std:.3f}",
        file=sys.stderr,
    )
    print(
        f"  concepts/trace= {summary.concepts_per_trace_mean:.2f}"
        f" ± {summary.concepts_per_trace_std:.2f}",
        file=sys.stderr,
    )

    if summary.validity_rate_mean < 1.0:
        _print_failure_samples(reports)


def _print_retrieval_summary(
    arm: Arm,
    summary: RetrievalAggregate,
    reports: list[RetrievalRunReport],
) -> None:
    print(f"\nArm: {arm.name} (retrieval)", file=sys.stderr)
    print(
        f"  prompt={summary.prompt_version}  model={summary.model}  k={summary.top_k}",
        file=sys.stderr,
    )
    print(
        f"  runs={summary.runs}  queries/run={summary.queries_per_run}",
        file=sys.stderr,
    )
    print(
        f"  recall@k     = {summary.recall_at_k_mean:.3f}"
        f" ± {summary.recall_at_k_std:.3f}",
        file=sys.stderr,
    )
    print(
        f"  MRR          = {summary.mrr_mean:.3f} ± {summary.mrr_std:.3f}",
        file=sys.stderr,
    )
    print(
        f"  intrusion    = {summary.intrusion_rate_mean:.3f}"
        f" ± {summary.intrusion_rate_std:.3f}",
        file=sys.stderr,
    )
    print(
        f"  distractor   = {summary.distractor_intrusion_mean:.3f}"
        f" ± {summary.distractor_intrusion_std:.3f}",
        file=sys.stderr,
    )
    if summary.persona_facts_per_query_mean > 0:
        print(
            f"  persona/qry  = {summary.persona_facts_per_query_mean:.2f}"
            f" ± {summary.persona_facts_per_query_std:.2f}"
            "  (filtered from rank set)",
            file=sys.stderr,
        )
    if reports and reports[0].ingest_failures > 0:
        print(
            f"  WARN: ingest failures (run 0): {reports[0].ingest_failures}",
            file=sys.stderr,
        )
    if summary.per_category:
        print("  per category (recall@k / MRR):", file=sys.stderr)
        for cat, vals in sorted(summary.per_category.items()):
            print(
                f"    {cat:14s} r={vals['recall_at_k_mean']:.3f}"
                f"  mrr={vals['mrr_mean']:.3f}",
                file=sys.stderr,
            )


def _print_situational_summary(
    arm: Arm,
    summary: SituationalAggregate,
    reports: list[SituationalRunReport],
) -> None:
    print(f"\nArm: {arm.name} (situational)", file=sys.stderr)
    print(
        f"  model={summary.model}  runs={summary.runs}  evals/run={summary.eval_count}",
        file=sys.stderr,
    )
    print(
        f"  action_accuracy        = {summary.action_accuracy_mean:.3f}"
        f" ± {summary.action_accuracy_std:.3f}",
        file=sys.stderr,
    )
    print(
        f"  extension_target_acc   = {summary.extension_target_accuracy_mean:.3f}"
        f" ± {summary.extension_target_accuracy_std:.3f}",
        file=sys.stderr,
    )
    print(
        f"  none_rate (actual)     = {summary.none_rate_actual_mean:.3f}"
        f"  (expected={summary.none_rate_expected:.3f})",
        file=sys.stderr,
    )
    print(
        f"  expected distribution  = {summary.expected_distribution}",
        file=sys.stderr,
    )
    if reports and reports[0].seed_traces_ingested == 0:
        print("  WARN: no seed traces ingested in run 0", file=sys.stderr)


def _print_surfacing_summary(
    arm: Arm,
    report: SurfacingRunReport,
    metrics: SurfacingMetrics,
) -> None:
    print(f"\nArm: {arm.name} (surfacing)", file=sys.stderr)
    print(
        f"  model={report.model}  seeded={report.seeded_traces}"
        f"  promoted={report.promoted_facts}",
        file=sys.stderr,
    )
    print(
        f"  relevant   S: n={metrics.relevant.n}"
        f"  mean={metrics.relevant.mean:.3f}  median={metrics.relevant.median:.3f}",
        file=sys.stderr,
    )
    print(
        f"  distractor S: n={metrics.distractor.n}"
        f"  mean={metrics.distractor.mean:.3f}"
        f"  median={metrics.distractor.median:.3f}",
        file=sys.stderr,
    )
    print(
        f"  overlap: max(dis)={metrics.overlap_max_distractor:.3f}"
        f"  min(rel)={metrics.overlap_min_relevant:.3f}"
        f"  separable={metrics.separable}",
        file=sys.stderr,
    )
    print(
        f"  block_precision = {metrics.block_precision_mean:.3f}"
        f"  (n={metrics.block_precision_n} non-distractor queries)",
        file=sys.stderr,
    )
    print(
        f"  distractor queries: {metrics.distractor_query_count}"
        f"  ({metrics.distractor_queries_with_surfaced} surfaced >=1 fact)",
        file=sys.stderr,
    )
    print("  threshold sweep (T: precision / recall):", file=sys.stderr)
    for p in metrics.threshold_sweep:
        print(
            f"    {p.threshold:.2f}  prec={p.precision:.3f}  recall={p.recall:.3f}"
            f"  (rel={p.kept_relevant} dis={p.kept_distractor})",
            file=sys.stderr,
        )


def _print_failure_samples(reports: list[RunReport], limit: int = 3) -> None:
    if not reports:
        return
    seen: set[str] = set()
    print("\n  Failures (first run, deduped):", file=sys.stderr)
    for entry in reports[0].entries:
        if entry.success or not entry.error:
            continue
        key = entry.error.split("\n", 1)[0][:200]
        if key in seen:
            continue
        seen.add(key)
        print(f"    [{entry.entry_id}] {key}", file=sys.stderr)
        if len(seen) >= limit:
            break


if __name__ == "__main__":
    sys.exit(main())
