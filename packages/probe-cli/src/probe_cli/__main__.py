"""probe CLI entry."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import Counter
from contextlib import aclosing
from pathlib import Path

from recollect.exceptions import MemorySDKError
from recollect.llm.pydantic_ai import PydanticAIProvider
from recollect.llm.types import CompletionParams

from probe_cli.aging import Aging, AgingReport, SweepPoint
from probe_cli.arm import Arm, load_arm
from probe_cli.corpus import (
    load_corpus,
    load_ground_truth,
    load_query_corpus,
    load_seed_groups,
)
from probe_cli.entity_reach import EntityReachReport, StorePreconditionError
from probe_cli.entity_reach import measure as measure_entity_reach
from probe_cli.fact_audit import FactAuditReport
from probe_cli.fusion_screen import FusionScreenReport, ScreenPreconditionError
from probe_cli.fusion_screen import screen as screen_fusion
from probe_cli.metrics import (
    AggregateMetrics,
    RetrievalAggregate,
    RetrievalRunReport,
    RunReport,
    aggregate_retrieval_runs,
    aggregate_runs,
)
from probe_cli.recall_answers import RecallAnswersReport
from probe_cli.recall_answers import read as read_recall_answers
from probe_cli.report import (
    write_case_breakdown,
    write_entity_reach_report,
    write_fusion_screen_report,
    write_recall_answers_report,
    write_retrieval_run_report,
    write_retrieval_summary,
    write_run_report,
    write_situational_lift,
    write_situational_run_report,
    write_situational_summary,
    write_situational_surfacing_report,
    write_summary,
    write_surfacing_run_report,
    write_surfacing_summary,
    write_weak_label_answers_report,
    write_weak_match_screen_report,
)
from probe_cli.runner import ArmRunner, RetrievalArmRunner
from probe_cli.situational_metrics import (
    SituationalAggregate,
    aggregate_situational_runs,
)
from probe_cli.situational_runner import SituationalArmRunner, SituationalRunReport
from probe_cli.surfacing_cases import CaseBreakdown, classify_cases
from probe_cli.surfacing_metrics import SurfacingMetrics, compute_surfacing_metrics
from probe_cli.surfacing_runner import SurfacingArmRunner, SurfacingRunReport
from probe_cli.surfacing_situational import SituationalLift, compute_situational_lift
from probe_cli.surfacing_situational_runner import (
    SituationalSurfacingArmRunner,
    SituationalSurfacingReport,
)
from probe_cli.task_metrics import TaskAggregate, aggregate_task_runs
from probe_cli.task_runner import (
    TaskArmRunner,
    TaskRunReport,
    TaskVerifyReport,
    VerdictOracleReport,
)
from probe_cli.weak_label_answers import QuestionAnswers, WeakLabelAnswersReport
from probe_cli.weak_label_answers import compare as compare_weak_labels
from probe_cli.weak_match_screen import WeakMatchScreenReport
from probe_cli.weak_match_screen import screen as screen_weak_matches

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
    run.add_argument(
        "--verify-reachability",
        action="store_true",
        help=(
            "Task arms only: verify each question's intended tier_label "
            "empirically and report drift; no answering runs."
        ),
    )
    run.add_argument(
        "--audit-fact-channel",
        action="store_true",
        help=(
            "Task arms only: seed each t3 tail text per round against the "
            "scratch audit DB and classify extraction-to-promotion outcomes "
            "(story-9 slice 1); no answering runs."
        ),
    )
    run.add_argument(
        "--age-days",
        type=_non_negative,
        nargs="+",
        help=(
            "Task-arm answering runs only: age the seeded store by this many "
            "emulated days before evaluating (0 = control, same path). "
            "Several ages sweep one seeding, restored before each point."
        ),
    )
    run.add_argument(
        "--passes-per-day",
        type=_non_negative,
        help=(
            "Emulated consolidation passes per aged day (default 48: one "
            "worker at the packaged interval; 0 = one closing pass)."
        ),
    )
    run.add_argument(
        "--tokens-off-control",
        action="store_true",
        help=(
            "With --age-days: also evaluate the same seeding at age 0 with "
            "the recall token hop disabled at read time."
        ),
    )
    reach = sub.add_parser(
        "entity-reach",
        help="Measure entity-graph reach on a seeded task-arm store (read-only).",
    )
    reach.add_argument("--arm", required=True, type=Path, help="Path to arm TOML.")
    reach.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: ./.env). Pass empty string to skip.",
    )
    reach.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.90, 0.85, 0.80],
        help="Embedding cosine thresholds, one rule each (default: 0.90 0.85 0.80).",
    )
    reach.set_defaults(verbose=False)
    fusion = sub.add_parser(
        "fusion-screen",
        help=(
            "Screen concept fusion forms on a copy of a seeded task-arm store "
            "(writes the copy only)."
        ),
    )
    fusion.add_argument(
        "--arm", required=True, type=Path, help="Task arm TOML that seeded the store."
    )
    fusion.add_argument(
        "--template-db",
        required=True,
        help="Seeded store to copy from; never connected to.",
    )
    fusion.add_argument(
        "--name", required=True, help="Evidence dir under the arm's output dir."
    )
    fusion.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: ./.env). Pass empty string to skip.",
    )
    fusion.set_defaults(verbose=False)
    weak = sub.add_parser(
        "weak-match-screen",
        help=(
            "Score fixed weak-match rules on captured think_about windows from a "
            "copy of a store (writes the copy only)."
        ),
    )
    weak.add_argument(
        "--arm", required=True, type=Path, help="Task arm TOML: config and DB URL."
    )
    weak.add_argument(
        "--template-db",
        required=True,
        help="Store to copy from; never connected to.",
    )
    weak.add_argument(
        "--out-dir", required=True, type=Path, help="Directory for the artifact."
    )
    weak.add_argument("--user-id", help="Store user (default: probe-<arm name>).")
    weak.add_argument(
        "--queries",
        type=Path,
        help=(
            "JSONL of {id, query, token_budget?, expected_strong?, expected_weak?} "
            "in place of the arm's task questions."
        ),
    )
    weak.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: ./.env). Pass empty string to skip.",
    )
    weak.set_defaults(verbose=False)
    labels = sub.add_parser(
        "weak-label-answers",
        help=(
            "Answer every task question from the current and the weak-labelled "
            "recall text of one captured window on a store copy (writes the copy only)."
        ),
    )
    labels.add_argument(
        "--arm", required=True, type=Path, help="Task arm TOML: config, DB URL, model."
    )
    labels.add_argument(
        "--template-db",
        required=True,
        help="Store to copy from; never connected to.",
    )
    labels.add_argument(
        "--out-dir", required=True, type=Path, help="Directory for the artifact."
    )
    labels.add_argument("--user-id", help="Store user (default: probe-<arm name>).")
    labels.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: ./.env). Pass empty string to skip.",
    )
    labels.set_defaults(verbose=False)
    recall = sub.add_parser(
        "recall-answers",
        help=(
            "Answer free-form queries from the MCP recall text of one captured "
            "window per query on a store copy (writes the copy only)."
        ),
    )
    recall.add_argument(
        "--arm", required=True, type=Path, help="Task arm TOML: config, DB URL, model."
    )
    recall.add_argument(
        "--template-db",
        required=True,
        help="Store to copy from; never connected to.",
    )
    recall.add_argument(
        "--queries",
        required=True,
        type=Path,
        help="JSONL of {id, text, relevant_trace_ids} (retrieval query corpus).",
    )
    recall.add_argument(
        "--out-dir", required=True, type=Path, help="Directory for the artifact."
    )
    recall.add_argument("--user-id", help="Store user (default: probe-<arm name>).")
    recall.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: ./.env). Pass empty string to skip.",
    )
    recall.set_defaults(verbose=False)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.env_file and str(args.env_file):
        _load_dotenv(args.env_file)

    if args.cmd == "run":
        if args.verify_reachability and args.audit_fact_channel:
            logger.error("--verify-reachability and --audit-fact-channel are exclusive")
            return 2
        if args.age_days is None and (
            args.passes_per_day is not None or args.tokens_off_control
        ):
            logger.error("--passes-per-day and --tokens-off-control require --age-days")
            return 2
        ages: list[int] = args.age_days or []
        if len(set(ages)) != len(ages):
            # Two points at one age would share an evidence dir.
            logger.error("--age-days lists an age twice")
            return 2
        return asyncio.run(
            _run_arm(
                args.arm,
                model_override=args.model,
                verify_reachability=args.verify_reachability,
                audit_fact_channel=args.audit_fact_channel,
                points=_points(
                    ages, args.passes_per_day, control=args.tokens_off_control
                ),
            )
        )
    if args.cmd == "entity-reach":
        return asyncio.run(_run_entity_reach(args.arm, args.thresholds))
    if args.cmd == "fusion-screen":
        return asyncio.run(
            _run_fusion_screen(args.arm, template_db=args.template_db, name=args.name)
        )
    if args.cmd == "weak-match-screen":
        return asyncio.run(
            _run_weak_match_screen(
                args.arm,
                template_db=args.template_db,
                out_dir=args.out_dir,
                user_id=args.user_id,
                queries_path=args.queries,
            )
        )
    if args.cmd in ("weak-label-answers", "recall-answers"):
        return asyncio.run(_run_answers(args))
    return 1


def _non_negative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"{value} is negative")
    return number


def _aging(age_days: int, passes_per_day: int | None) -> Aging:
    if passes_per_day is None:
        return Aging(age_days=age_days)
    return Aging(age_days=age_days, passes_per_day=passes_per_day)


def _points(
    ages: list[int], passes_per_day: int | None, *, control: bool
) -> list[SweepPoint]:
    points = [SweepPoint(aging=_aging(age, passes_per_day)) for age in ages]
    if control:
        points.append(SweepPoint(aging=_aging(0, passes_per_day), tokens_off=True))
    return points


def _single_age(points: list[SweepPoint]) -> Aging | None:
    """The one aged run that keeps a seeding to itself; None for a sweep."""
    if len(points) == 1 and not points[0].tokens_off:
        return points[0].aging
    return None


def _existing_evidence(arm: Arm, points: list[SweepPoint]) -> list[str]:
    """Evidence dirs an aged run would write that already hold evidence."""
    if not points:
        return []
    # A single-age arm already carries its age suffix.
    single = _single_age(points) is not None
    names = [arm.name] if single else [p.arm_name(arm.name) for p in points]
    return [name for name in names if (Path(arm.output.dir) / name).exists()]


def _prepared_arm(
    arm_path: Path,
    *,
    model_override: str | None,
    points: list[SweepPoint],
    answering: bool,
) -> Arm | None:
    """The arm ready to run; None (after logging) on a usage error."""
    arm = load_arm(arm_path)
    if points and not (answering and arm.task.enabled):
        logger.error("--age-days applies to task-arm answering runs only")
        return None
    aging = _single_age(points)
    if aging is not None:
        # Arm name keys the evidence dir and the user_id: an aged run is
        # its own arm and never overwrites the unaged one. A sweep keeps
        # the arm's name for its one seeding and suffixes per point.
        arm = arm.model_copy(update={"name": aging.arm_name(arm.name)})
    if model_override:
        arm.extraction.pydantic_ai_model = model_override
    if not arm.extraction.pydantic_ai_model:
        logger.error(
            "model required: set arm.extraction.pydantic_ai_model in TOML or "
            "pass --model"
        )
        return None
    return arm


async def _run_arm(
    arm_path: Path,
    *,
    model_override: str | None,
    verify_reachability: bool = False,
    audit_fact_channel: bool = False,
    points: list[SweepPoint] | None = None,
) -> int:
    points = points or []
    arm = _prepared_arm(
        arm_path,
        model_override=model_override,
        points=points,
        answering=not (verify_reachability or audit_fact_channel),
    )
    if arm is None:
        return 2
    arm.apply_process_overrides()
    taken = _existing_evidence(arm, points)
    if taken:
        # A sweep point and a single-age run share dir names, and every
        # sweep of an arm shares one seeding identity: evidence written
        # over or beside an earlier run's reads as one seeding.
        logger.error(
            "evidence exists under %s for %s; move it or run a new arm name",
            arm.output.dir,
            ", ".join(taken),
        )
        return 2

    provider = PydanticAIProvider(
        model=arm.extraction.pydantic_ai_model,
        defaults=CompletionParams(**arm.extraction.model_settings),
        max_retries=arm.extraction.max_retries,
    )
    out_dir = Path(arm.output.dir) / arm.name

    if arm.task.enabled:
        if audit_fact_channel:
            return await _run_fact_audit(arm, provider, out_dir)
        return await _run_task_arm(
            arm,
            provider,
            out_dir,
            verify_reachability=verify_reachability,
            points=points,
        )
    if verify_reachability or audit_fact_channel:
        logger.error(
            "--verify-reachability and --audit-fact-channel require a task arm"
        )
        return 2
    if arm.surfacing.enabled:
        if arm.surfacing.seed_groups_path:
            return await _run_situational_surfacing_arm(arm, provider, out_dir)
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


async def _run_task_arm(
    arm: Arm,
    provider: PydanticAIProvider,
    out_dir: Path,
    *,
    verify_reachability: bool,
    points: list[SweepPoint],
) -> int:
    from probe_cli.report import write_task_verify_report

    aging = _single_age(points)
    runner = TaskArmRunner.from_arm(arm, provider, aging=aging)
    if verify_reachability:
        verify = await runner.verify_reachability()
        write_task_verify_report(verify, out_dir)
        _print_task_verify(arm, verify)
        return 0 if not verify.drifted else 1

    if points and aging is None:
        # Evidence lands as each point completes: a later point that fails
        # does not take the finished ones with it.
        async with aclosing(runner.run_sweep(points)) as results:
            async for result in results:
                point_arm = arm.model_copy(update={"name": result.arm_name})
                summary = _write_task_evidence(
                    point_arm, result.reports, result.oracle, result.aging
                )
                _print_task_summary(point_arm, runner.answer_model, summary)
                _print_verdict_oracle(result.oracle)
        return 0

    reports, oracle = await runner.run_all()
    summary = _write_task_evidence(arm, reports, oracle, runner.aging_report)
    _print_task_summary(arm, runner.answer_model, summary)
    _print_verdict_oracle(oracle)
    return 0


def _write_task_evidence(
    arm: Arm,
    reports: list[TaskRunReport],
    oracle: VerdictOracleReport,
    aging_report: AgingReport | None,
) -> TaskAggregate:
    from probe_cli.report import (
        write_aging_report,
        write_task_run_report,
        write_task_summary,
        write_verdict_oracle,
    )

    out_dir = Path(arm.output.dir) / arm.name
    for report in reports:
        write_task_run_report(report, out_dir)
    summary = aggregate_task_runs(arm.name, reports)
    write_task_summary(summary, out_dir)
    write_verdict_oracle(oracle, out_dir)
    if aging_report is not None:
        write_aging_report(aging_report, out_dir)
    return summary


async def _run_fact_audit(arm: Arm, provider: PydanticAIProvider, out_dir: Path) -> int:
    from probe_cli.fact_audit import FactAuditRunner
    from probe_cli.report import write_fact_audit_report

    runner = FactAuditRunner.from_arm(arm, provider)
    report = await runner.run()
    write_fact_audit_report(report, out_dir)
    _print_fact_audit(arm, report)
    return 0


async def _run_entity_reach(arm_path: Path, thresholds: list[float]) -> int:
    arm = load_arm(arm_path)
    if not arm.task.enabled:
        logger.error("entity-reach requires a task arm")
        return 2
    try:
        report = await measure_entity_reach(arm, thresholds)
    except StorePreconditionError as exc:
        logger.error("%s", exc)
        return 2
    write_entity_reach_report(report, Path(arm.output.dir) / arm.name)
    _print_entity_reach(report)
    return 0


async def _run_fusion_screen(arm_path: Path, *, template_db: str, name: str) -> int:
    arm = load_arm(arm_path)
    if not arm.task.enabled:
        logger.error("fusion-screen requires a task arm")
        return 2
    try:
        report = await screen_fusion(arm, template=template_db)
    except ScreenPreconditionError as exc:
        logger.error("%s", exc)
        return 2
    write_fusion_screen_report(report, Path(arm.output.dir) / name)
    _print_fusion_screen(report)
    # Exit 1 = a pool whose offline prod replay missed think_about's window.
    return 0 if all(v.matched == v.checks for v in report.validity) else 1


async def _run_weak_match_screen(
    arm_path: Path,
    *,
    template_db: str,
    out_dir: Path,
    user_id: str | None,
    queries_path: Path | None,
) -> int:
    arm = load_arm(arm_path)
    if not arm.task.enabled:
        logger.error("weak-match-screen requires a task arm")
        return 2
    arm.apply_process_overrides()
    try:
        report = await screen_weak_matches(
            arm,
            template=template_db,
            user_id=user_id or f"probe-{arm.name}",
            queries_path=queries_path,
        )
    except ScreenPreconditionError as exc:
        logger.error("%s", exc)
        return 2
    write_weak_match_screen_report(report, out_dir)
    _print_weak_match_screen(report)
    # Exit 1 = a query whose offline replay missed think_about's window.
    return 0 if report.validity.matched == report.validity.checks else 1


async def _run_answers(args: argparse.Namespace) -> int:
    # One branch for both answer commands keeps main under the C901 cap.
    if args.cmd == "recall-answers":
        return await _run_recall_answers(
            args.arm,
            template_db=args.template_db,
            queries_path=args.queries,
            out_dir=args.out_dir,
            user_id=args.user_id,
        )
    return await _run_weak_label_answers(
        args.arm,
        template_db=args.template_db,
        out_dir=args.out_dir,
        user_id=args.user_id,
    )


async def _run_weak_label_answers(
    arm_path: Path, *, template_db: str, out_dir: Path, user_id: str | None
) -> int:
    arm = load_arm(arm_path)
    if not arm.task.enabled or not arm.task.answer_model:
        logger.error("weak-label-answers requires a task arm with an answer_model")
        return 2
    # The provider reads extraction timeout, retries and settings from process config.
    arm.apply_process_overrides()
    try:
        report = await compare_weak_labels(
            arm,
            template=template_db,
            user_id=user_id or f"probe-{arm.name}",
            provider=PydanticAIProvider(model=arm.task.answer_model),
        )
    except ScreenPreconditionError as exc:
        logger.error("%s", exc)
        return 2
    except MemorySDKError:
        logger.exception("weak-label-answers failed; no artifact written")
        return 1
    write_weak_label_answers_report(report, out_dir)
    _print_weak_label_answers(report)
    # Exit 1 = a question whose offline replay missed think_about's window.
    return 0 if report.validity.matched == report.validity.checks else 1


async def _run_recall_answers(
    arm_path: Path,
    *,
    template_db: str,
    queries_path: Path,
    out_dir: Path,
    user_id: str | None,
) -> int:
    arm = load_arm(arm_path)
    if not arm.task.enabled or not arm.task.answer_model:
        logger.error("recall-answers requires a task arm with an answer_model")
        return 2
    # The provider reads extraction timeout, retries and settings from process config.
    arm.apply_process_overrides()
    try:
        report = await read_recall_answers(
            arm,
            template=template_db,
            user_id=user_id or f"probe-{arm.name}",
            queries_path=queries_path,
            provider=PydanticAIProvider(model=arm.task.answer_model),
        )
    except ScreenPreconditionError as exc:
        logger.error("%s", exc)
        return 2
    except MemorySDKError:
        logger.exception("recall-answers failed; no artifact written")
        return 1
    write_recall_answers_report(report, out_dir)
    _print_recall_answers(report)
    # Exit 1 = a query whose offline replay missed think_about's window.
    return 0 if report.validity.matched == report.validity.checks else 1


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

    # Story-3 spike: floor matches the production recall floor so the below-floor
    # band is exactly the set the live gate drops.
    floor = float(arm.to_memory_config().get("persona.recall_relevance_floor", 0.65))
    lift = compute_situational_lift(report, floor=floor)
    write_situational_lift(lift, out_dir)
    _print_situational_lift(lift)
    return 0


async def _run_situational_surfacing_arm(
    arm: Arm, provider: PydanticAIProvider, out_dir: Path
) -> int:
    runner = SituationalSurfacingArmRunner.from_arm(arm, provider)
    report = await runner.run_measurement()
    write_situational_surfacing_report(report, out_dir)
    _print_situational_surfacing_substrate(arm, report)

    # Bridge measurement: floor matches the production recall floor so the
    # below-floor band is exactly what the live gate drops. compute_situational_lift
    # reads a SurfacingRunReport-shaped object -- adapt the queries across.
    if report.queries:
        floor = float(
            arm.to_memory_config().get("persona.recall_relevance_floor", 0.65)
        )
        lift = compute_situational_lift(
            SurfacingRunReport(
                arm_name=report.arm_name,
                model=report.model,
                seeded_traces=report.seeded_traces,
                promoted_facts=report.promoted_facts,
                queries=report.queries,
            ),
            floor=floor,
        )
        write_situational_lift(lift, out_dir)
        _print_situational_lift(lift)

    # Slice-1c: ground-truth-aware per-case breakdown over the associative corpus.
    if arm.surfacing.ground_truth_path and report.queries:
        breakdown = _classify_arm_cases(arm, report)
        write_case_breakdown(breakdown, out_dir)
        _print_case_breakdown(breakdown)
    return 0


def _classify_arm_cases(arm: Arm, report: SituationalSurfacingReport) -> CaseBreakdown:
    queries = load_query_corpus(arm.surfacing.query_corpus_path).entries
    gt = load_ground_truth(arm.surfacing.ground_truth_path)
    groups = load_seed_groups(arm.surfacing.seed_groups_path)
    surface = {q.id: set(q.relevant_trace_ids) for q in queries}
    forbid = {g.query_id: set(g.forbid) for g in gt}
    grouped = {m for grp in groups for m in grp.member_trace_ids}
    cfg = arm.to_memory_config()
    floor = float(cfg.get("persona.recall_relevance_floor", 0.65))
    # Refined slice-2 simulation: gate the bridge on propagated_sim so only
    # strong situational activation recovers a below-floor fact.
    activation_floor = float(cfg.get("persona.bridge_activation_floor", 0.0))
    # Production-faithful cut: persona.max_facts_per_query packaged default is 5,
    # even though the arm raises it to expose the full statistical cohort. The
    # case breakdown is what slice-2 would actually do; the lift is the wide view.
    return classify_cases(
        report.queries,
        surface_by_query=surface,
        forbid_by_query=forbid,
        grouped=grouped,
        floor=floor,
        top_k=5,
        activation_floor=activation_floor,
    )


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


def _print_task_summary(arm: Arm, answer_model: str, summary: TaskAggregate) -> None:
    print(f"\nArm: {arm.name} (task)", file=sys.stderr)
    print(
        f"  answer_model={answer_model}  runs={summary.runs}"
        f"  density_tier={arm.task.density_tier}",
        file=sys.stderr,
    )
    print(
        f"  overall: with={summary.overall.with_memory_mean:.3f}"
        f"  without={summary.overall.without_memory_mean:.3f}"
        f"  delta={summary.overall.delta_mean:+.3f}",
        file=sys.stderr,
    )
    print("  per tier label (with / without / delta):", file=sys.stderr)
    for label, acc in sorted(summary.per_label.items()):
        print(
            f"    {label:5s} {acc.with_memory_mean:.3f}"
            f" / {acc.without_memory_mean:.3f}"
            f" / {acc.delta_mean:+.3f}"
            f"  (n={acc.questions_per_run})",
            file=sys.stderr,
        )
    print(
        f"  fabrication: with={summary.fabrication_rate_with:.3f}"
        f"  without={summary.fabrication_rate_without:.3f}",
        file=sys.stderr,
    )


def _print_verdict_oracle(oracle: VerdictOracleReport) -> None:
    print(
        "  verdict oracle (used = alias-bearing shown line + correct answer):",
        file=sys.stderr,
    )
    print(
        f"    facts surfaced/used    = {oracle.facts_surfaced}/{oracle.facts_used}",
        file=sys.stderr,
    )
    print(
        f"    thoughts surfaced/used = {oracle.thoughts_surfaced}"
        f"/{oracle.thoughts_used}",
        file=sys.stderr,
    )
    print(
        f"    correct-with={oracle.correct_with_total}"
        f"  unattributed={oracle.unattributed_correct}",
        file=sys.stderr,
    )
    print(
        f"    would-promote (usage-as-mention): {len(oracle.would_promote)}",
        file=sys.stderr,
    )
    for entry in oracle.would_promote:
        print(f"      {entry}", file=sys.stderr)


def _print_task_verify(arm: Arm, verify: TaskVerifyReport) -> None:
    print(f"\nArm: {arm.name} (task reachability)", file=sys.stderr)
    held = len(verify.checks) - len(verify.drifted)
    print(
        f"  checks={len(verify.checks)}  held={held}  drifted={len(verify.drifted)}",
        file=sys.stderr,
    )
    for c in verify.drifted:
        print(
            f"    DRIFT [{c.question_id}] {c.tier_label}: {c.detail}",
            file=sys.stderr,
        )


def _print_fact_audit(arm: Arm, report: FactAuditReport) -> None:
    print(f"\nArm: {arm.name} (fact-channel audit)", file=sys.stderr)
    print(
        f"  model={report.model}  rounds={report.rounds}"
        f"  confidence_threshold={report.confidence_threshold:.2f}",
        file=sys.stderr,
    )
    stages: dict[str, Counter[str]] = {}
    for rnd in report.round_records:
        for o in rnd.outcomes:
            if o.gate_stage:
                stages.setdefault(o.tail_id, Counter())[o.gate_stage] += 1
    print("  per tail (promoted / gated / never):", file=sys.stderr)
    for tail, counts in report.per_tail.items():
        qid = next(
            (
                o.question_id
                for rnd in report.round_records
                for o in rnd.outcomes
                if o.tail_id == tail
            ),
            "?",
        )
        stage = (
            ", ".join(f"{s}={n}" for s, n in stages[tail].most_common())
            if tail in stages
            else ""
        )
        print(
            f"    {tail:10s} [{qid}]  {counts.get('promoted', 0)}"
            f" / {counts.get('gated', 0)}"
            f" / {counts.get('never_extracted', 0)}"
            f"{'  gates: ' + stage if stage else ''}",
            file=sys.stderr,
        )
    for rnd in report.round_records:
        by_status = ", ".join(
            f"{s}={n}" for s, n in sorted(rnd.facts_by_status.items())
        )
        print(
            f"  flood r{rnd.round_index}: seeded={rnd.seeded}"
            f"  relations={rnd.relations_total}  facts={rnd.facts_total}"
            f"  ({by_status})"
            + (
                f"  ingest_failures={rnd.ingest_failures}"
                if rnd.ingest_failures
                else ""
            ),
            file=sys.stderr,
        )


def _print_entity_reach(report: EntityReachReport) -> None:
    print(
        f"\nArm: {report.arm_name} (entity reach)  traces={report.store_traces}"
        f"  entities={report.entities}"
        f"  single_trace={report.single_trace_entities}"
    )
    for rule in report.rules:
        t3 = [q for q in rule.questions if q.tier_label == "t3"]
        reached = [q.candidates for q in t3 if q.candidates is not None]
        span = f"{min(reached)}-{max(reached)}" if reached else "-"
        labels = "  ".join(f"{k}={n}" for k, n in rule.label_counts.items())
        print(
            f"{rule.rule:16s} pairs={rule.pair_count:<4d} {labels}"
            f"  t3_reach={len(reached)}/{len(t3)}  candidates={span}"
        )


def _print_fusion_screen(report: FusionScreenReport) -> None:
    print(
        f"\nStore: {report.template_db}  user={report.user_id}"
        f"  traces={report.store_traces}  tier={report.density_tier}"
    )
    for v in report.validity:
        print(
            f"validity {v.pool}: {v.matched}/{v.checks}"
            f"  concept_appended={v.concept_appended}"
        )
    ref = report.reference
    print(
        f"reference {ref.path}: {ref.reference_in_window} -> {ref.screen_in_window}"
        f"  +{','.join(ref.diff.gained) or '-'}  -{','.join(ref.diff.lost) or '-'}"
    )
    for cell in report.cells:
        diff = cell.versus_prod
        misses = " ".join(
            f"{q.question_id}@{q.fused_rank or '-'}"
            for q in cell.questions
            if not q.in_window
        )
        stretch = "-" if cell.stretch is None else f"{cell.stretch:.1f}"
        print(
            f"{cell.pool:15s} {cell.form:16s} {cell.weight:.2f}"
            f"  {cell.required_in_window:2d}/{len(cell.questions)}"
            f"  +{','.join(diff.gained) or '-'}  -{','.join(diff.lost) or '-'}"
            f"  clamp={cell.slots_at_clamp}  slot1={len(cell.slot1_changed)}"
            f"  stretch={stretch}  dropped={cell.concepts_dropped}  miss: {misses}"
        )


def _print_weak_match_screen(report: WeakMatchScreenReport) -> None:
    v = report.validity
    print(
        f"\nStore: {report.template_db}  user={report.user_id}"
        f"  traces={report.store_traces}  validity={v.matched}/{v.checks}"
        f"  required_in_window={len(report.required_in_window)}"
        f"  delivered_mean={report.delivered_mean:.2f}"
    )
    print(
        f"{'rule':15s} {'param':>5s} {'weak':>5s} {'lost tok/dir':>12s}"
        f" {'none strong':>11s} {'cost cur':>8s} {'cost cut':>8s}"
    )
    for cell in report.cells:
        lost = cell.required_lost
        param = "-" if cell.parameter is None else f"{cell.parameter:.2f}"
        print(
            f"{cell.rule:15s} {param:>5s} {cell.weak_total:5d}"
            f" {f'{len(lost.token)}/{len(lost.direct)}':>12s}"
            f" {cell.none_tier_strong_total:11d}"
            f" {cell.cost_current_mean:8.1f} {cell.cost_cut_mean:8.1f}"
        )


def _print_weak_label_answers(report: WeakLabelAnswersReport) -> None:
    n = report.repeats
    v = report.validity
    print(
        f"\nStore: {report.template_db}  user={report.user_id}"
        f"  validity={v.matched}/{v.checks}  repeats={n}"
        f"  noise_floor={report.noise_floor}"
    )
    for q in report.questions:
        differs = q.current.correct != q.labelled.correct
        if q.identical or not (differs or q.required_weak):
            continue
        print(
            f"{q.question_id:8s} {q.tier_label:4s} weak={q.weak}"
            f"  required_weak={','.join(q.required_weak) or '-'}"
            f"  current {q.current.correct}/{n}  labelled {q.labelled.correct}/{n}"
        )
    _print_weak_label_totals(report)


def _correct_totals(
    questions: list[QuestionAnswers], repeats: int
) -> tuple[int, int, int]:
    return (
        sum(q.current.correct for q in questions),
        sum(q.labelled.correct for q in questions),
        len(questions) * repeats,
    )


def _print_weak_label_totals(report: WeakLabelAnswersReport) -> None:
    answerable = [q for q in report.questions if q.answer_type != "abstain"]
    cur, lab, total = _correct_totals(answerable, report.repeats)
    print(f"answerable correct: current {cur}/{total}  labelled {lab}/{total}")
    # A none-tier question is an abstain question: correct means abstained.
    none_tier = [q for q in report.questions if q.tier_label == "none"]
    cur, lab, total = _correct_totals(none_tier, report.repeats)
    print(
        f"none-tier abstained: current {cur}/{total} (fabricated {total - cur})"
        f"  labelled {lab}/{total} (fabricated {total - lab})"
    )


def _print_recall_answers(report: RecallAnswersReport) -> None:
    v = report.validity
    print(
        f"\nStore: {report.template_db}  user={report.user_id}"
        f"  validity={v.matched}/{v.checks}  repeats={report.repeats}"
    )
    for q in report.queries:
        # repr keeps a multi-line response on its query's one line.
        counts = Counter(q.responses).most_common()
        answers = "  ".join(f"{n}x {response!r}" for response, n in counts)
        print(f"{q.query_id}  delivered={q.delivered}  {answers}")


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


def _print_situational_lift(lift: SituationalLift) -> None:
    print(
        f"\n  situational grounding spike (floor={lift.floor:.2f}):",
        file=sys.stderr,
    )
    print(
        f"    below-floor band: n={lift.below_floor_total}"
        f"  relevant={lift.below_floor_relevant}"
        f"  baseline_prec={lift.baseline_precision:.3f}",
        file=sys.stderr,
    )
    print(
        f"    activated:     n={lift.activated.n}"
        f"  rel={lift.activated.relevant}  prec={lift.activated.precision:.3f}",
        file=sys.stderr,
    )
    print(
        f"    not-activated: n={lift.not_activated.n}"
        f"  rel={lift.not_activated.relevant}"
        f"  prec={lift.not_activated.precision:.3f}",
        file=sys.stderr,
    )
    print(
        f"    lift={lift.lift:+.3f}  recovered_relevant={lift.recovered_relevant}"
        f"  readmitted_distractor={lift.readmitted_distractor}",
        file=sys.stderr,
    )
    print(
        f"    substrate: queries_with_activation={lift.queries_with_activation}"
        f"  distinct_activated_traces={lift.distinct_activated_traces}",
        file=sys.stderr,
    )


def _print_situational_surfacing_substrate(
    arm: Arm, report: SituationalSurfacingReport
) -> None:
    cov = report.coverage
    print(
        f"\nArm: {arm.name} (situational-surfacing substrate check)",
        file=sys.stderr,
    )
    print(
        f"  model={report.model}  seeded={report.seeded_traces}"
        f"  promoted={report.promoted_facts}",
        file=sys.stderr,
    )
    print(
        f"  facts: total={cov.total_facts}"
        f"  on_grouped_traces={cov.facts_on_grouped_traces}"
        f"  distinct_grouped_source_traces={cov.distinct_grouped_source_traces}",
        file=sys.stderr,
    )
    print("  per-group fact counts:", file=sys.stderr)
    for gid, n in sorted(cov.per_group.items()):
        print(f"    {gid}: {n}", file=sys.stderr)


def _print_case_breakdown(b: CaseBreakdown) -> None:
    print(
        "\n  per-case breakdown (Axis 1, persona-fact channel, production top-5):",
        file=sys.stderr,
    )
    print(
        f"    A vector={b.a_vector}  C bridge={b.c_bridge}"
        f"  E bypass={b.e_bypass}  anomaly={b.anomaly}",
        file=sys.stderr,
    )
    print(
        f"    C miss(grouped)={b.c_miss}  B dropped(ungrouped)={b.b_dropped}",
        file=sys.stderr,
    )
    print(
        f"    F readmit={b.f_readmit}  G violation={b.g_violation}"
        f" (bridge={b.g_bridge})  H noise={b.h_noise}"
        f"  E offtarget={b.e_offtarget}  -> bridge-clean={b.clean}",
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
