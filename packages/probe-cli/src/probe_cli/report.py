"""Per-run + aggregate JSON serialization."""

from __future__ import annotations

import logging
from pathlib import Path

from probe_cli.metrics import (
    AggregateMetrics,
    RetrievalAggregate,
    RetrievalRunReport,
    RunReport,
)
from probe_cli.situational_metrics import SituationalAggregate
from probe_cli.situational_runner import SituationalRunReport
from probe_cli.surfacing_cases import CaseBreakdown
from probe_cli.surfacing_metrics import SurfacingMetrics
from probe_cli.surfacing_runner import SurfacingRunReport
from probe_cli.surfacing_situational import SituationalLift
from probe_cli.surfacing_situational_runner import SituationalSurfacingReport
from probe_cli.task_metrics import TaskAggregate
from probe_cli.task_runner import TaskRunReport, TaskVerifyReport

logger = logging.getLogger(__name__)


def write_run_report(report: RunReport, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"run_{report.run_index:03d}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_summary(metrics: AggregateMetrics, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "summary.json"
    path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_retrieval_run_report(
    report: RetrievalRunReport, output_dir: Path | str
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"retrieval_run_{report.run_index:03d}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_retrieval_summary(
    metrics: RetrievalAggregate, output_dir: Path | str
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "retrieval_summary.json"
    path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_situational_run_report(
    report: SituationalRunReport, output_dir: Path | str
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"situational_run_{report.run_index:03d}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_situational_summary(
    metrics: SituationalAggregate, output_dir: Path | str
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "situational_summary.json"
    path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_surfacing_run_report(
    report: SurfacingRunReport, output_dir: Path | str
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "surfacing_run.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_surfacing_summary(
    metrics: SurfacingMetrics, output_dir: Path | str
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "surfacing_summary.json"
    path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_situational_lift(
    lift: SituationalLift, output_dir: Path | str
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "surfacing_situational_lift.json"
    path.write_text(lift.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_situational_surfacing_report(
    report: SituationalSurfacingReport, output_dir: Path | str
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "situational_surfacing_substrate.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_case_breakdown(
    breakdown: CaseBreakdown, output_dir: Path | str
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "associative_case_breakdown.json"
    path.write_text(breakdown.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_task_run_report(report: TaskRunReport, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"task_run_{report.run_index:03d}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_task_summary(summary: TaskAggregate, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "task_summary.json"
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def write_task_verify_report(
    report: TaskVerifyReport, output_dir: Path | str
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "task_reachability.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return path
