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
