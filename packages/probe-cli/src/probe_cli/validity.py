"""Seeding validity: every check over one seeding's artifacts, before any reading."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from probe_cli.corpus import (
    load_corpus,
    load_eval_corpus,
    load_query_corpus,
    load_seed_groups,
    load_seed_traces,
)
from probe_cli.metrics import RetrievalRunReport
from probe_cli.provenance import PromptStamp
from probe_cli.representations import RepresentationsReport
from probe_cli.situational_runner import SituationalRunReport
from probe_cli.surfacing_runner import SurfacingRunReport

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from probe_cli.arm import Arm

Mode = Literal["qualification", "surfacing", "guard"]
# (count, detail naming the ids behind the count)
type Finding = tuple[int, str]

# Production's own retry path: a final failure logs its own record and leaves
# the trace degraded.
EXTRACTION_RETRY = "recollect.core: LLM extraction failed, retrying"
_ALLOWED = frozenset({EXTRACTION_RETRY})

_COMMON = ("missing_artifact", "prompt_mismatch", "swallowed_failures")
_KEYS: dict[Mode, tuple[str, ...]] = {
    "qualification": (
        *_COMMON,
        "run_files",
        "ingest_failures",
        "uningested",
        "query_ids",
        "failed_queries",
        "statement_ids",
        "degraded",
        "traces_sha256",
        "user_id",
    ),
    "surfacing": (
        *_COMMON,
        "failed_seeds",
        "unseeded",
        "degraded",
        "failed_queries",
        "query_ids",
        "fact_count",
    ),
    "guard": (
        *_COMMON,
        "run_files",
        "unseeded",
        "groups",
        "degraded_seeds",
        "eval_ids",
        "failed_evals",
        "degraded_evals",
    ),
}


class SeedingValidity(BaseModel):
    arm_name: str
    mode: Mode
    state: Literal["VALID", "INVALID"]
    # Every check, zeros included, in a fixed order.
    problems: dict[str, int]
    # One line per non-zero problem, naming ids.
    reasons: list[str]
    prompt: PromptStamp
    expected_prompt: PromptStamp


class QualificationExpected(BaseModel):
    arm_name: str
    runs: int
    trace_ids: list[str]
    query_ids: list[str]
    traces_sha256: str


class SurfacingExpected(BaseModel):
    trace_ids: list[str]
    query_ids: list[str]


class GuardExpected(BaseModel):
    runs: int
    seed_ids: list[str]
    group_ids: list[str]
    eval_ids: list[str]


@dataclass(frozen=True)
class RunFiles[T]:
    # (expected run index, report) per expected file that parsed.
    indexed: list[tuple[int, T]]
    # Expected file names absent or unparsable.
    missing: list[str]
    # Files matching the run-file pattern, expected or not.
    found: int

    @property
    def reports(self) -> list[T]:
        return [report for _, report in self.indexed]


def _named(names: Sequence[str]) -> Finding:
    return len(names), ", ".join(names)


def _differs(actual: str, expected: str) -> Finding:
    return int(actual != expected), f"{actual!r} != {expected!r}"


def _short_of(expected: int, actual: int, label: str) -> Finding:
    return abs(expected - actual), f"{label} {actual} of {expected}"


def id_problems(actual: Sequence[str], expected: Sequence[str]) -> list[str]:
    """Missing, unexpected and duplicated ids; order not checked."""
    seen = Counter(actual)
    wanted = set(expected)
    missing = [f"missing {i}" for i in expected if i not in seen]
    unexpected = [f"unexpected {i}" for i in seen if i not in wanted]
    duplicated = [f"duplicated {i}" for i, n in seen.items() for _ in range(n - 1)]
    return [*missing, *unexpected, *duplicated]


def order_problems(actual: Sequence[str], expected: Sequence[str]) -> list[str]:
    """Positions where the ids differ: a gap, an extra or a reorder each count."""
    return [
        f"#{i} {a or '-'} != {e or '-'}"
        for i, (a, e) in enumerate(zip_longest(actual, expected))
        if a != e
    ]


def _stamp_text(stamp: PromptStamp) -> str:
    return f"{stamp.version or '-'} {stamp.sha256[:12] or '-'} {stamp.chars}"


def common_findings(
    stamps: Sequence[PromptStamp],
    *,
    expected_prompt: PromptStamp,
    swallowed: Mapping[str, int],
    missing: Sequence[str],
) -> dict[str, Finding]:
    # No artifact stamp is an empty one.
    off = [s for s in stamps or [PromptStamp()] if not s.sha256 or s != expected_prompt]
    unallowed = {k: n for k, n in swallowed.items() if k not in _ALLOWED and n}
    expected = _stamp_text(expected_prompt)
    return {
        "missing_artifact": _named(missing),
        "prompt_mismatch": (
            len(off),
            "; ".join(f"{_stamp_text(s)} != {expected}" for s in off),
        ),
        "swallowed_failures": (
            sum(unallowed.values()),
            "; ".join(f"{k} x{n}" for k, n in unallowed.items()),
        ),
    }


def qualification_findings(
    runs: RunFiles[RetrievalRunReport],
    reps: RepresentationsReport | None,
    expected: QualificationExpected,
) -> dict[str, Finding]:
    found = _qualification_run_findings(runs, expected)
    if reps is None:
        return found
    statement_ids = [s.id for s in reps.statements]
    return found | {
        "statement_ids": _named(order_problems(statement_ids, expected.trace_ids)),
        "degraded": _named(reps.degraded_statements),
        "traces_sha256": _differs(reps.traces_sha256, expected.traces_sha256),
        "user_id": _differs(reps.user_id, f"probe-{expected.arm_name}"),
    }


def _qualification_run_findings(
    runs: RunFiles[RetrievalRunReport], expected: QualificationExpected
) -> dict[str, Finding]:
    pairs = runs.indexed
    short = [(i, abs(len(expected.trace_ids) - r.ingested)) for i, r in pairs]
    return {
        "run_files": _short_of(expected.runs, runs.found, "run files"),
        "ingest_failures": (
            sum(r.ingest_failures for _, r in pairs),
            ", ".join(f"r{i} {r.ingest_failures}" for i, r in pairs),
        ),
        "uningested": (
            sum(n for _, n in short),
            ", ".join(f"r{i} ingested {r.ingested}" for i, r in pairs),
        ),
        "query_ids": _named(
            [
                f"r{i} {p}"
                for i, r in pairs
                for p in id_problems(
                    [q.query_id for q in r.queries], expected.query_ids
                )
            ]
        ),
        "failed_queries": _named(
            [f"r{i} {q.query_id}" for i, r in pairs for q in r.queries if not q.success]
        ),
    }


def surfacing_findings(
    report: SurfacingRunReport | None, expected: SurfacingExpected
) -> dict[str, Finding]:
    if report is None:
        return {}
    answered = [q.query_id for q in report.queries] + report.failed_query_ids
    return {
        "failed_seeds": _named(report.failed_seed_ids),
        "unseeded": _short_of(len(expected.trace_ids), report.seeded_traces, "seeded"),
        "degraded": (report.degraded_traces, "degraded_traces"),
        "failed_queries": _named(report.failed_query_ids),
        "query_ids": _named(id_problems(answered, expected.query_ids)),
        "fact_count": _named(_fact_count_problems(report)),
    }


def _fact_count_problems(report: SurfacingRunReport) -> list[str]:
    # The lookup returns min(limit, pool) facts unless a step failed and was
    # swallowed; an unstamped limit cannot be checked.
    cap = report.max_facts_per_query
    if cap == 0:
        return [f"{q.query_id} limit unstamped" for q in report.queries]
    want = min(cap, len(report.surfacing_pool))
    return [
        f"{q.query_id} surfaced {len(q.surfaced)} of {want}"
        for q in report.queries
        if len(q.surfaced) != want
    ]


def guard_findings(
    runs: RunFiles[SituationalRunReport], expected: GuardExpected
) -> dict[str, Finding]:
    return _guard_seeding_findings(runs, expected) | _guard_eval_findings(
        runs, expected
    )


def _worst(findings: Sequence[Finding]) -> Finding:
    # Seeding-wide counts ride on every run report: the worst, never a sum.
    return max(findings, default=(0, ""))


def _guard_seeding_findings(
    runs: RunFiles[SituationalRunReport], expected: GuardExpected
) -> dict[str, Finding]:
    reports = runs.reports
    misindexed = [
        f"r{i} holds run_index {r.run_index}"
        for i, r in runs.indexed
        if r.run_index != i
    ]
    count, detail = _short_of(expected.runs, runs.found, "run files")
    seeds, groups = len(expected.seed_ids), len(expected.group_ids)
    return {
        "run_files": (count + len(misindexed), "; ".join([detail, *misindexed])),
        "unseeded": _worst(
            [_short_of(seeds, r.seed_traces_ingested, "ingested") for r in reports]
        ),
        "groups": _worst(
            [_short_of(groups, r.seed_groups_restored, "restored") for r in reports]
        ),
        "degraded_seeds": _worst(
            [(r.seed_traces_degraded, "seed_traces_degraded") for r in reports]
        ),
    }


def _guard_eval_findings(
    runs: RunFiles[SituationalRunReport], expected: GuardExpected
) -> dict[str, Finding]:
    pairs = runs.indexed
    return {
        "eval_ids": _named(
            [
                f"r{i} {p}"
                for i, r in pairs
                for p in order_problems(
                    [e.entry_id for e in r.eval_results], expected.eval_ids
                )
            ]
        ),
        "failed_evals": _named(
            [
                f"r{i} {e.entry_id}"
                for i, r in pairs
                for e in r.eval_results
                if not e.success
            ]
        ),
        "degraded_evals": _named(
            [
                f"r{i} {e.entry_id}"
                for i, r in pairs
                for e in r.eval_results
                if e.extraction_degraded
            ]
        ),
    }


def assemble(
    arm_name: str,
    mode: Mode,
    findings: Mapping[str, Finding],
    *,
    prompt: PromptStamp,
    expected_prompt: PromptStamp,
) -> SeedingValidity:
    problems = {k: findings.get(k, (0, ""))[0] for k in _KEYS[mode]}
    reasons = [f"{k}={n}: {findings[k][1]}" for k, n in problems.items() if n]
    return SeedingValidity(
        arm_name=arm_name,
        mode=mode,
        state="INVALID" if any(problems.values()) else "VALID",
        problems=problems,
        reasons=reasons,
        prompt=prompt,
        expected_prompt=expected_prompt,
    )


def seeding_mode(arm: Arm) -> Mode:
    # probe run's dispatch order: the runner that wrote the artifacts.
    if not arm.task.enabled:
        if arm.surfacing.enabled:
            return "surfacing"
        if arm.situational.enabled:
            return "guard"
        if arm.retrieval.enabled:
            return "qualification"
    raise ValueError(f"{arm.name} runs no qualification, surfacing or guard seeding")


def check_seeding(arm: Arm, *, expected_prompt: PromptStamp) -> SeedingValidity:
    """Reads the arm's evidence dir and corpora; an absent dir reads INVALID."""
    mode = seeding_mode(arm)
    out_dir = Path(arm.output.dir) / arm.name
    if mode == "qualification":
        return _check_qualification(arm, out_dir, expected_prompt)
    if mode == "surfacing":
        return _check_surfacing(arm, out_dir, expected_prompt)
    return _check_guard(arm, out_dir, expected_prompt)


def write_validity(verdict: SeedingValidity, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "validity.json"
    path.write_text(verdict.model_dump_json(indent=2), encoding="utf-8")
    return path


def validity_line(verdict: SeedingValidity) -> str:
    counts = " ".join(f"{k}={n}" for k, n in verdict.problems.items())
    return f"{verdict.state} {verdict.arm_name} {counts}"


def _read[T: BaseModel](path: Path, model: type[T]) -> T | None:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_runs[T: BaseModel](
    out_dir: Path, *, stem: str, runs: int, model: type[T]
) -> RunFiles[T]:
    indexed: list[tuple[int, T]] = []
    missing: list[str] = []
    for i in range(runs):
        name = f"{stem}_{i:03d}.json"
        report = _read(out_dir / name, model)
        if report is None:
            missing.append(name)
        else:
            indexed.append((i, report))
    found = len(list(out_dir.glob(f"{stem}_*.json")))
    return RunFiles(indexed=indexed, missing=missing, found=found)


def _check_qualification(
    arm: Arm, out_dir: Path, expected_prompt: PromptStamp
) -> SeedingValidity:
    traces_path = Path(arm.retrieval.traces_corpus_path)
    expected = QualificationExpected(
        arm_name=arm.name,
        runs=arm.runs,
        trace_ids=[e.id for e in load_corpus(traces_path).entries],
        query_ids=[
            q.id for q in load_query_corpus(arm.retrieval.query_corpus_path).entries
        ],
        traces_sha256=hashlib.sha256(traces_path.read_bytes()).hexdigest(),
    )
    runs = _read_runs(
        out_dir, stem="retrieval_run", runs=arm.runs, model=RetrievalRunReport
    )
    reps = _read(out_dir / "representations.json", RepresentationsReport)
    # Each run file carries its own run's span.
    swallowed: Counter[str] = Counter()
    for report in runs.reports:
        swallowed.update(report.swallowed_failures)
    stamps = [r.prompt for r in runs.reports]
    missing = [*runs.missing, *(["representations.json"] if reps is None else [])]
    findings = common_findings(
        stamps, expected_prompt=expected_prompt, swallowed=swallowed, missing=missing
    ) | qualification_findings(runs, reps, expected)
    return assemble(
        arm.name,
        "qualification",
        findings,
        prompt=stamps[0] if stamps else PromptStamp(),
        expected_prompt=expected_prompt,
    )


def _check_surfacing(
    arm: Arm, out_dir: Path, expected_prompt: PromptStamp
) -> SeedingValidity:
    expected = SurfacingExpected(
        trace_ids=[e.id for e in load_corpus(arm.surfacing.traces_corpus_path).entries],
        query_ids=[
            q.id for q in load_query_corpus(arm.surfacing.query_corpus_path).entries
        ],
    )
    report = _read(out_dir / "surfacing_run.json", SurfacingRunReport)
    stamps = [] if report is None else [report.prompt]
    findings = common_findings(
        stamps,
        expected_prompt=expected_prompt,
        swallowed={} if report is None else report.swallowed_failures,
        missing=["surfacing_run.json"] if report is None else [],
    ) | surfacing_findings(report, expected)
    return assemble(
        arm.name,
        "surfacing",
        findings,
        prompt=stamps[0] if stamps else PromptStamp(),
        expected_prompt=expected_prompt,
    )


def _check_guard(
    arm: Arm, out_dir: Path, expected_prompt: PromptStamp
) -> SeedingValidity:
    s = arm.situational
    expected = GuardExpected(
        runs=arm.runs,
        seed_ids=[t.id for t in load_seed_traces(s.seed_traces_path)],
        group_ids=[g.group_id for g in load_seed_groups(s.seed_groups_path)],
        eval_ids=[e.id for e in load_eval_corpus(s.eval_corpus_path)],
    )
    runs = _read_runs(
        out_dir, stem="situational_run", runs=arm.runs, model=SituationalRunReport
    )
    # Seeding-wide: every run report carries the same mapping.
    swallowed: dict[str, int] = {}
    for report in runs.reports:
        for key, n in report.swallowed_failures.items():
            swallowed[key] = max(swallowed.get(key, 0), n)
    stamps = [r.prompt for r in runs.reports]
    findings = common_findings(
        stamps,
        expected_prompt=expected_prompt,
        swallowed=swallowed,
        missing=runs.missing,
    ) | guard_findings(runs, expected)
    return assemble(
        arm.name,
        "guard",
        findings,
        prompt=stamps[0] if stamps else PromptStamp(),
        expected_prompt=expected_prompt,
    )
