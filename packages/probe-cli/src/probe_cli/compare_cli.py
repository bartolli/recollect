"""probe drift, surfacing-compare, guard-compare: the frozen comparison rule over
valid seedings."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError
from recollect.config import config as sdk_config

from probe_cli.comparison import MeasureRead, Seedings, UnvalidatedInput
from probe_cli.corpus import load_query_corpus
from probe_cli.drift import (
    DriftInput,
    DriftInputs,
    DriftReport,
    StatementDrift,
    compare_drift,
    load_reader_key,
)
from probe_cli.guard_compare import (
    GuardComparison,
    GuardSeeding,
    compare_guard,
    guard_seeding,
)
from probe_cli.representations import RepresentationsReport
from probe_cli.situational_metrics import SituationalAggregate
from probe_cli.surfacing_compare import (
    SurfacingComparison,
    SurfacingInput,
    compare_surfacing,
    load_reading,
)
from probe_cli.surfacing_metrics import SurfacingMetrics
from probe_cli.surfacing_runner import SurfacingRunReport

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

_FLOOR_KEY = "persona.recall_relevance_floor"
_THRESHOLD_KEY = "persona.confidence_threshold"
_ABSENT = "absent"


class _Validity(BaseModel):
    # Written by `probe validity`; only the state is read here, verbatim, so a
    # refusal names whatever state was found.
    state: str


def add_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    drift = sub.add_parser(
        "drift", help="Per-statement drift of two candidate representations."
    )
    drift.add_argument("--control", required=True, type=Path)
    drift.add_argument("--candidate", required=True, type=Path, action="append")
    drift.add_argument("--reader-key", required=True, type=Path)
    _add_common(drift, "representations.json")
    surfacing = sub.add_parser(
        "surfacing-compare", help="Surfacing calls of two candidate seedings."
    )
    surfacing.add_argument("--config", required=True, type=Path, help="reading.toml")
    _add_seedings(surfacing)
    _add_common(surfacing, "seeding directory")
    guard = sub.add_parser(
        "guard-compare", help="Situational guard calls of two candidate seedings."
    )
    _add_seedings(guard)
    _add_common(guard, "seeding directory")


def _add_seedings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--control", required=True, type=Path, action="append")
    parser.add_argument("--candidate", required=True, type=Path, action="append")


def _add_common(parser: argparse.ArgumentParser, kind: str) -> None:
    parser.add_argument("--fresh-control", type=Path, help=f"Fresh control {kind}.")
    parser.add_argument("--fresh-candidate", type=Path, help=f"Fresh candidate {kind}.")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON file.")
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help=(
            "Dry runs on artifacts older than `probe validity`: admit inputs with "
            "no validity.json. A validity.json not saying VALID is still refused."
        ),
    )
    # main() reads both before dispatch; these commands touch no env or DB.
    parser.set_defaults(verbose=False, env_file=None)


async def dispatch(args: argparse.Namespace) -> int | None:
    commands: dict[str, Callable[[argparse.Namespace], int]] = {
        "drift": _run_drift,
        "surfacing-compare": _run_surfacing,
        "guard-compare": _run_guard,
    }
    run = commands.get(args.cmd)
    if run is None:
        return None
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        logger.error("%s refused: %s", args.cmd, exc)
        return 2


def _check_two(flag: str, values: Sequence[Path]) -> None:
    if len(values) != 2:
        raise ValueError(f"expected two {flag}, got {len(values)}")


def _fresh_paths(args: argparse.Namespace) -> list[Path]:
    fresh = [p for p in (args.fresh_control, args.fresh_candidate) if p is not None]
    if len(fresh) == 1:
        raise ValueError("--fresh-control and --fresh-candidate go together")
    return fresh


def _seeding_paths(args: argparse.Namespace) -> list[Path]:
    _check_two("--control", args.control)
    _check_two("--candidate", args.candidate)
    return [*args.control, *args.candidate, *_fresh_paths(args)]


def _validity(directory: Path) -> str:
    path = directory / "validity.json"
    if not path.is_file():
        return _ABSENT
    try:
        return _Validity.model_validate_json(path.read_text("utf-8")).state
    except ValidationError:
        return "unreadable"


def _admit(directories: Sequence[Path], *, allow: bool) -> list[UnvalidatedInput]:
    # --allow-unvalidated admits only an artifact older than `probe validity`:
    # a seeding known not to be valid is never read.
    states = [(d, _validity(d)) for d in directories]
    admissible = {"VALID", _ABSENT} if allow else {"VALID"}
    refused = [(d, state) for d, state in states if state not in admissible]
    if refused:
        named = ", ".join(f"{d} (validity: {state})" for d, state in refused)
        raise ValueError(f"not a valid seeding: {named}")
    admitted = [d for d, state in states if state == _ABSENT]
    for d in admitted:
        logger.warning("admitted input without validity.json: %s", d)
    return [UnvalidatedInput(path=str(d)) for d in admitted]


def _sdk_float(key: str) -> float:
    value = sdk_config.get(key)
    if value is None:
        raise ValueError(f"SDK config has no {key}")
    return float(value)


def _write(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", path)


def _loaded(
    args: argparse.Namespace, load: Callable[[Path], BaseModel]
) -> dict[str, object]:
    return {
        "control": [load(p) for p in args.control],
        "candidate": [load(p) for p in args.candidate],
        "fresh_control": load(args.fresh_control) if args.fresh_control else None,
        "fresh_candidate": load(args.fresh_candidate) if args.fresh_candidate else None,
    }


def _surfacing_input(directory: Path) -> SurfacingInput:
    run = (directory / "surfacing_run.json").read_text("utf-8")
    summary_path = directory / "surfacing_summary.json"
    summary = (
        SurfacingMetrics.model_validate_json(summary_path.read_text("utf-8"))
        if summary_path.is_file()
        else None
    )
    return SurfacingInput(
        label=str(directory),
        report=SurfacingRunReport.model_validate_json(run),
        summary=summary,
    )


def _guard_seeding(directory: Path) -> GuardSeeding:
    text = (directory / "situational_summary.json").read_text("utf-8")
    return guard_seeding(str(directory), SituationalAggregate.model_validate_json(text))


def _drift_input(path: Path) -> DriftInput:
    report = RepresentationsReport.model_validate_json(path.read_text("utf-8"))
    return DriftInput(label=str(path), report=report)


def _run_surfacing(args: argparse.Namespace) -> int:
    unvalidated = _admit(_seeding_paths(args), allow=args.allow_unvalidated)
    floor = _sdk_float(_FLOOR_KEY)
    reading = load_reading(args.config)
    queries = load_query_corpus(reading.queries).entries
    inputs = Seedings[SurfacingInput].model_validate(_loaded(args, _surfacing_input))
    comparison = compare_surfacing(inputs, queries, reading=reading, floor=floor)
    comparison = comparison.model_copy(update={"unvalidated_inputs": unvalidated})
    _write(args.out, comparison)
    _print_surfacing(comparison)
    return 0


def _run_guard(args: argparse.Namespace) -> int:
    unvalidated = _admit(_seeding_paths(args), allow=args.allow_unvalidated)
    inputs = Seedings[GuardSeeding].model_validate(_loaded(args, _guard_seeding))
    comparison = compare_guard(inputs).model_copy(
        update={"unvalidated_inputs": unvalidated}
    )
    _write(args.out, comparison)
    _print_guard(comparison)
    return 0


def _run_drift(args: argparse.Namespace) -> int:
    _check_two("--candidate", args.candidate)
    paths = [args.control, *args.candidate, *_fresh_paths(args)]
    # Validity is the sibling validity.json of each representations.json.
    unvalidated = _admit([p.parent for p in paths], allow=args.allow_unvalidated)
    threshold = _sdk_float(_THRESHOLD_KEY)
    key = load_reader_key(args.reader_key)
    inputs = DriftInputs(
        control=_drift_input(args.control),
        candidate=[_drift_input(p) for p in args.candidate],
        fresh_control=_drift_input(args.fresh_control) if args.fresh_control else None,
        fresh_candidate=(
            _drift_input(args.fresh_candidate) if args.fresh_candidate else None
        ),
    )
    report = compare_drift(inputs, key=key, threshold=threshold).model_copy(
        update={"unvalidated_inputs": unvalidated}
    )
    _write(args.out, report)
    _print_drift(report)
    return 0


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _roles[T: BaseModel](seedings: Seedings[T]) -> list[tuple[str, T]]:
    roles = [(f"control {i + 1}", s) for i, s in enumerate(seedings.control)]
    roles += [(f"candidate {i + 1}", s) for i, s in enumerate(seedings.candidate)]
    if seedings.fresh_control is not None and seedings.fresh_candidate is not None:
        roles += [("fresh control", seedings.fresh_control)]
        roles += [("fresh candidate", seedings.fresh_candidate)]
    return roles


def _print_measures(measures: Sequence[MeasureRead]) -> None:
    _emit(f"  {'measure':<26} {'control':<12} {'candidate':<12} {'call':<11} fresh")
    for m in measures:
        control = " ".join(_fmt(v) for v in m.control)
        candidate = " ".join(_fmt(v) for v in m.candidate)
        fresh = "-"
        if m.confirmation is not None:
            fresh = (
                f"{_fmt(m.fresh_control)} {_fmt(m.fresh_candidate)} {m.confirmation}"
            )
        reason = f"  ({m.reason})" if m.reason else ""
        values = f"{control:<12} {candidate:<12} {m.call:<11}"
        _emit(f"  {m.name:<26} {values} {fresh}{reason}")


def _print_unvalidated(unvalidated: Sequence[UnvalidatedInput]) -> None:
    for item in unvalidated:
        _emit(f"  UNVALIDATED INPUT {item.path} (validity: {item.state})")


def _print_surfacing(c: SurfacingComparison) -> None:
    _emit(f"surfacing-compare  floor={c.floor:.2f}  verdict={c.verdict}")
    head = f"{'cov_all':>8} {'cov_wo':>8} {'prec_fl':>8} {'bypass':>7} {'facts':>6}"
    _emit(f"  {'seeding':<16} {'arm':<28} {head}")
    for role, s in _roles(c.seedings):
        bypass = f"{s.gate.relevant_bypass_passes}/{s.gate.bypass_passes}"
        values = (
            f"{s.coverage_all:>8.3f} {s.coverage_without_overlap:>8.3f} "
            f"{_fmt(s.precision_floor):>8} {bypass:>7} {s.fact_population.total:>6}"
        )
        _emit(f"  {role:<16} {s.arm_name:<28} {values}")
    _print_measures(c.measures)
    for loss in c.named_losses:
        causes = ",".join(cause or "-" for cause in loss.candidate_causes)
        confirmation = loss.confirmation or "-"
        pair = f"{loss.query_id}/{loss.required_id}"
        _emit(f"  named loss  {pair}  causes={causes}  confirmation={confirmation}")
    for gain in c.named_gains:
        flag = "  exemplar_overlap" if gain.exemplar_overlap else ""
        _emit(f"  named gain  {gain.query_id}/{gain.required_id}{flag}")
    _print_unvalidated(c.unvalidated_inputs)


def _print_guard(c: GuardComparison) -> None:
    _emit(f"guard-compare  verdict={c.verdict}")
    _emit(f"  {'seeding':<16} {'arm':<28} {'action':>7} {'ext_tgt':>7} {'none':>6}")
    for role, s in _roles(c.seedings):
        values = f"{s.action_accuracy:>7.3f} {s.extension_target_accuracy:>7.3f}"
        _emit(f"  {role:<16} {s.arm_name:<28} {values} {s.none_rate:>6.3f}")
    if c.verdict == "uninformative":
        bound = c.uninformative_at_or_below
        _emit(f"  a control action accuracy is at or below {bound:.2f}: no call")
    _print_measures(c.measures)
    _print_unvalidated(c.unvalidated_inputs)


def _print_drift(r: DriftReport) -> None:
    _emit(f"drift  threshold={r.threshold:.2f}  control={r.control}")
    head = f"{'fact_type':<28} {'significance':<15} {'rel':<6} {'at_thr':<6}"
    _emit(f"  {'statement':<14} {head} events")
    for row in zip(*(c.statements for c in r.candidates), strict=True):
        _print_drift_row(row)
    for loss in r.relation_losses:
        at = "/".join(str(n) for n in loss.candidate_at_threshold)
        confirmation = loss.confirmation or "-"
        _emit(
            f"  relation loss  {loss.statement_id}  {loss.certainty}  {loss.status}"
            f"  at_thr={loss.control_at_threshold}/{at}  confirmation={confirmation}"
        )
    _print_unvalidated(r.unvalidated_inputs)


def _print_drift_row(row: Sequence[StatementDrift]) -> None:
    # Values read control first, then each candidate in input order.
    shapes = [row[0].control, *(d.candidate for d in row)]
    fact_types = "/".join(s.fact_type for s in shapes)
    significance = " ".join(f"{s.significance:.2f}" for s in shapes)
    relations = "/".join(str(s.relations) for s in shapes)
    at = "/".join(str(s.relations_at_threshold) for s in shapes)
    events = "; ".join(
        f"k{i + 1}:{','.join(d.events)}" for i, d in enumerate(row) if d.events
    )
    values = f"{fact_types:<28} {significance:<15} {relations:<6} {at:<6}"
    _emit(f"  {row[0].statement_id:<14} {values} {events}")
