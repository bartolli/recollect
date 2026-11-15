"""Guard validity: situational run files of one seeding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from probe_cli.arm import Arm, CorpusConfig, OutputConfig, SituationalConfig
from probe_cli.provenance import PromptStamp
from probe_cli.report import write_situational_run_report
from probe_cli.situational_runner import EvalResult, SituationalRunReport
from probe_cli.validity import EXTRACTION_RETRY, check_seeding

if TYPE_CHECKING:
    from pathlib import Path

_STAMP = PromptStamp(version="1.3.0", sha256="c" * 64, chars=13462)
_GROUP = (
    '{"group_id": "G1", "person_ref": "p", "situation": "s", "implications": ["i"],'
    ' "significance": 0.5, "member_trace_ids": ["s1"]}\n'
)
_KEYS = [
    "missing_artifact",
    "prompt_mismatch",
    "swallowed_failures",
    "run_files",
    "unseeded",
    "groups",
    "degraded_seeds",
    "eval_ids",
    "failed_evals",
    "degraded_evals",
]


def _arm(tmp_path: Path) -> Arm:
    paths = [tmp_path / n for n in ("seeds.jsonl", "groups.jsonl", "evals.jsonl")]
    seeds = '{"id": "s1", "text": "a"}\n{"id": "s2", "text": "b"}\n'
    evals = "".join(
        f'{{"id": "{e}", "text": "{e}", "expected_action": "none"}}\n'
        for e in ("e1", "e2")
    )
    for path, text in zip(paths, (seeds, _GROUP, evals), strict=True):
        path.write_text(text, encoding="utf-8")
    situational = SituationalConfig(
        enabled=True,
        seed_traces_path=str(paths[0]),
        seed_groups_path=str(paths[1]),
        eval_corpus_path=str(paths[2]),
    )
    return Arm(
        name="guard-s1",
        runs=2,
        corpus=CorpusConfig(path=""),
        output=OutputConfig(dir=str(tmp_path / "out")),
        situational=situational,
    )


def _evals(*ids: str) -> list[EvalResult]:
    return [EvalResult(entry_id=e, expected_action="none") for e in ids]


def _report(run_index: int, **update: object) -> SituationalRunReport:
    report = SituationalRunReport(
        arm_name="guard-s1",
        run_index=run_index,
        model="m",
        seed_groups_restored=1,
        seed_traces_ingested=2,
        eval_results=_evals("e1", "e2"),
        prompt=_STAMP,
        swallowed_failures={EXTRACTION_RETRY: 3},
    )
    return report.model_copy(update=update)


def _write(tmp_path: Path, arm: Arm, *reports: SituationalRunReport) -> None:
    for report in reports:
        write_situational_run_report(report, tmp_path / "out" / arm.name)


_FAILED = [EvalResult(entry_id="e1", expected_action="none", success=False)]
_DEGRADED = [
    EvalResult(entry_id="e1", expected_action="none", extraction_degraded=True)
]
_UPDATES: dict[str, dict[str, object]] = {
    "prompt_mismatch": {"prompt": PromptStamp(version="1.2.1")},
    "swallowed_failures": {"swallowed_failures": {"recollect.core: lost": 1}},
    "unseeded": {"seed_traces_ingested": 1},
    "groups": {"seed_groups_restored": 0},
    "degraded_seeds": {"seed_traces_degraded": 1},
    "eval_ids": {"eval_results": _evals("e2", "e1")},
    "failed_evals": {"eval_results": _FAILED + _evals("e2")},
    "degraded_evals": {"eval_results": _DEGRADED + _evals("e2")},
}


def test_a_complete_seeding_is_valid(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    _write(tmp_path, arm, _report(0), _report(1))

    verdict = check_seeding(arm, expected_prompt=_STAMP)

    assert (verdict.state, verdict.mode, verdict.reasons) == ("VALID", "guard", [])
    assert list(verdict.problems) == _KEYS


@pytest.mark.parametrize("key", _UPDATES)
def test_each_report_problem_makes_the_seeding_invalid(
    tmp_path: Path, key: str
) -> None:
    arm = _arm(tmp_path)
    _write(tmp_path, arm, _report(0), _report(1, **_UPDATES[key]))

    verdict = check_seeding(arm, expected_prompt=_STAMP)

    assert verdict.state == "INVALID"
    assert {k for k, n in verdict.problems.items() if n} == {key}
    assert verdict.reasons[0].startswith(f"{key}=")


def test_seeding_wide_counts_are_not_summed_across_run_files(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    wide = {
        "swallowed_failures": {"recollect.core: lost": 1},
        "seed_traces_degraded": 2,
    }
    _write(tmp_path, arm, _report(0, **wide), _report(1, **wide))

    verdict = check_seeding(arm, expected_prompt=_STAMP)

    assert verdict.problems["swallowed_failures"] == 1
    assert verdict.problems["degraded_seeds"] == 2


@pytest.mark.parametrize(
    "reports",
    [
        [_report(0), _report(1), _report(2)],
        [_report(0), _report(0)],
    ],
)
def test_run_files_must_number_arm_runs_from_zero(
    tmp_path: Path, reports: list[SituationalRunReport]
) -> None:
    arm = _arm(tmp_path)
    out = tmp_path / "out" / arm.name
    write_situational_run_report(reports[0], out)
    for i, report in enumerate(reports[1:], start=1):
        name = f"situational_run_{i:03d}.json"
        (out / name).write_text(report.model_dump_json(), encoding="utf-8")

    verdict = check_seeding(arm, expected_prompt=_STAMP)

    assert {k for k, n in verdict.problems.items() if n} == {"run_files"}


def test_a_missing_run_file_and_a_missing_dir_are_invalid(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    assert check_seeding(arm, expected_prompt=_STAMP).problems["missing_artifact"] == 2

    _write(tmp_path, arm, _report(0))
    verdict = check_seeding(arm, expected_prompt=_STAMP)

    assert verdict.problems["missing_artifact"] == 1
    assert "situational_run_001.json" in verdict.reasons[0]
