"""Surfacing validity: surfacing_run.json, and failures the lookup swallows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from probe_cli.arm import Arm, CorpusConfig, OutputConfig, SurfacingConfig
from probe_cli.provenance import PromptStamp, capture_swallowed
from probe_cli.report import write_surfacing_run_report
from probe_cli.surfacing_runner import (
    PoolFact,
    QuerySurfacing,
    SurfacedFact,
    SurfacingRunReport,
)
from probe_cli.validity import EXTRACTION_RETRY, SeedingValidity, check_seeding

if TYPE_CHECKING:
    from pathlib import Path

_STAMP = PromptStamp(version="1.2.1", sha256="b" * 64, chars=11060)


def _arm(tmp_path: Path) -> Arm:
    traces, queries = tmp_path / "traces.jsonl", tmp_path / "queries.jsonl"
    traces.write_text('{"id": "a", "text": "a"}\n{"id": "b", "text": "b"}\n', "utf-8")
    queries.write_text(
        '{"id": "q1", "text": "x", "relevant_trace_ids": ["a"]}\n'
        '{"id": "q2", "text": "y"}\n',
        "utf-8",
    )
    surfacing = SurfacingConfig(
        enabled=True, traces_corpus_path=str(traces), query_corpus_path=str(queries)
    )
    return Arm(
        name="surf-s1",
        corpus=CorpusConfig(path=str(traces)),
        output=OutputConfig(dir=str(tmp_path / "out")),
        surfacing=surfacing,
    )


def _query(qid: str, surfaced: int = 5) -> QuerySurfacing:
    facts = [SurfacedFact(fact_id=f"f{i}") for i in range(surfaced)]
    return QuerySurfacing(query_id=qid, surfaced=facts)


def _report(**update: object) -> SurfacingRunReport:
    report = SurfacingRunReport(
        arm_name="surf-s1",
        model="m",
        seeded_traces=2,
        promoted_facts=3,
        queries=[_query("q1"), _query("q2")],
        prompt=_STAMP,
        swallowed_failures={EXTRACTION_RETRY: 1},
        surfacing_pool=[PoolFact(fact_id=f"f{i}", status="promoted") for i in range(7)],
        max_facts_per_query=5,
    )
    return report.model_copy(update=update)


def _check(tmp_path: Path, report: SurfacingRunReport) -> SeedingValidity:
    arm = _arm(tmp_path)
    write_surfacing_run_report(report, tmp_path / "out" / arm.name)
    return check_seeding(arm, expected_prompt=_STAMP)


def _captured(*records: tuple[str, ...]) -> dict[str, int]:
    with capture_swallowed() as swallowed:
        for name, msg, *args in records:
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                logging.getLogger(name).exception(msg, *args)
    return dict(swallowed)


_UPDATES: dict[str, dict[str, object]] = {
    "prompt_mismatch": {"prompt": PromptStamp()},
    "swallowed_failures": {"swallowed_failures": {"probe_cli.x: lost %s": 2}},
    "failed_seeds": {"failed_seed_ids": ["b"]},
    "unseeded": {"seeded_traces": 1},
    "degraded": {"degraded_traces": 1},
    "failed_queries": {"queries": [_query("q1")], "failed_query_ids": ["q2"]},
    "query_ids": {"queries": [_query("q1"), _query("q1"), _query("qx")]},
    "fact_count": {"queries": [_query("q1", surfaced=4), _query("q2")]},
}


def test_a_complete_seeding_is_valid(tmp_path: Path) -> None:
    verdict = _check(tmp_path, _report())

    assert (verdict.state, verdict.mode, verdict.reasons) == ("VALID", "surfacing", [])
    assert list(verdict.problems)[:3] == [
        "missing_artifact",
        "prompt_mismatch",
        "swallowed_failures",
    ]
    assert list(verdict.problems)[3:] == [
        "failed_seeds",
        "unseeded",
        "degraded",
        "failed_queries",
        "query_ids",
        "fact_count",
    ]


@pytest.mark.parametrize("key", _UPDATES)
def test_each_report_problem_makes_the_seeding_invalid(
    tmp_path: Path, key: str
) -> None:
    verdict = _check(tmp_path, _report(**_UPDATES[key]))

    assert verdict.state == "INVALID"
    assert {k for k, n in verdict.problems.items() if n} == {key}
    assert verdict.reasons[0].startswith(f"{key}=")


@pytest.mark.parametrize("content", [None, "{not json"])
def test_an_absent_or_unparsable_run_file_is_invalid(
    tmp_path: Path, content: str | None
) -> None:
    arm = _arm(tmp_path)
    if content is not None:
        (tmp_path / "out" / arm.name).mkdir(parents=True)
        (tmp_path / "out" / arm.name / "surfacing_run.json").write_text(content)

    verdict = check_seeding(arm, expected_prompt=_STAMP)

    assert verdict.state == "INVALID"
    assert verdict.problems["missing_artifact"] == 1


def test_a_pool_under_the_limit_is_the_expected_count(tmp_path: Path) -> None:
    pool = [PoolFact(fact_id="f0"), PoolFact(fact_id="f1")]
    queries = [_query("q1", surfaced=2), _query("q2", surfaced=2)]

    assert _check(tmp_path, _report(surfacing_pool=pool, queries=queries)).state == (
        "VALID"
    )


def test_an_unstamped_limit_counts_every_query(tmp_path: Path) -> None:
    verdict = _check(tmp_path, _report(max_facts_per_query=0))

    assert verdict.problems["fact_count"] == 2


def test_a_failed_concept_lookup_with_five_facts_reads_invalid(tmp_path: Path) -> None:
    swallowed = _captured(("recollect.core", "Fact concept attention lookup failed"))

    verdict = _check(tmp_path, _report(swallowed_failures=swallowed))

    assert verdict.state == "INVALID"
    assert (verdict.problems["swallowed_failures"], verdict.problems["fact_count"]) == (
        1,
        0,
    )


def test_a_failed_persona_lookup_reads_invalid_through_both(tmp_path: Path) -> None:
    swallowed = _captured(("recollect.core", "Persona fact lookup failed"))
    queries = [_query("q1", surfaced=0), _query("q2")]

    verdict = _check(tmp_path, _report(swallowed_failures=swallowed, queries=queries))

    assert (verdict.problems["swallowed_failures"], verdict.problems["fact_count"]) == (
        1,
        1,
    )


def test_the_retry_and_the_temperature_fallback_leave_a_seeding_valid(
    tmp_path: Path,
) -> None:
    swallowed = _captured(
        ("recollect.core", "LLM extraction failed, retrying"),
        (
            "recollect.llm.pydantic_ai",
            "Model %s rejects temperature; omitting it for this provider",
            "m",
        ),
    )

    assert swallowed == {EXTRACTION_RETRY: 1}
    assert _check(tmp_path, _report(swallowed_failures=swallowed)).state == "VALID"


def test_the_final_extraction_failure_makes_a_seeding_invalid(tmp_path: Path) -> None:
    final = "LLM extraction failed on retry, storing with empty extraction"
    swallowed = _captured(("recollect.core", final))

    verdict = _check(tmp_path, _report(swallowed_failures=swallowed))

    assert verdict.problems["swallowed_failures"] == 1
