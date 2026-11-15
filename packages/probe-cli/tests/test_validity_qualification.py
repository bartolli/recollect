"""Qualification validity: retrieval run files plus representations.json."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
from probe_cli.arm import Arm, CorpusConfig, OutputConfig, RetrievalConfig
from probe_cli.metrics import QueryResult, RetrievalRunReport
from probe_cli.provenance import PromptStamp
from probe_cli.report import write_representations_report, write_retrieval_run_report
from probe_cli.representations import RepresentationsReport, StatementRepresentation
from probe_cli.validity import EXTRACTION_RETRY, SeedingValidity, check_seeding
from recollect.llm.types import ExtractionResult

if TYPE_CHECKING:
    from pathlib import Path

_STAMP = PromptStamp(version="1.2.1", sha256="a" * 64, chars=11060)
_KEYS = [
    "missing_artifact",
    "prompt_mismatch",
    "swallowed_failures",
    "run_files",
    "ingest_failures",
    "uningested",
    "query_ids",
    "failed_queries",
    "statement_ids",
    "degraded",
    "traces_sha256",
    "user_id",
]


def _arm(tmp_path: Path) -> Arm:
    traces, queries = tmp_path / "traces.jsonl", tmp_path / "queries.jsonl"
    traces.write_text('{"id": "a", "text": "a"}\n{"id": "b", "text": "b"}\n', "utf-8")
    queries.write_text(
        '{"id": "q-a", "text": "x", "relevant_trace_ids": ["a"]}\n'
        '{"id": "q-b", "text": "y", "relevant_trace_ids": ["b"]}\n',
        "utf-8",
    )
    return Arm(
        name="qual-s1",
        runs=1,
        corpus=CorpusConfig(path=str(traces)),
        output=OutputConfig(dir=str(tmp_path / "out")),
        retrieval=RetrievalConfig(
            enabled=True, traces_corpus_path=str(traces), query_corpus_path=str(queries)
        ),
    )


def _run() -> RetrievalRunReport:
    return RetrievalRunReport(
        arm_name="qual-s1",
        run_index=0,
        model="m",
        prompt_version="1.2.1",
        top_k=5,
        ingested=2,
        queries=[QueryResult(query_id=q, success=True) for q in ("q-a", "q-b")],
        prompt=_STAMP,
        swallowed_failures={EXTRACTION_RETRY: 2},
    )


def _statement(sid: str) -> StatementRepresentation:
    return StatementRepresentation(
        id=sid,
        trace_content=sid,
        trace_status="active",
        extraction=ExtractionResult(concepts=["c"]),
        degraded=False,
        entity_relations=[],
        persona_facts=[],
        pinned_fact_lines=[],
    )


def _reps(tmp_path: Path) -> RepresentationsReport:
    digest = hashlib.sha256((tmp_path / "traces.jsonl").read_bytes()).hexdigest()
    return RepresentationsReport(
        arm_name="answers",
        template_db="keep",
        user_id="probe-qual-s1",
        traces_path="traces.jsonl",
        traces_sha256=digest,
        degraded_statements=[],
        statements=[_statement("a"), _statement("b")],
    )


def _write(out: Path, run: RetrievalRunReport, reps: RepresentationsReport) -> None:
    write_retrieval_run_report(run, out)
    write_representations_report(reps, out)


def _assert_only(verdict: SeedingValidity, key: str) -> None:
    assert verdict.state == "INVALID"
    assert {k for k, n in verdict.problems.items() if n} == {key}
    (reason,) = verdict.reasons
    assert reason.startswith(f"{key}=")


_INGEST = {"probe_cli.runner: ingest failed for %s": 1}
_QUERIES = [QueryResult(query_id=q, success=True) for q in ("q-a", "q-a", "q-x")]
_FAILED = [
    QueryResult(query_id="q-a", success=True),
    QueryResult(query_id="q-b", success=False),
]
# key -> (retrieval run update, representations update)
_UPDATES: dict[str, tuple[dict[str, object], dict[str, object]]] = {
    "prompt_mismatch": ({"prompt": PromptStamp()}, {}),
    "swallowed_failures": ({"swallowed_failures": _INGEST}, {}),
    "ingest_failures": ({"ingest_failures": 1}, {}),
    "uningested": ({"ingested": 1}, {}),
    "query_ids": ({"queries": _QUERIES}, {}),
    "failed_queries": ({"queries": _FAILED}, {}),
    "statement_ids": ({}, {"statements": [_statement("b"), _statement("a")]}),
    "degraded": ({}, {"degraded_statements": ["b"]}),
    "traces_sha256": ({}, {"traces_sha256": "0" * 64}),
    "user_id": ({}, {"user_id": "probe-qual-s2"}),
}


def test_a_complete_seeding_is_valid(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    _write(tmp_path / "out" / arm.name, _run(), _reps(tmp_path))

    verdict = check_seeding(arm, expected_prompt=_STAMP)

    assert (verdict.state, verdict.mode, verdict.arm_name) == (
        "VALID",
        "qualification",
        "qual-s1",
    )
    assert list(verdict.problems) == _KEYS
    assert set(verdict.problems.values()) == {0}
    assert verdict.reasons == []
    assert verdict.prompt == verdict.expected_prompt == _STAMP


@pytest.mark.parametrize("key", _UPDATES)
def test_each_report_problem_makes_the_seeding_invalid(
    tmp_path: Path, key: str
) -> None:
    arm = _arm(tmp_path)
    run_update, reps_update = _UPDATES[key]
    _write(
        tmp_path / "out" / arm.name,
        _run().model_copy(update=run_update),
        _reps(tmp_path).model_copy(update=reps_update),
    )

    _assert_only(check_seeding(arm, expected_prompt=_STAMP), key)


def test_an_absent_representations_file_is_a_missing_artifact(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    write_retrieval_run_report(_run(), tmp_path / "out" / arm.name)

    _assert_only(check_seeding(arm, expected_prompt=_STAMP), "missing_artifact")


def test_a_run_file_beyond_arm_runs_is_a_run_files_problem(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    out = tmp_path / "out" / arm.name
    _write(out, _run(), _reps(tmp_path))
    write_retrieval_run_report(_run().model_copy(update={"run_index": 1}), out)

    _assert_only(check_seeding(arm, expected_prompt=_STAMP), "run_files")


def test_a_missing_evidence_dir_is_invalid(tmp_path: Path) -> None:
    arm = _arm(tmp_path)

    verdict = check_seeding(arm, expected_prompt=_STAMP)

    assert verdict.state == "INVALID"
    assert verdict.problems["missing_artifact"] == 2
    assert "retrieval_run_000.json" in verdict.reasons[0]
