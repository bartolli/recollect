"""Qualification fixture sanity: ids, order, cross-references, arm shape."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Literal

import pytest
from probe_cli.arm import load_arm
from probe_cli.corpus import CorpusEntry, load_corpus, load_query_corpus
from pydantic import BaseModel, ConfigDict, Field
from recollect.prompts import load_prompt_file

_FIX = Path(__file__).parent.parent / "fixtures" / "qualification"
_IDS = {
    *(f"{kind}-{n}" for kind in ("might", "will", "reported") for n in (1, 2, 3)),
    "corrected-1a",
    "corrected-1b",
    "corrected-2a",
    "corrected-2b",
    *(f"screen-{n}" for n in (1, 2, 3, 4)),
}
_PAIRS = (("corrected-1a", "corrected-1b"), ("corrected-2a", "corrected-2b"))
_RELATIVE = re.compile(r"\b(tomorrow|next week|next month|yesterday|today)\b", re.I)


class _CurrentState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current: str = Field(min_length=1)
    superseded: str = Field(min_length=1)


class _KeyRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["might", "will", "reported", "corrected", "screen"]
    source: str = Field(min_length=1)
    certainty: Literal["possible", "conditional", "committed", "opinion"]
    current_state: _CurrentState | None
    checks: list[str] = Field(min_length=1)


def _traces() -> list[CorpusEntry]:
    return load_corpus(_FIX / "traces.jsonl").entries


def _key_rows() -> list[_KeyRow]:
    lines = (_FIX / "reader_key.jsonl").read_text(encoding="utf-8").splitlines()
    return [
        _KeyRow.model_validate_json(line)
        for line in lines
        if line.strip() and not line.startswith("#")
    ]


def test_traces_are_the_seventeen_ids_with_each_correction_spaced() -> None:
    order = [t.id for t in _traces()]

    assert len(order) == len(set(order)) == 17
    assert set(order) == _IDS
    for a, b in _PAIRS:
        assert order.index(b) - order.index(a) >= 3, f"{a} -> {b} too close"


def test_one_query_per_trace_except_superseded_halves() -> None:
    trace_ids = {t.id for t in _traces()}
    queries = load_query_corpus(_FIX / "queries.jsonl").entries

    superseded = {a for a, _ in _PAIRS}
    ids = [q.id for q in queries]

    assert len(ids) == len(set(ids))
    assert set(ids) == {f"q-{tid}" for tid in trace_ids - superseded}
    for q in queries:
        assert q.relevant_trace_ids == [q.id.removeprefix("q-")]
        assert set(q.relevant_trace_ids) <= trace_ids


def test_reader_key_covers_every_trace_and_pairs_share_state() -> None:
    rows = _key_rows()
    by_id = {row.id: row for row in rows}
    texts = {t.id: t.text for t in _traces()}

    assert len(rows) == len(by_id)
    assert set(by_id) == set(texts)
    for row in rows:
        assert row.kind == row.id.split("-")[0]
        assert (row.current_state is not None) == (row.kind == "corrected")
    for a, b in _PAIRS:
        state = by_id[a].current_state
        assert state is not None
        assert state == by_id[b].current_state
        assert state.superseded in texts[a]
        assert state.current in texts[b]


def test_arm_is_a_one_run_haiku55_retrieval_arm_with_overrides_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECOLLECT_PROBE_DB_URL", "postgresql://probe/probe_eval")
    path = _FIX / "arm.toml"

    arm = load_arm(path)

    assert arm.name == "qualification-haiku55"
    assert arm.runs == 1
    assert arm.retrieval.enabled
    assert arm.retrieval.db_url == "postgresql://probe/probe_eval"
    assert arm.extraction.pydantic_ai_model == "anthropic:claude-haiku-5-5"
    assert arm.recollect_overrides == {}
    assert arm.retrieval.traces_corpus_path == arm.corpus.path
    assert arm.corpus.path.endswith("qualification/traces.jsonl")
    assert arm.retrieval.query_corpus_path.endswith("qualification/queries.jsonl")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    assert raw["retrieval"]["db_url"] == "${RECOLLECT_PROBE_DB_URL}"


def test_authored_statements_are_bounded_and_free_of_relative_dates() -> None:
    authored = [t for t in _traces() if not t.id.startswith("screen-")]

    assert len(authored) == 13
    for t in authored:
        assert 15 <= len(t.text.split()) <= 35, f"{t.id} length"
        assert not _RELATIVE.search(t.text), f"{t.id} has a relative date"


def test_p121_arm_equals_the_control_arm_except_name_and_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECOLLECT_PROBE_DB_URL", "postgresql://probe/probe_eval")
    p121 = load_arm(_FIX / "arm-p121.toml")
    control = load_arm(_FIX / "arm.toml")

    assert p121.name == "qualification-p121-haiku55"
    assert control.extraction.instructions == ""
    blank = p121.extraction.model_copy(update={"instructions": ""})
    renamed = p121.model_copy(update={"name": control.name, "extraction": blank})
    assert renamed == control


def test_candidate_prompt_loads_and_names_no_reported_source() -> None:
    loaded = load_prompt_file(str(_FIX / "extraction.1.3.0.md"))

    assert (loaded.version, loaded.applies_to) == ("1.3.0", "extraction-template")
    sources = {row.source for row in _key_rows()} - {"user"}
    assert sources
    for source in sources:
        assert source not in loaded.body, f"prompt names {source}"
