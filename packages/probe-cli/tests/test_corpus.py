"""Corpus loader tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from probe_cli.corpus import load_corpus, load_query_corpus


def _write_jsonl(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    p = tmp_path / "corpus.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p


def test_loads_simple_corpus(tmp_path: Path) -> None:
    p = _write_jsonl(
        tmp_path,
        [
            {"id": "a1", "text": "Sarah is allergic to peanuts."},
            {"id": "a2", "text": "Bob works at Google."},
        ],
    )
    c = load_corpus(p)
    assert len(c) == 2
    assert c.entries[0].id == "a1"
    assert c.entries[1].text == "Bob works at Google."
    assert c.source_path == str(p)


def test_skips_blank_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text(
        '# header comment\n'
        '\n'
        '{"id": "a", "text": "first"}\n'
        '\n'
        '{"id": "b", "text": "second"}\n',
        encoding="utf-8",
    )
    c = load_corpus(p)
    assert len(c) == 2


def test_ground_truth_fields_optional(tmp_path: Path) -> None:
    p = _write_jsonl(
        tmp_path,
        [
            {"id": "x", "text": "foo"},
            {
                "id": "y",
                "text": "bar",
                "expected_category": "health",
                "expected_predicate": "is_allergic_to",
            },
        ],
    )
    c = load_corpus(p)
    assert c.entries[0].expected_category is None
    assert c.entries[1].expected_category == "health"
    assert c.entries[1].expected_predicate == "is_allergic_to"


def test_invalid_predicate_rejected(tmp_path: Path) -> None:
    p = _write_jsonl(
        tmp_path,
        [{"id": "x", "text": "t", "expected_predicate": "bogus_predicate"}],
    )
    with pytest.raises(ValueError, match="invalid corpus entry"):
        load_corpus(p)


def test_invalid_category_rejected(tmp_path: Path) -> None:
    p = _write_jsonl(
        tmp_path,
        [{"id": "x", "text": "t", "expected_category": "made_up"}],
    )
    with pytest.raises(ValueError, match="invalid corpus entry"):
        load_corpus(p)


def test_missing_required_field_rejected(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"id": "x"}])  # text missing
    with pytest.raises(ValueError, match="invalid corpus entry"):
        load_corpus(p)


def test_query_corpus_loads(tmp_path: Path) -> None:
    p = _write_jsonl(
        tmp_path,
        [
            {
                "id": "q1",
                "text": "what is sarah allergic to?",
                "relevant_trace_ids": ["h1"],
                "expected_category": "health",
                "phrasing_style": "literal",
            },
            {
                "id": "q2",
                "text": "irrelevant query",
                "relevant_trace_ids": [],
            },
        ],
    )
    qc = load_query_corpus(p)
    assert len(qc) == 2
    assert qc.entries[0].relevant_trace_ids == ["h1"]
    assert qc.entries[0].expected_category == "health"
    assert qc.entries[0].phrasing_style == "literal"
    assert qc.entries[1].relevant_trace_ids == []  # distractor


def test_query_corpus_invalid_category_rejected(tmp_path: Path) -> None:
    p = _write_jsonl(
        tmp_path,
        [
            {
                "id": "q1",
                "text": "x",
                "relevant_trace_ids": [],
                "expected_category": "made_up_cat",
            }
        ],
    )
    with pytest.raises(ValueError, match="invalid query entry"):
        load_query_corpus(p)


def test_query_corpus_missing_required_rejected(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"id": "q1"}])  # text missing
    with pytest.raises(ValueError, match="invalid query entry"):
        load_query_corpus(p)


def test_query_corpus_skips_blank_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "q.jsonl"
    p.write_text(
        '# header\n\n{"id": "q1", "text": "ask"}\n\n{"id": "q2", "text": "ask2"}\n',
        encoding="utf-8",
    )
    qc = load_query_corpus(p)
    assert len(qc) == 2
