"""Unit tests for situational corpus schemas + loaders."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from probe_cli.corpus import (
    EvalEntry,
    SeedGroup,
    SeedTrace,
    load_eval_corpus,
    load_seed_groups,
    load_seed_traces,
)
from pydantic import ValidationError


def _write_jsonl(tmp_path: Path, name: str, lines: list[str]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


class TestSeedTrace:
    def test_minimal(self) -> None:
        t = SeedTrace(id="c1.1", text="hello")
        assert t.expected_category is None
        assert t.seed_group_id is None

    def test_grouped(self) -> None:
        t = SeedTrace(id="c1.1", text="hello", seed_group_id="G1")
        assert t.seed_group_id == "G1"


class TestSeedGroup:
    def test_label_assembly_matches_core_formula(self) -> None:
        g = SeedGroup(
            group_id="G1",
            person_ref="household",
            situation="overflow",
            implications=["a", "b", "c"],
            significance=0.6,
            member_trace_ids=["c1.1"],
        )
        assert g.label == "household | overflow | a, b, c"

    def test_defaults(self) -> None:
        g = SeedGroup(
            group_id="G1",
            person_ref="x",
            situation="y",
            implications=["z"],
            significance=0.5,
            member_trace_ids=["c1.1"],
        )
        assert g.strength == 1.0
        assert g.status == "active"

    def test_significance_range(self) -> None:
        with pytest.raises(ValidationError):
            SeedGroup(
                group_id="G1",
                person_ref="x",
                situation="y",
                implications=["z"],
                significance=1.5,
                member_trace_ids=["c1.1"],
            )

    def test_implications_required(self) -> None:
        with pytest.raises(ValidationError):
            SeedGroup(
                group_id="G1",
                person_ref="x",
                situation="y",
                implications=[],
                significance=0.5,
                member_trace_ids=["c1.1"],
            )

    def test_member_ids_required(self) -> None:
        with pytest.raises(ValidationError):
            SeedGroup(
                group_id="G1",
                person_ref="x",
                situation="y",
                implications=["z"],
                significance=0.5,
                member_trace_ids=[],
            )


class TestEvalEntry:
    def test_extend_minimal(self) -> None:
        e = EvalEntry(
            id="e1", text="x", expected_action="extend", expected_group_id="G1"
        )
        assert e.expected_action == "extend"
        assert e.expected_significance is None

    def test_action_enum_enforced(self) -> None:
        with pytest.raises(ValidationError):
            EvalEntry(id="e1", text="x", expected_action="bogus")  # type: ignore[arg-type]

    def test_significance_range(self) -> None:
        with pytest.raises(ValidationError):
            EvalEntry(
                id="e1",
                text="x",
                expected_action="revise",
                expected_group_id="G1",
                expected_significance=1.5,
            )


class TestLoaders:
    def test_seed_traces_round_trip(self, tmp_path: Path) -> None:
        body = textwrap.dedent(
            """
            # comment
            {"id": "c1.1", "text": "first", "seed_group_id": "G1"}

            {"id": "c1.u1", "text": "lone"}
            """
        ).strip()
        p = _write_jsonl(tmp_path, "seed.jsonl", [body])
        traces = load_seed_traces(p)
        assert len(traces) == 2
        assert traces[0].seed_group_id == "G1"
        assert traces[1].seed_group_id is None

    def test_seed_groups_round_trip(self, tmp_path: Path) -> None:
        line = json.dumps(
            {
                "group_id": "G1",
                "person_ref": "household",
                "situation": "overflow",
                "implications": ["a", "b"],
                "significance": 0.6,
                "member_trace_ids": ["c1.1", "c1.2"],
            }
        )
        p = _write_jsonl(tmp_path, "groups.jsonl", [line])
        groups = load_seed_groups(p)
        assert groups[0].label == "household | overflow | a, b"

    def test_eval_corpus_round_trip(self, tmp_path: Path) -> None:
        line = json.dumps(
            {
                "id": "c1.e1",
                "text": "swap",
                "expected_action": "extend",
                "expected_group_id": "G1",
                "expected_implication": "imp",
                "expected_significance": 0.7,
                "chain": "C1",
            }
        )
        p = _write_jsonl(tmp_path, "eval.jsonl", [line])
        entries = load_eval_corpus(p)
        assert entries[0].expected_action == "extend"
        assert entries[0].chain == "C1"

    def test_invalid_action_rejected(self, tmp_path: Path) -> None:
        p = _write_jsonl(
            tmp_path,
            "eval.jsonl",
            ['{"id": "e1", "text": "x", "expected_action": "junk"}'],
        )
        with pytest.raises(ValueError, match="invalid eval entry"):
            load_eval_corpus(p)


