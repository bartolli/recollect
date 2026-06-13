"""SurfacingArmRunner labeling + score-query loop (mocked memory) + factory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli.arm import load_arm
from probe_cli.corpus import QueryEntry
from probe_cli.surfacing_runner import (
    SurfacingArmRunner,
    _build_query_surfacing,
    _label_relevant,
)
from recollect.exceptions import StorageError
from recollect.models import PersonaFact


def _fact(source_trace_id: str | None = "db-1") -> PersonaFact:
    return PersonaFact(
        subject="s", predicate="p", object="o", content="c",
        source_trace_id=source_trace_id,
    )


def _runner() -> SurfacingArmRunner:
    # Bypass __init__ -- _score_query reads no self attributes.
    return SurfacingArmRunner.__new__(SurfacingArmRunner)


class TestLabelRelevant:
    def test_mapped_in_relevant_set(self) -> None:
        assert _label_relevant("db-1", {"db-1": "eval-1"}, {"eval-1"}) is True

    def test_none_source_trace(self) -> None:
        assert _label_relevant(None, {"db-1": "eval-1"}, {"eval-1"}) is False

    def test_unmapped_source_trace(self) -> None:
        assert _label_relevant("db-x", {"db-1": "eval-1"}, {"eval-1"}) is False

    def test_mapped_but_not_relevant(self) -> None:
        assert _label_relevant("db-1", {"db-1": "eval-2"}, {"eval-1"}) is False


class TestBuildQuerySurfacing:
    def test_labels_and_scores(self) -> None:
        f = _fact("db-1")
        q = QueryEntry(
            id="q1", text="x", relevant_trace_ids=["eval-1"], phrasing_style="terse"
        )
        qs = _build_query_surfacing(q, [f], {f.id: 0.72}, {"db-1": "eval-1"})
        assert qs.query_id == "q1"
        assert qs.phrasing_style == "terse"
        assert qs.is_distractor is False
        assert len(qs.surfaced) == 1
        assert qs.surfaced[0].score == pytest.approx(0.72)
        assert qs.surfaced[0].relevant is True

    def test_distractor_query_marks_all_noise(self) -> None:
        f = _fact("db-9")
        q = QueryEntry(id="d1", text="x", relevant_trace_ids=[])
        qs = _build_query_surfacing(q, [f], {f.id: 0.3}, {"db-9": "eval-9"})
        assert qs.is_distractor is True
        assert qs.surfaced[0].relevant is False

    def test_missing_score_defaults_zero(self) -> None:
        f = _fact("db-1")
        qs = _build_query_surfacing(
            QueryEntry(id="q1", text="x", relevant_trace_ids=["eval-1"]),
            [f], {}, {"db-1": "eval-1"},
        )
        assert qs.surfaced[0].score == pytest.approx(0.0)


@pytest.mark.asyncio
class TestScoreQuery:
    async def test_uses_engine_and_labels(self) -> None:
        f = _fact("db-1")
        mem = MagicMock()
        mem._embeddings.generate_embedding = AsyncMock(return_value=[0.1, 0.2])
        mem._find_relevant_persona_facts = AsyncMock(return_value=([f], {f.id: 0.72}))
        q = QueryEntry(id="q1", text="hi", relevant_trace_ids=["eval-1"])
        qs = await _runner()._score_query(mem, q, {"db-1": "eval-1"}, user_id="u")
        assert qs.surfaced[0].score == pytest.approx(0.72)  # blended S from engine
        assert qs.surfaced[0].relevant is True
        mem._find_relevant_persona_facts.assert_awaited_once()
        mem._embeddings.generate_embedding.assert_awaited_once()

    async def test_returns_none_on_engine_error(self) -> None:
        # A mid-run embedding/engine failure must not abort the whole run
        # (which already paid the seed cost) -- skip the query, like _seed.
        mem = MagicMock()
        mem._embeddings.generate_embedding = AsyncMock(side_effect=StorageError("x"))
        mem._find_relevant_persona_facts = AsyncMock()
        q = QueryEntry(id="q1", text="hi", relevant_trace_ids=["eval-1"])
        result = await _runner()._score_query(mem, q, {}, user_id="u")
        assert result is None
        mem._find_relevant_persona_facts.assert_not_awaited()


def _arm_file(tmp_path: Path, *, surfacing: str) -> Path:
    p = tmp_path / "arm.toml"
    p.write_text(
        '[arm]\nname = "s"\n[corpus]\npath = ""\n' + surfacing, encoding="utf-8"
    )
    return p


class TestFromArmFactory:
    def test_rejects_disabled(self, tmp_path: Path) -> None:
        arm = load_arm(_arm_file(tmp_path, surfacing=""))
        with pytest.raises(ValueError, match=r"surfacing\.enabled"):
            SurfacingArmRunner.from_arm(arm, MagicMock(model_name="m"))

    def test_rejects_missing_paths(self, tmp_path: Path) -> None:
        arm = load_arm(
            _arm_file(tmp_path, surfacing="[surfacing]\nenabled = true\n")
        )
        with pytest.raises(ValueError, match="traces_corpus_path"):
            SurfacingArmRunner.from_arm(arm, MagicMock(model_name="m"))

    def test_accepts_enabled_with_paths(self, tmp_path: Path) -> None:
        body = (
            "[surfacing]\nenabled = true\n"
            'traces_corpus_path = "t.jsonl"\n'
            'query_corpus_path = "q.jsonl"\n'
        )
        arm = load_arm(_arm_file(tmp_path, surfacing=body))
        runner = SurfacingArmRunner.from_arm(arm, MagicMock(model_name="m"))
        assert runner.model == "m"
        assert runner._traces_path == "t.jsonl"
