"""SituationalArmRunner eval-loop tests (Mode-A, mocked memory)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli.arm import load_arm
from probe_cli.corpus import EvalEntry
from probe_cli.situational_runner import (
    SituationalArmRunner,
    _resolve_actual_group,
)
from recollect.exceptions import StorageError
from recollect.llm.types import SituationalAssessment, TokenAssessment
from recollect.models import MemoryTrace


def _runner() -> SituationalArmRunner:
    # Bypass __init__ — eval-loop methods only need attributes the tests set.
    r = SituationalArmRunner.__new__(SituationalArmRunner)
    r._arm_name = "t"
    return r


def _entry(action: str, gid: str | None = None) -> EvalEntry:
    return EvalEntry(
        id="e1",
        text="x",
        expected_action=action,  # type: ignore[arg-type]
        expected_group_id=gid,
    )


def _memory_with_assessment(outcome: SituationalAssessment | None) -> MagicMock:
    mem = MagicMock()
    mem.experience = AsyncMock(return_value=MemoryTrace(id="uuid-eval", content="x"))
    mem.assess_situational = AsyncMock(return_value=outcome)
    mem.erase = AsyncMock()
    return mem


class TestResolveActualGroup:
    def test_extend_resolves_via_inverse_map(self) -> None:
        gid = _resolve_actual_group(
            "extend",
            1,
            ["tok-1", "tok-2"],
            {"tok-1": "G1", "tok-2": "G2"},
        )
        assert gid == "G1"

    def test_revise_resolves_second_group(self) -> None:
        gid = _resolve_actual_group(
            "revise",
            2,
            ["tok-1", "tok-2"],
            {"tok-1": "G1", "tok-2": "G2"},
        )
        assert gid == "G2"

    def test_create_returns_none(self) -> None:
        assert _resolve_actual_group("create", 1, ["tok-1"], {"tok-1": "G1"}) is None

    def test_none_action_returns_none(self) -> None:
        assert _resolve_actual_group("none", 1, ["tok-1"], {"tok-1": "G1"}) is None

    def test_out_of_bounds_returns_none(self) -> None:
        assert _resolve_actual_group("extend", 5, ["tok-1"], {"tok-1": "G1"}) is None

    def test_unknown_token_returns_none(self) -> None:
        assert _resolve_actual_group("extend", 1, ["tok-x"], {"tok-1": "G1"}) is None


@pytest.mark.asyncio
class TestAssessOne:
    async def test_records_extend_with_resolved_group(self) -> None:
        runner = _runner()
        outcome = SituationalAssessment(
            assessment=TokenAssessment(
                action="extend",
                group_number=1,
                implication="ok",
                significance=0.7,
            ),
            related_trace_ids=["uuid-seed-1"],
            candidate_token_ids=["tok-G1"],
        )
        mem = _memory_with_assessment(outcome)
        result = await runner._assess_one(
            mem,
            _entry("extend", "G1"),
            {"tok-G1": "G1"},
            session_id="s",
            user_id="u",
        )
        assert result.actual_action == "extend"
        assert result.actual_group_id == "G1"
        assert result.actual_implication == "ok"
        assert result.actual_significance == pytest.approx(0.7)
        assert result.success
        mem.erase.assert_awaited_once_with("uuid-eval")

    async def test_records_none_when_no_related(self) -> None:
        runner = _runner()
        mem = _memory_with_assessment(None)
        result = await runner._assess_one(
            mem,
            _entry("none"),
            {},
            session_id="s",
            user_id="u",
        )
        assert result.actual_action == "none"
        assert result.actual_group_id is None
        assert result.success
        mem.erase.assert_awaited_once()

    async def test_records_ingest_failure(self) -> None:
        runner = _runner()
        mem = MagicMock()
        mem.experience = AsyncMock(side_effect=StorageError("db down"))
        mem.assess_situational = AsyncMock()
        mem.erase = AsyncMock()
        result = await runner._assess_one(
            mem,
            _entry("extend", "G1"),
            {},
            session_id="s",
            user_id="u",
        )
        assert not result.success
        assert "ingest" in result.error
        mem.assess_situational.assert_not_awaited()
        mem.erase.assert_not_awaited()

    async def test_records_assessment_failure_and_cleans_up(self) -> None:
        runner = _runner()
        mem = MagicMock()
        mem.experience = AsyncMock(
            return_value=MemoryTrace(id="uuid-eval", content="x")
        )
        mem.assess_situational = AsyncMock(side_effect=StorageError("llm boom"))
        mem.erase = AsyncMock()
        result = await runner._assess_one(
            mem,
            _entry("extend", "G1"),
            {},
            session_id="s",
            user_id="u",
        )
        assert not result.success
        assert "assess" in result.error
        mem.erase.assert_awaited_once_with("uuid-eval")

    async def test_unresolved_group_records_none_target_marker(self) -> None:
        # LLM returned group_number that maps via candidate_token_ids to a token
        # the harness didn't seed (foreign group). actual_group_id stays None;
        # action accuracy still passes if expected==actual.
        runner = _runner()
        outcome = SituationalAssessment(
            assessment=TokenAssessment(
                action="extend",
                group_number=1,
                implication="x",
            ),
            related_trace_ids=[],
            candidate_token_ids=["tok-stray"],
        )
        mem = _memory_with_assessment(outcome)
        result = await runner._assess_one(
            mem,
            _entry("extend", "G1"),
            {"tok-G1": "G1"},
            session_id="s",
            user_id="u",
        )
        assert result.actual_action == "extend"
        assert result.actual_group_id is None


def _arm_file(tmp_path: Path, *, situational: str) -> Path:
    p = tmp_path / "arm.toml"
    p.write_text(
        '[arm]\nname = "s"\n[corpus]\npath = ""\n' + situational,
        encoding="utf-8",
    )
    return p


class TestFromArmFactory:
    def test_rejects_disabled(self, tmp_path: Path) -> None:
        arm = load_arm(_arm_file(tmp_path, situational=""))
        with pytest.raises(ValueError, match=r"situational\.enabled"):
            SituationalArmRunner.from_arm(arm, MagicMock(model_name="m"))

    def test_rejects_missing_paths(self, tmp_path: Path) -> None:
        arm = load_arm(
            _arm_file(tmp_path, situational="[situational]\nenabled = true\n")
        )
        with pytest.raises(ValueError, match="seed_traces_path"):
            SituationalArmRunner.from_arm(arm, MagicMock(model_name="m"))

    def test_disables_recall_tokens_in_config(self, tmp_path: Path) -> None:
        body = (
            "[situational]\nenabled = true\n"
            'seed_traces_path = "s.jsonl"\n'
            'seed_groups_path = "g.jsonl"\n'
            'eval_corpus_path = "e.jsonl"\n'
        )
        arm = load_arm(_arm_file(tmp_path, situational=body))
        runner = SituationalArmRunner.from_arm(arm, MagicMock(model_name="m"))
        assert runner._config.get("recall_tokens.enabled") is False
