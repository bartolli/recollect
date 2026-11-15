"""Situational run reports: seeding degraded count, eval degraded, prompt, failures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli import situational_runner
from probe_cli.arm import Arm, CorpusConfig
from probe_cli.provenance import stamp_prompt
from probe_cli.situational_runner import SituationalArmRunner, extraction_degraded
from recollect.exceptions import MemorySDKError
from recollect.extraction import PatternExtractor
from recollect.models import MemoryTrace

if TYPE_CHECKING:
    from pathlib import Path

    from probe_cli.situational_runner import SituationalRunReport

_EVAL = '{{"id": "{0}", "text": "{0}", "expected_action": "none"}}\n'


def _memory(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mem = MagicMock()
    mem.connect = AsyncMock()
    mem.close = AsyncMock()
    mem.experience = AsyncMock(return_value=MemoryTrace(id="db-s1", content="seed"))
    mem.storage.traces.count_degraded = AsyncMock(return_value=2)
    mem.assess_situational = AsyncMock(return_value=None)
    mem.erase = AsyncMock()

    def _connected(**_: object) -> MagicMock:
        return mem

    monkeypatch.setattr(situational_runner, "CognitiveMemory", _connected)
    return mem


async def _run_all(
    tmp_path: Path, evals_text: str = "", seeds: int = 1
) -> list[SituationalRunReport]:
    paths = [tmp_path / n for n in ("s.jsonl", "g.jsonl", "e.jsonl")]
    seed_lines = "".join(f'{{"id": "s{i}", "text": "seed"}}\n' for i in range(seeds))
    for path, text in zip(paths, (seed_lines, "", evals_text), strict=True):
        path.write_text(text, encoding="utf-8")
    runner = SituationalArmRunner(
        arm_name="g",
        provider=MagicMock(model_name="m"),
        config=Arm(name="g", corpus=CorpusConfig(path="")).to_memory_config(),
        seed_traces_path=str(paths[0]),
        seed_groups_path=str(paths[1]),
        eval_corpus_path=str(paths[2]),
        runs=2,
    )
    return await runner.run_all()


@pytest.mark.parametrize(
    ("pattern", "degraded"),
    [
        ({}, True),
        ({"concepts": []}, True),
        ({"concepts": None}, True),
        ({"concepts": "a"}, True),
        ({"concepts": ["a"]}, False),
        ({"session_summary": True, "concepts": []}, False),
        ({"session_summary": False, "concepts": []}, True),
    ],
)
def test_extraction_degraded_mirrors_count_degraded(
    pattern: dict[str, Any], degraded: bool
) -> None:
    assert extraction_degraded(pattern) is degraded


async def test_every_eval_result_after_experience_carries_its_degraded_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mem = _memory(monkeypatch)
    seed = MemoryTrace(id="db-s0", content="seed", pattern={"concepts": ["s"]})
    bare = MemoryTrace(id="db-e1", content="e1", pattern={"concepts": []})
    full = MemoryTrace(id="db-e2", content="e2", pattern={"concepts": ["a"]})
    lost = MemorySDKError("ingest")
    mem.experience = AsyncMock(side_effect=[seed, bare, full, lost, bare, full, lost])
    mem.assess_situational = AsyncMock(side_effect=[None, MemorySDKError("x")] * 2)
    evals = "".join(_EVAL.format(e) for e in ("e1", "e2", "e3"))
    reports = await _run_all(tmp_path, evals)

    for report in reports:
        flags = [
            (r.entry_id, r.success, r.extraction_degraded) for r in report.eval_results
        ]
        assert flags == [("e1", True, True), ("e2", False, False), ("e3", False, False)]


async def test_every_run_report_carries_the_prompt_and_seeding_wide_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mem = _memory(monkeypatch)
    seed = MemoryTrace(id="db-s0", content="seed")
    mem.experience = AsyncMock(side_effect=[seed, MemorySDKError("seed"), seed, seed])
    mem.erase = AsyncMock(side_effect=[None, MemorySDKError("erase")])
    reports = await _run_all(tmp_path, _EVAL.format("e1"), seeds=2)

    expected = {
        "probe_cli.situational_runner: seed ingest failed for %s": 1,
        "probe_cli.situational_runner: cleanup erase failed for %s": 1,
    }
    assert [r.swallowed_failures for r in reports] == [expected, expected]
    config = Arm(name="g", corpus=CorpusConfig(path="")).to_memory_config()
    stamp = stamp_prompt(PatternExtractor(MagicMock(), config=config))
    assert [r.prompt for r in reports] == [stamp, stamp]


async def test_every_run_report_carries_the_seeding_degraded_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mem = _memory(monkeypatch)
    reports = await _run_all(tmp_path)
    assert [r.seed_traces_degraded for r in reports] == [2, 2]
    count = mem.storage.traces.count_degraded
    count.assert_awaited_once_with(user_id="probe-g")
    names = [name for name, _, _ in mem.mock_calls]
    assert names.index("storage.traces.count_degraded") > names.index("experience")
