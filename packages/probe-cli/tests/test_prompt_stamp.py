"""Prompt stamp: the assembled system message on run reports and from the CLI."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli import __main__ as cli
from probe_cli import runner as runner_module
from probe_cli.arm import Arm, CorpusConfig, ExtractionConfig, RetrievalConfig, load_arm
from probe_cli.corpus import Corpus, CorpusEntry, QueryCorpus
from probe_cli.provenance import PromptStamp, stamp_prompt
from probe_cli.runner import RetrievalArmRunner
from recollect.extraction import PatternExtractor
from recollect.models import MemoryTrace

_FIX = Path(__file__).parent.parent / "fixtures"


def _stamp(arm: Arm) -> PromptStamp:
    return stamp_prompt(PatternExtractor(MagicMock(), config=arm.to_memory_config()))


def test_the_stamp_is_the_assembled_system_message() -> None:
    extraction = ExtractionConfig(instructions="Suffix.")
    arm = Arm(name="a", corpus=CorpusConfig(path=""), extraction=extraction)
    extractor = PatternExtractor(MagicMock(), config=arm.to_memory_config())
    prompt = extractor._build_prompt()

    assert prompt.endswith("\n\nSuffix.")
    assert stamp_prompt(extractor) == PromptStamp(
        version=extractor.template_version,
        sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        chars=len(prompt),
    )


async def test_a_retrieval_run_report_carries_the_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mem = MagicMock()
    mem.connect = AsyncMock()
    mem.close = AsyncMock()
    mem.experience = AsyncMock(return_value=MemoryTrace(id="db-a", content="a"))

    def _connected(**_: object) -> MagicMock:
        return mem

    monkeypatch.setattr(runner_module, "CognitiveMemory", _connected)
    retrieval = RetrievalConfig(
        enabled=True, traces_corpus_path="t", query_corpus_path="q"
    )
    extraction = ExtractionConfig(instructions="Suffix.")
    arm = Arm(
        name="r",
        corpus=CorpusConfig(path=""),
        extraction=extraction,
        retrieval=retrieval,
    )
    report = await RetrievalArmRunner(arm, MagicMock()).run_once(
        Corpus(entries=[CorpusEntry(id="a", text="a")]), QueryCorpus(), run_index=0
    )

    assert report.prompt == _stamp(arm)
    assert report.prompt.chars > len("Suffix.")


def test_prompt_stamp_prints_one_line_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("RECOLLECT_PROBE_DB_URL", "postgresql://probe/probe_eval")
    path = _FIX / "surfacing-h55" / "p130.toml"
    absent = str(tmp_path / "absent.env")
    argv = ["probe", "prompt-stamp", "--arm", str(path), "--env-file", absent]
    monkeypatch.setattr(sys, "argv", argv)

    assert cli.main() == 0
    expected = _stamp(load_arm(path))
    assert expected.version == "1.3.0"
    out = capsys.readouterr().out
    assert out == f"{expected.version} {expected.sha256} {expected.chars}\n"


def test_prompt_stamp_on_an_unset_db_variable_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RECOLLECT_PROBE_DB_URL", raising=False)
    path = _FIX / "surfacing-h55" / "p121.toml"
    absent = str(tmp_path / "absent.env")
    argv = ["probe", "prompt-stamp", "--arm", str(path), "--env-file", absent]
    monkeypatch.setattr(sys, "argv", argv)

    assert cli.main() == 2
