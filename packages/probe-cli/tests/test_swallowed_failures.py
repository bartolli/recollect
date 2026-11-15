"""Swallowed failures: ERROR records under recollect and probe_cli, by template."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from probe_cli import runner as runner_module
from probe_cli.arm import Arm, CorpusConfig, RetrievalConfig
from probe_cli.corpus import Corpus, CorpusEntry, QueryCorpus, QueryEntry
from probe_cli.provenance import capture_swallowed
from probe_cli.runner import RetrievalArmRunner
from recollect.exceptions import MemorySDKError
from recollect.models import MemoryTrace

if TYPE_CHECKING:
    from probe_cli.metrics import RetrievalRunReport

_CORE = logging.getLogger("recollect.core")
_PROBE = logging.getLogger("probe_cli.surfacing_runner")


def _log_failure(logger: logging.Logger, msg: str, *args: object) -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception(msg, *args)


def _handlers() -> list[list[logging.Handler]]:
    return [list(logging.getLogger(n).handlers) for n in ("recollect", "probe_cli")]


def test_error_records_inside_the_span_are_counted_by_template() -> None:
    before = _handlers()
    _log_failure(_CORE, "Persona fact lookup failed")
    with capture_swallowed() as swallowed:
        _log_failure(_CORE, "Persona fact lookup failed")
        _log_failure(_PROBE, "score failed for query %s", "q1")
        _log_failure(_PROBE, "score failed for query %s", "q2")
        _CORE.warning("a warning is not a failure")
    _log_failure(_CORE, "Fact concept attention lookup failed")

    assert dict(swallowed) == {
        "recollect.core: Persona fact lookup failed": 1,
        "probe_cli.surfacing_runner: score failed for query %s": 2,
    }
    assert _handlers() == before


def test_the_handler_is_removed_when_the_span_raises() -> None:
    before = _handlers()
    with pytest.raises(RuntimeError), capture_swallowed():
        raise RuntimeError("span failed")

    assert _handlers() == before


def test_provider_records_are_not_counted() -> None:
    provider = logging.getLogger("recollect.llm.pydantic_ai")
    with capture_swallowed() as swallowed:
        _log_failure(provider, "pydantic-ai API error for model %s", "m")
        _log_failure(logging.getLogger("recollect.llm"), "provider failed")
        _log_failure(logging.getLogger("recollect.llmx"), "not the provider")

    assert dict(swallowed) == {"recollect.llmx: not the provider": 1}


async def test_a_retrieval_run_report_carries_its_run_span_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mem = MagicMock()
    mem.connect = AsyncMock()
    mem.close = AsyncMock()
    seeded = MemoryTrace(id="db-a", content="a")
    mem.experience = AsyncMock(side_effect=[seeded, MemorySDKError("boom")])
    mem.think_about = AsyncMock(return_value=[])

    def _connected(**_: object) -> MagicMock:
        return mem

    monkeypatch.setattr(runner_module, "CognitiveMemory", _connected)
    retrieval = RetrievalConfig(
        enabled=True, traces_corpus_path="t", query_corpus_path="q"
    )
    arm = Arm(name="r", corpus=CorpusConfig(path=""), retrieval=retrieval)
    _log_failure(_CORE, "before the run")
    report: RetrievalRunReport = await RetrievalArmRunner(arm, MagicMock()).run_once(
        Corpus(entries=[CorpusEntry(id="a", text="a"), CorpusEntry(id="b", text="b")]),
        QueryCorpus(entries=[QueryEntry(id="q1", text="x", relevant_trace_ids=["a"])]),
        run_index=0,
    )

    assert report.swallowed_failures == {"probe_cli.runner: ingest failed for %s": 1}
    assert report.ingest_failures == 1
