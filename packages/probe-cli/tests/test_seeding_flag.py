"""--seeding sN: the arm name, evidence dir and user id take the suffix; one attempt."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest
from probe_cli import __main__ as cli
from probe_cli.metrics import RetrievalRunReport
from probe_cli.runner import RetrievalArmRunner

if TYPE_CHECKING:
    from pathlib import Path

    from probe_cli.corpus import Corpus, QueryCorpus

_TASK = """
[task]
enabled = true
seed_traces_path = "seeds.jsonl"
questions_path = "questions.jsonl"
answer_model = "test"
"""


def _arm(tmp_path: Path, *, task: bool = False) -> Path:
    (tmp_path / "t.jsonl").write_text('{"id": "a", "text": "a"}\n', encoding="utf-8")
    (tmp_path / "q.jsonl").write_text("", encoding="utf-8")
    retrieval = (
        f'[retrieval]\nenabled = true\ntraces_corpus_path = "{tmp_path / "t.jsonl"}"\n'
        f'query_corpus_path = "{tmp_path / "q.jsonl"}"\n'
    )
    path = tmp_path / "arm.toml"
    path.write_text(
        '[arm]\nname = "q"\nruns = 1\n[extraction]\npydantic_ai_model = "test"\n'
        f'[corpus]\npath = "unused"\n[output]\ndir = "{tmp_path / "out"}"\n'
        + (_TASK if task else retrieval),
        encoding="utf-8",
    )
    return path


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm_path: Path, *flags: str
) -> int:
    argv = ["probe", "run", "--arm", str(arm_path)]
    argv += ["--env-file", str(tmp_path / "absent.env"), *flags]
    monkeypatch.setattr(sys, "argv", argv)
    return cli.main()


def _refuse_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("built a provider before refusing the run")

    monkeypatch.setattr(cli, "PydanticAIProvider", refuse)


def test_the_seeding_reaches_the_evidence_dir_and_the_user_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    users: list[str] = []

    async def run_all(
        self: RetrievalArmRunner, _traces: Corpus, _queries: QueryCorpus
    ) -> list[RetrievalRunReport]:
        users.append(self._user_id)
        report = RetrievalRunReport(
            arm_name=self._arm.name, run_index=0, model="m", prompt_version="v", top_k=5
        )
        return [report]

    monkeypatch.setattr(RetrievalArmRunner, "run_all", run_all)

    assert _run(monkeypatch, tmp_path, _arm(tmp_path), "--seeding", "s3") == 0

    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["q-s3"]
    run = json.loads((out / "q-s3" / "retrieval_run_000.json").read_text("utf-8"))
    assert run["arm_name"] == "q-s3"
    assert users == ["probe-q-s3"]


@pytest.mark.parametrize("value", ["s0", "s01", "s100", "x1", "S1", "1"])
def test_a_bad_seeding_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _run(monkeypatch, tmp_path, _arm(tmp_path), "--seeding", value)

    assert exit_info.value.code == 2


def test_an_existing_seeding_dir_refuses_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _refuse_provider(monkeypatch)
    (tmp_path / "out" / "q-s1").mkdir(parents=True)

    assert _run(monkeypatch, tmp_path, _arm(tmp_path), "--seeding", "s1") == 2


def test_a_seeding_with_age_points_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _refuse_provider(monkeypatch)
    arm = _arm(tmp_path, task=True)

    assert _run(monkeypatch, tmp_path, arm, "--seeding", "s1", "--age-days", "7") == 2
