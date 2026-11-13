"""probe entity-reach: preconditions exit 2; a read writes one artifact."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest
from probe_cli import __main__ as cli
from probe_cli import entity_reach
from probe_cli.entity_reach import StoreMention, StoreRead, StoreTrace

if TYPE_CHECKING:
    from pathlib import Path

_SEEDS = [
    ("head", "Dinner Friday at the Thai place on Main Street."),
    ("mid", "The Thai restaurant on Main Street needs a reservation."),
    ("tail", "The reservation deposit is due March 30."),
]
_QUESTION = {
    "id": "t3q01",
    "question": "What is due before dinner at the Thai place on Main Street?",
    "answers": ["March 30"],
    "answer_type": "date",
    "requires_trace_ids": ["tail"],
    "tier_label": "t3",
}
_UNBRIDGED = {**_QUESTION, "id": "aq01", "tier_label": "any"}
_MENTIONS = [
    ("s-head", "Thai place on  Main Street", "thai place on main street"),
    ("s-mid", "Thai restaurant on Main Street", "thai restaurant on main street"),
    ("s-mid", "reservation", "reservation"),
    ("s-tail", "Reservation", "reservation"),
]


def _fixture(tmp_path: Path, *, task: bool) -> Path:
    (tmp_path / "seeds.jsonl").write_text(
        "\n".join(
            json.dumps({"id": i, "text": t, "density_tier": 0}) for i, t in _SEEDS
        ),
        encoding="utf-8",
    )
    (tmp_path / "questions.jsonl").write_text(
        "# header\n" + json.dumps(_QUESTION) + "\n" + json.dumps(_UNBRIDGED) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "entity_pair_labels.jsonl").write_text(
        '{"a": "thai restaurant on main street", "b": "thai place on main street",'
        ' "label": "same"}\n',
        encoding="utf-8",
    )
    task_block = (
        f'[task]\nenabled = true\nseed_traces_path = "{tmp_path / "seeds.jsonl"}"\n'
        f'questions_path = "{tmp_path / "questions.jsonl"}"\n'
    )
    path = tmp_path / "reach.toml"
    path.write_text(
        '[arm]\nname = "reach"\n[corpus]\npath = ""\n'
        f'[output]\ndir = "{tmp_path / "out"}"\n' + (task_block if task else ""),
        encoding="utf-8",
    )
    return path


def _store(*, users: list[str | None], texts: list[str] | None = None) -> StoreRead:
    texts = texts if texts is not None else [f"  {t} " for _, t in _SEEDS]
    ids = [f"s-{i}" for i, _ in _SEEDS]
    return StoreRead(
        user_ids=users,
        traces=[StoreTrace(id=i, content=t) for i, t in zip(ids, texts, strict=True)],
        mentions=[
            StoreMention(trace_id=t, entity_name=n, name_key=k) for t, n, k in _MENTIONS
        ],
    )


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: Path, store: StoreRead | None
) -> int:
    async def read_store(_db_url: str) -> StoreRead:
        assert store is not None, "read the store before checking the arm"
        return store

    async def embed_names(keys: list[str]) -> list[list[float]]:
        # The two Thai names sit at cosine 0.8; everything else is orthogonal.
        basis = {"thai place on main street": [1.0, 0.0, 0.0, 0.0]}
        basis["thai restaurant on main street"] = [0.8, 0.6, 0.0, 0.0]
        basis["reservation"] = [0.0, 0.0, 1.0, 0.0]
        return [basis[key] for key in keys]

    monkeypatch.setattr(entity_reach, "read_store", read_store)
    monkeypatch.setattr(entity_reach, "embed_names", embed_names)
    argv = ["probe", "entity-reach", "--arm", str(arm)]
    argv += ["--env-file", str(tmp_path / "absent.env"), "--thresholds", "0.9", "0.75"]
    monkeypatch.setattr(sys, "argv", argv)
    return cli.main()


def test_non_task_arm_exits_2_before_reading_the_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("built a provider for a read-only instrument")

    monkeypatch.setattr(cli, "PydanticAIProvider", refuse)
    arm = _fixture(tmp_path, task=False)

    assert _run(monkeypatch, tmp_path, arm, None) == 2
    assert "requires a task arm" in caplog.text


def test_store_of_two_users_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    arm = _fixture(tmp_path, task=True)

    assert _run(monkeypatch, tmp_path, arm, _store(users=["u1", "u2"])) == 2
    assert "2 users" in caplog.text
    assert not (tmp_path / "out").exists()


def test_trace_outside_the_fixture_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    arm = _fixture(tmp_path, task=True)
    texts = [_SEEDS[0][1], _SEEDS[1][1], "An ad-hoc memory."]

    assert _run(monkeypatch, tmp_path, arm, _store(users=["u1"], texts=texts)) == 2
    assert "s-tail matches no fixture seed text" in caplog.text


def test_read_writes_the_artifact_and_prints_one_line_per_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arm = _fixture(tmp_path, task=True)

    assert _run(monkeypatch, tmp_path, arm, _store(users=["u1"])) == 0

    report = json.loads((tmp_path / "out/reach/entity_reach.json").read_text("utf-8"))
    assert (report["arm_name"], report["store_user_id"]) == ("reach", "u1")
    assert (report["store_traces"], report["entities"]) == (3, 3)
    assert report["single_trace_entities"] == 2
    assert report["thresholds"] == [0.9, 0.75]
    assert report["mentions"]["tail"] == ["reservation"]
    rules = {rule["rule"]: rule for rule in report["rules"]}
    assert list(rules) == ["exact", "containment", "embedding>=0.90", "embedding>=0.75"]
    assert rules["exact"]["questions"] == [
        {
            "id": "t3q01",
            "tier_label": "t3",
            "required": "tail",
            "seed_entities": ["thai place on main street"],
            "hop": None,
            "candidates": None,
        }
    ]
    joined = rules["embedding>=0.75"]
    assert joined["pair_count"] == 1
    assert joined["pairs"][0]["similarity"] == pytest.approx(0.8)
    assert joined["label_counts"] == {
        "same": 1,
        "related": 0,
        "different": 0,
        "unlabeled": 0,
    }
    assert (joined["questions"][0]["hop"], joined["questions"][0]["candidates"]) == (
        1,
        3,
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split()[0] for line in lines[-4:]] == list(rules)
    assert "t3_reach=1/1" in lines[-1]
