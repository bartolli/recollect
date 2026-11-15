"""Reader: one structured call per packet item, holding only the brief and that item."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from probe_cli.reading_packet import PacketItem
from probe_cli.reading_reader import (
    ReaderAnswer,
    ReaderOutputExistsError,
    ReaderRecord,
    answer_item,
    item_messages,
    run_reader,
)
from recollect.exceptions import ExtractionError

if TYPE_CHECKING:
    from pydantic import BaseModel
    from recollect.llm.types import Message

_BRIEF = "Brief text."
_ROOT = Path(__file__).resolve().parents[3]
_BRIEF_FILE = _ROOT / "packages/probe-cli/fixtures/qualification/reader_brief.md"
_FIRST = _ROOT / "out" / "qualification-haiku55" / "reading_first.json"


def _section(lines: list[str], heading: str) -> list[str]:
    start = lines.index(heading) + 1
    end = next((n for n in range(start, len(lines)) if not lines[n]), len(lines))
    return [line.removeprefix("- ") for line in lines[start:end]]


def test_brief_rulings_and_hint_rule_equal_the_first_reading() -> None:
    lines = _BRIEF_FILE.read_text(encoding="utf-8").splitlines()
    first = json.loads(_FIRST.read_text(encoding="utf-8"))

    assert _section(lines, "Earlier rulings:") == first["rulings"]
    rules = _section(lines, "Rules, fixed before any reading:")
    hint = [rule for rule in rules if rule.startswith("A retrieval hint")]
    assert hint == [first["ruling_on_added_claims"]]
    assert all(line == line.lstrip() for line in lines)


def _item(n: int, checks: int = 2) -> PacketItem:
    return PacketItem(
        item_id=f"item-{n:04d}",
        statement=f"Statement {n}.",
        checks=[f"Check {n}.{c}?" for c in range(checks)],
        kind="context" if n % 2 else "line",
        representation=f"Representation {n}.",
    )


class _FakeReader:
    model_name = "fake"

    def __init__(self, fail: frozenset[str] = frozenset(), answers: int = 2) -> None:
        self.calls: list[list[Message]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.fail = fail
        self.answers = answers

    async def complete(self, messages: list[Message], **kwargs: Any) -> str:
        raise AssertionError("the reader asks for structured output")

    async def complete_structured(
        self, messages: list[Message], output_type: type[BaseModel], **kwargs: Any
    ) -> Any:
        assert output_type is ReaderAnswer
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        if any(f"Statement {n}." in messages[1].content for n in self.fail):
            raise ExtractionError("model refused")
        return ReaderAnswer(
            answers=["yes", "no", "not_applicable"][: self.answers],
            added_claim_quotes=["a span"],
        )


async def _run(
    provider: _FakeReader, out: Path, items: list[PacketItem], *, resume: bool = False
) -> list[ReaderRecord]:
    ask = partial(answer_item, provider=provider, brief=_BRIEF, max_tokens=4096)
    return await run_reader(items, ask=ask, out_path=out, resume=resume)


async def test_one_call_per_item_holding_only_the_brief_and_that_item(
    tmp_path: Path,
) -> None:
    items = [_item(1), _item(2), _item(3)]
    provider = _FakeReader()

    records = await _run(provider, tmp_path / "reader_a.jsonl", items)

    assert len(provider.calls) == len(items)
    assert all(kw == {"max_tokens": 4096} for kw in provider.kwargs)
    for item, messages in zip(items, provider.calls, strict=True):
        assert messages == item_messages(item, brief=_BRIEF)
        system, user = messages
        assert (system.role, system.content) == ("system", _BRIEF)
        assert user.role == "user"
        assert item.statement in user.content
        assert f"1. {item.checks[0]}\n2. {item.checks[1]}" in user.content
        assert item.kind in user.content
        assert item.representation in user.content
        assert "item-" not in user.content
        others = [o for o in items if o is not item]
        assert not any(o.representation in user.content for o in others)
    assert [r.success for r in records] == [True, True, True]
    assert records[0].answers == ["yes", "no"]
    assert records[0].added_claim_quotes == ["a span"]


async def test_failed_call_and_wrong_answer_count_are_recorded_as_failed(
    tmp_path: Path,
) -> None:
    items = [_item(1), _item(2, checks=3)]
    out = tmp_path / "reader_a.jsonl"

    records = await _run(_FakeReader(fail=frozenset({"1"})), out, items)

    assert [(r.item_id, r.success) for r in records] == [
        ("item-0001", False),
        ("item-0002", False),
    ]
    assert "model refused" in records[0].error
    assert records[1].error == "2 answers for 3 checks"
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(w["item_id"], w["success"]) for w in written] == [
        ("item-0001", False),
        ("item-0002", False),
    ]


async def test_existing_output_is_refused_without_resume(tmp_path: Path) -> None:
    out = tmp_path / "reader_a.jsonl"
    out.write_text("", encoding="utf-8")
    provider = _FakeReader()

    with pytest.raises(ReaderOutputExistsError):
        await _run(provider, out, [_item(1)])
    assert provider.calls == []


async def test_resume_calls_only_items_without_a_successful_answer(
    tmp_path: Path,
) -> None:
    items = [_item(1), _item(2), _item(3)]
    out = tmp_path / "reader_a.jsonl"
    await _run(_FakeReader(fail=frozenset({"2"})), out, items)
    provider = _FakeReader()

    records = await _run(provider, out, items, resume=True)

    assert len(provider.calls) == 1
    assert "Statement 2." in provider.calls[0][1].content
    assert [(r.item_id, r.success) for r in records] == [
        ("item-0001", True),
        ("item-0002", True),
        ("item-0003", True),
    ]
    written = [json.loads(line)["item_id"] for line in out.read_text().splitlines()]
    assert written == ["item-0001", "item-0002", "item-0003"]
