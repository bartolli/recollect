"""Reader: one model call per packet item, holding the brief and that item alone."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field
from recollect.exceptions import ExtractionError
from recollect.llm.types import Message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from recollect.llm.protocol import LLMProvider

    from probe_cli.reading_packet import PacketItem

logger = logging.getLogger(__name__)

Answer = Literal["yes", "no", "not_applicable"]


class ReaderAnswer(BaseModel):
    # One per check, in check order.
    answers: list[Answer]
    # Verbatim spans of the representation.
    added_claim_quotes: list[str]
    note: str = ""


class ReaderRecord(BaseModel):
    item_id: str
    answers: list[Answer] = Field(default_factory=list)
    added_claim_quotes: list[str] = Field(default_factory=list)
    note: str = ""
    success: bool
    error: str = ""


class ReaderSeat(BaseModel):
    model: str = Field(min_length=1)
    max_tokens: int = Field(ge=1)
    model_settings: dict[str, Any] = Field(default_factory=dict)


class ReaderMeta(BaseModel):
    reader: str
    model: str
    max_tokens: int = Field(ge=1)
    model_settings: dict[str, Any]
    brief_sha256: str
    packet_sha256: str
    answered: int = Field(ge=0)
    failed: int = Field(ge=0)


class ReaderOutputExistsError(ValueError):
    pass


def item_messages(item: PacketItem, *, brief: str) -> list[Message]:
    checks = "\n".join(f"{n}. {c}" for n, c in enumerate(item.checks, start=1))
    user = (
        f"Statement:\n{item.statement}\n\n"
        f"Checks:\n{checks}\n\n"
        f"Representation kind: {item.kind}\n\n"
        f"Representation:\n{item.representation}"
    )
    return [Message(role="system", content=brief), Message(role="user", content=user)]


async def answer_item(
    item: PacketItem, *, provider: LLMProvider, brief: str, max_tokens: int
) -> ReaderRecord:
    """A failed call is recorded, not retried beyond the provider's own retries."""
    try:
        answer = await provider.complete_structured(
            item_messages(item, brief=brief), ReaderAnswer, max_tokens=max_tokens
        )
    except ExtractionError as exc:
        logger.exception("reader item=%s failed", item.item_id)
        return ReaderRecord(item_id=item.item_id, success=False, error=str(exc))
    if len(answer.answers) != len(item.checks):
        error = f"{len(answer.answers)} answers for {len(item.checks)} checks"
        logger.warning("reader item=%s %s", item.item_id, error)
        return ReaderRecord(
            item_id=item.item_id, success=False, error=error, **answer.model_dump()
        )
    return ReaderRecord(item_id=item.item_id, success=True, **answer.model_dump())


def parse_reader_records(text: str) -> list[ReaderRecord]:
    lines = text.splitlines()
    return [ReaderRecord.model_validate_json(line) for line in lines if line.strip()]


def _answered(out_path: Path, *, resume: bool) -> dict[str, ReaderRecord]:
    if not out_path.exists():
        return {}
    if not resume:
        raise ReaderOutputExistsError(
            f"{out_path} exists; pass --resume to continue it"
        )
    records = parse_reader_records(out_path.read_text(encoding="utf-8"))
    return {r.item_id: r for r in records if r.success}


async def run_reader(
    items: Sequence[PacketItem],
    *,
    ask: Callable[[PacketItem], Awaitable[ReaderRecord]],
    out_path: Path,
    resume: bool,
) -> list[ReaderRecord]:
    """One record per item in packet order; resume calls only items not yet answered."""
    answered = _answered(out_path, resume=resume)
    records: list[ReaderRecord] = []
    for item in items:
        record = answered.get(item.item_id)
        if record is None:
            record = await ask(item)
            # Appended as it lands: an interrupted run resumes from here.
            with out_path.open("a", encoding="utf-8") as out:
                out.write(f"{record.model_dump_json()}\n")
        records.append(record)
    out_path.write_text(
        "".join(f"{r.model_dump_json()}\n" for r in records), encoding="utf-8"
    )
    return records
