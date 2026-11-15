"""Qualification reading commands: shuffled packet, one reader seat per run, tally."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tomllib
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from recollect.config import config as sdk_config
from recollect.llm.pydantic_ai import PydanticAIProvider
from recollect.llm.types import CompletionParams

from probe_cli.reading_packet import (
    PacketError,
    PacketMeta,
    build_packet,
    packet_meta,
    parse_key,
    parse_listings,
    parse_packet_items,
    parse_reader_key,
    read_source,
    write_packet,
)
from probe_cli.reading_reader import (
    ReaderMeta,
    ReaderRecord,
    ReaderSeat,
    answer_item,
    parse_reader_records,
    run_reader,
)
from probe_cli.reading_tally import (
    TallyError,
    TallyInputs,
    TallyRequest,
    summary,
    tally,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from probe_cli.reading_packet import PacketItem

logger = logging.getLogger(__name__)

_THRESHOLD_KEY = "persona.confidence_threshold"


class PacketConfig(BaseModel):
    seed: int
    reader_key: Path
    brief: Path


class ReadingConfig(BaseModel):
    packet: PacketConfig
    readers: dict[str, ReaderSeat]


def load_reading_config(path: Path) -> ReadingConfig:
    return ReadingConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def _source(value: str) -> tuple[str, Path]:
    label, sep, path = value.partition("=")
    if not (sep and label and path):
        raise argparse.ArgumentTypeError(f"{value!r} is not LABEL=PATH")
    return label, Path(path)


def add_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    packet = sub.add_parser(
        "reading-packet",
        help="Build the shuffled qualification reading packet and its key.",
    )
    packet.add_argument("--config", required=True, type=Path, help="reading.toml.")
    packet.add_argument(
        "--source",
        required=True,
        action="append",
        type=_source,
        metavar="LABEL=PATH",
        help="A representations.json under a label; repeat per source.",
    )
    packet.add_argument("--out-dir", required=True, type=Path, help="Packet dir.")
    packet.set_defaults(verbose=False, env_file=None)
    _add_reader_parser(sub)
    _add_tally_parser(sub)


def _add_reader_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    reader = sub.add_parser(
        "read-packet",
        help="Answer every packet item with one reader seat from reading.toml.",
    )
    reader.add_argument("--config", required=True, type=Path, help="reading.toml.")
    reader.add_argument("--reader", required=True, help="Seat under [readers].")
    reader.add_argument("--packet-dir", required=True, type=Path, help="Packet dir.")
    reader.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing answer file; call only unanswered items.",
    )
    reader.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: ./.env). Pass empty string to skip.",
    )
    reader.set_defaults(verbose=False)


def _add_tally_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    tallied = sub.add_parser(
        "reading-tally",
        help="Tally both readers' answers per qualification check; writes tally.json.",
    )
    tallied.add_argument("--config", required=True, type=Path, help="reading.toml.")
    tallied.add_argument("--packet-dir", required=True, type=Path, help="Packet dir.")
    tallied.add_argument("--control", required=True, help="Control source label.")
    tallied.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate source label; give it twice, or not at all to calibrate.",
    )
    tallied.set_defaults(verbose=False, env_file=None)


async def dispatch(args: argparse.Namespace) -> int | None:
    """None when the command is not a reading command."""
    if args.cmd == "reading-packet":
        return _run_packet(args.config, sources=args.source, out_dir=args.out_dir)
    if args.cmd == "read-packet":
        return await _run_reader(
            args.config,
            reader=args.reader,
            packet_dir=args.packet_dir,
            resume=args.resume,
        )
    if args.cmd == "reading-tally":
        return _run_tally(
            args.config,
            packet_dir=args.packet_dir,
            control=args.control,
            candidates=args.candidate,
        )
    return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_packet(
    config_path: Path, *, sources: list[tuple[str, Path]], out_dir: Path
) -> int:
    labels = [label for label, _ in sources]
    if len(set(labels)) != len(labels):
        logger.error("a --source label repeats: %s", ", ".join(labels))
        return 2
    if (out_dir / "packet.jsonl").exists():
        logger.error("%s already holds a packet; one packet per directory", out_dir)
        return 2
    config = load_reading_config(config_path)
    key_bytes = config.packet.reader_key.read_bytes()
    try:
        rows = parse_reader_key(key_bytes.decode("utf-8"))
        loaded = [read_source(label, path) for label, path in sources]
        packet = build_packet(
            [(s.label, report) for s, report in loaded], rows, seed=config.packet.seed
        )
    except PacketError as exc:
        logger.error("%s", exc)
        return 2
    meta = packet_meta(
        packet,
        seed=config.packet.seed,
        sources=[s for s, _ in loaded],
        reader_key_sha256=_sha256(key_bytes),
    )
    write_packet(packet, meta, out_dir)
    _print_packet(meta)
    return 0


def _refusal(
    out_path: Path, meta_path: Path, start: ReaderMeta, *, resume: bool
) -> str | None:
    """Why an existing answer file may not be continued; None when it may."""
    if not out_path.exists():
        return None
    if not resume:
        return f"{out_path} exists; pass --resume to continue it"
    if not meta_path.exists():
        return f"{meta_path} is missing: no packet or brief hash to continue against"
    recorded = ReaderMeta.model_validate_json(meta_path.read_bytes())
    if (recorded.packet_sha256, recorded.brief_sha256) == (
        start.packet_sha256,
        start.brief_sha256,
    ):
        return None
    return (
        f"{meta_path} records packet {recorded.packet_sha256} and brief"
        f" {recorded.brief_sha256}; now packet {start.packet_sha256} and brief"
        f" {start.brief_sha256}"
    )


def _counted(
    meta: ReaderMeta, records: Sequence[ReaderRecord], total: int
) -> ReaderMeta:
    # failed counts items without a successful answer, before and after a run.
    answered = len({r.item_id for r in records if r.success})
    return meta.model_copy(update={"answered": answered, "failed": total - answered})


def _write_meta(path: Path, meta: ReaderMeta) -> None:
    path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")


def _ask(
    seat: ReaderSeat, brief: str
) -> Callable[[PacketItem], Awaitable[ReaderRecord]]:
    # Defaults from the seat, never the process extraction.model_settings.
    provider = PydanticAIProvider(
        model=seat.model, defaults=CompletionParams(**seat.model_settings)
    )
    return partial(
        answer_item, provider=provider, brief=brief, max_tokens=seat.max_tokens
    )


def _start_meta(
    reader: str, seat: ReaderSeat, *, brief: bytes, packet: bytes
) -> ReaderMeta:
    return ReaderMeta(
        reader=reader,
        model=seat.model,
        max_tokens=seat.max_tokens,
        model_settings=seat.model_settings,
        brief_sha256=_sha256(brief),
        packet_sha256=_sha256(packet),
        answered=0,
        failed=0,
    )


async def _run_reader(
    config_path: Path, *, reader: str, packet_dir: Path, resume: bool
) -> int:
    config = load_reading_config(config_path)
    seat = config.readers.get(reader)
    if seat is None:
        logger.error("no [readers.%s] in %s", reader, config_path)
        return 2
    packet_bytes = (packet_dir / "packet.jsonl").read_bytes()
    brief_bytes = config.packet.brief.read_bytes()
    items = parse_packet_items(packet_bytes.decode("utf-8"))
    out_path = packet_dir / f"reader_{reader}.jsonl"
    meta_path = packet_dir / f"reader_{reader}_meta.json"
    start = _start_meta(reader, seat, brief=brief_bytes, packet=packet_bytes)
    refusal = _refusal(out_path, meta_path, start, resume=resume)
    if refusal is not None:
        logger.error("%s", refusal)
        return 2
    existing = (
        parse_reader_records(out_path.read_text(encoding="utf-8"))
        if out_path.exists()
        else []
    )
    # Written before the first call: an interrupted run keeps its hashes for --resume.
    _write_meta(meta_path, _counted(start, existing, len(items)))
    ask = _ask(seat, brief_bytes.decode("utf-8"))
    records = await run_reader(items, ask=ask, out_path=out_path, resume=resume)
    meta = _counted(start, records, len(items))
    _write_meta(meta_path, meta)
    _emit(
        f"reader {reader} model={seat.model}"
        f" answered={meta.answered} failed={meta.failed}"
    )
    # Exit 1 = an item without a successful answer; --resume calls only those.
    return 0 if meta.failed == 0 else 1


def _reader_records(packet_dir: Path, name: str) -> list[ReaderRecord]:
    path = packet_dir / f"reader_{name}.jsonl"
    if not path.exists():
        # Every item of this reader is then incomplete.
        logger.warning("no answers from reader %s at %s", name, path)
        return []
    return parse_reader_records(path.read_text(encoding="utf-8"))


def _tally_inputs(config: ReadingConfig, packet_dir: Path) -> TallyInputs:
    return TallyInputs(
        key=parse_key((packet_dir / "key.jsonl").read_text(encoding="utf-8")),
        items=parse_packet_items(
            (packet_dir / "packet.jsonl").read_text(encoding="utf-8")
        ),
        listings=parse_listings((packet_dir / "listings.json").read_bytes()),
        rows=parse_reader_key(config.packet.reader_key.read_text(encoding="utf-8")),
        answers={name: _reader_records(packet_dir, name) for name in config.readers},
    )


def _run_tally(
    config_path: Path, *, packet_dir: Path, control: str, candidates: list[str]
) -> int:
    threshold = sdk_config.get(_THRESHOLD_KEY)
    if threshold is None:
        logger.error("SDK config has no %s", _THRESHOLD_KEY)
        return 2
    request = TallyRequest(
        control=control, candidates=candidates, confidence_threshold=float(threshold)
    )
    try:
        report = tally(
            _tally_inputs(load_reading_config(config_path), packet_dir), request
        )
    except (PacketError, TallyError) as exc:
        logger.error("%s", exc)
        return 2
    (packet_dir / "tally.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    for line in summary(report):
        _emit(line)
    return 0


def _emit(text: str) -> None:
    # sys.stdout rather than print: T201 binds this package outside __main__.
    sys.stdout.write(text + "\n")


def _print_packet(meta: PacketMeta) -> None:
    kinds = "  ".join(f"{kind}={n}" for kind, n in meta.items_by_kind.items())
    _emit(f"packet seed={meta.seed}  {kinds}")
    for source in meta.sources:
        counts = "  ".join(
            f"{name}={n}" for name, n in meta.listing_counts[source.label].items()
        )
        _emit(f"{source.label}  validity={source.validity}  {counts}")
