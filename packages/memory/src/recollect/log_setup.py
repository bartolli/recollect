"""Structured logging: JSON Lines file output and @logged decorator."""

from __future__ import annotations

import functools
import inspect
import logging
import sys
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson


class JSONLineFormatter(logging.Formatter):
    """Logging formatter that outputs one JSON object per line.

    Keys: ts, level, logger, msg. Adds exc with formatted traceback
    when an exception is attached to the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = "".join(traceback.format_exception(*record.exc_info))
        return orjson.dumps(entry).decode()


def logged[F: Callable[..., Any]](func: F) -> F:
    """Decorator that logs entry, completion with duration, and exceptions.

    Works on both async and sync functions. Uses DEBUG for entry/completion
    and ERROR (with traceback) for exceptions. Re-raises after logging.
    """
    logger = logging.getLogger(func.__module__)
    qualname = func.__qualname__

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.debug("%s called", qualname)
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
            except Exception:
                elapsed = time.monotonic() - start
                logger.exception("%s failed after %.3fs", qualname, elapsed)
                raise
            elapsed = time.monotonic() - start
            logger.debug("%s completed in %.3fs", qualname, elapsed)
            return result

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug("%s called", qualname)
        start = time.monotonic()
        try:
            result = func(*args, **kwargs)
        except Exception:
            elapsed = time.monotonic() - start
            logger.exception("%s failed after %.3fs", qualname, elapsed)
            raise
        elapsed = time.monotonic() - start
        logger.debug("%s completed in %.3fs", qualname, elapsed)
        return result

    return sync_wrapper  # type: ignore[return-value]


def configure_logging(
    *,
    level: int = logging.WARNING,
    log_file: str | Path | None = None,
    file_level: int = logging.DEBUG,
    verbose: bool = False,
) -> None:
    """Set up dual output: human-readable to stderr, JSON Lines to file.

    Args:
        level: Console handler log level (default WARNING).
        log_file: Path for JSON Lines file output. No file handler if None.
        file_level: File handler log level (default DEBUG).
        verbose: If True, override console level to DEBUG.
    """
    if verbose:
        level = logging.DEBUG

    root = logging.getLogger()
    root.setLevel(min(level, file_level))

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(JSONLineFormatter())
        root.addHandler(file_handler)
