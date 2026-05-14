"""Operator CLI: python -m recollect.bootstrap [--check | --apply].

--apply runs bootstrap.apply on the configured DB, applying all pending
migrations. --check is read-only: lists pending migrations + preflight
invariant status without writing.

DATABASE_URL env var or --db-url overrides config-resolved URL.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import asyncpg

from recollect.bootstrap.migrations import default_registry
from recollect.bootstrap.preflight import preflight_all, raise_on_failures
from recollect.bootstrap.runner import (
    _APPLIED_MIGRATIONS_DDL,
    _APPLIED_NAMES_SQL,
    apply,
)
from recollect.config import config as recollect_config
from recollect.embeddings import FastEmbedProvider

logger = logging.getLogger("recollect.bootstrap.cli")


async def _check(pool: asyncpg.Pool[asyncpg.Record]) -> int:
    registry = default_registry()
    async with pool.acquire() as conn:
        await conn.execute(_APPLIED_MIGRATIONS_DDL)
        applied = [r["name"] for r in await conn.fetch(_APPLIED_NAMES_SQL)]
    pending = [m.name for m in registry.pending(applied)]
    ahead = registry.schema_ahead(applied)

    logger.info("registered=%s", registry.names())
    logger.info("applied=%s", applied)
    logger.info("pending=%s", pending)
    if ahead:
        logger.warning("schema_ahead=%s", ahead)

    provider = FastEmbedProvider()
    invariants = await preflight_all(pool, provider)
    for inv in invariants:
        level = logging.INFO if inv.ok else logging.WARNING
        logger.log(level, "preflight %s: ok=%s detail=%s", inv.name, inv.ok, inv.detail)

    failures = [i for i in invariants if not i.ok]
    return 0 if not (pending or ahead or failures) else 1


async def _apply(pool: asyncpg.Pool[asyncpg.Record]) -> int:
    result = await apply(pool, default_registry())
    logger.info(
        "applied=%s skipped=%s schema_ahead=%s",
        result.applied, result.skipped, result.schema_ahead,
    )
    provider = FastEmbedProvider()
    invariants = await preflight_all(pool, provider)
    raise_on_failures(invariants)
    return 0


async def _run(args: argparse.Namespace) -> int:
    db_url = (
        args.db_url
        or os.environ.get("DATABASE_URL")
        or recollect_config.database_url
    )
    if not db_url:
        logger.error("DATABASE_URL not set; pass --db-url")
        return 2
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=4)
    try:
        if args.check:
            return await _check(pool)
        return await _apply(pool)
    finally:
        await pool.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db-url", default=None, help="overrides DATABASE_URL env var")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="read-only: list pending")
    group.add_argument("--apply", action="store_true", help="run pending migrations")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
