"""Re-embed every stored vector with v0.7.0 task-prefixed FastEmbed.

Three surfaces affected:

  memory_traces.embedding       source=content   task=search_document
  persona_facts.embedding       source=content   task=search_document
  concept_embeddings.embedding  source=concept   task=search_document

Idempotent: same source text + same task prefix yields the same vector;
rerun on partial failure is safe.

Operator preconditions (not enforced — operator's responsibility):
  pg_dump -Fc $DATABASE_URL > recollect.$(date +%Y%m%d).dump
  pkill -f recollect-mcp
"""

from __future__ import annotations

# ruff: noqa: S608
# SQL identifiers (table/column names) come from the module-level SURFACES
# constant. asyncpg cannot parameter-bind identifiers, so f-string substitution
# is the only path; the inputs are trusted by construction.
import argparse
import asyncio
import logging
import os
import sys
import time

import asyncpg
from recollect.embeddings import FastEmbedProvider
from recollect.exceptions import EmbeddingError, StorageError
from recollect.storage_utils import embedding_to_pgvector

logger = logging.getLogger("re_embed_v07")


SURFACES: list[dict[str, str]] = [
    {
        "name": "memory_traces",
        "id_col": "id",
        "source_col": "content",
        "embedding_col": "embedding",
        "index_name": "idx_traces_embedding",
    },
    {
        "name": "persona_facts",
        "id_col": "id",
        "source_col": "content",
        "embedding_col": "embedding",
        "index_name": "idx_facts_embedding",
    },
    {
        "name": "concept_embeddings",
        "id_col": "id",
        "source_col": "concept",
        "embedding_col": "embedding",
        "index_name": "idx_concept_emb_hnsw",
    },
]


async def assert_pgvector(conn: asyncpg.Connection) -> None:
    row = await conn.fetchrow(
        "SELECT extname FROM pg_extension WHERE extname = 'vector'"
    )
    if row is None:
        raise StorageError("pgvector extension not installed")


async def count_surface(conn: asyncpg.Connection, spec: dict[str, str]) -> int:
    sql = (
        f"SELECT COUNT(*) FROM {spec['name']} "
        f"WHERE {spec['source_col']} IS NOT NULL "
        f"AND length({spec['source_col']}) > 0"
    )
    n = await conn.fetchval(sql)
    return int(n or 0)


async def confirm_or_abort(counts: dict[str, int], yes: bool) -> None:
    total = sum(counts.values())
    logger.info("plan: %d eligible rows across %d surfaces", total, len(counts))
    for name, n in counts.items():
        logger.info("  surface=%s rows=%d", name, n)
    if yes:
        return
    sys.stderr.write(
        "\nType 'yes' to proceed (assumes DB backup + MCP stopped): "
    )
    sys.stderr.flush()
    answer = await asyncio.to_thread(sys.stdin.readline)
    if answer.strip().lower() != "yes":
        raise SystemExit("aborted by operator")


async def re_embed_batch(
    conn: asyncpg.Connection,
    spec: dict[str, str],
    embedder: FastEmbedProvider,
    rows: list[asyncpg.Record],
    dry_run: bool,
) -> int:
    ids = [r[spec["id_col"]] for r in rows]
    texts = [r[spec["source_col"]] for r in rows]
    try:
        embeddings = await embedder.generate_embeddings_batch(
            texts, task="search_document"
        )
    except EmbeddingError:
        logger.exception("embedding batch failed for surface=%s", spec["name"])
        return 0
    if dry_run:
        return len(rows)
    sql_update = (
        f"UPDATE {spec['name']} "
        f"SET {spec['embedding_col']} = $2::vector "
        f"WHERE {spec['id_col']} = $1"
    )
    async with conn.transaction():
        for row_id, emb in zip(ids, embeddings, strict=True):
            await conn.execute(sql_update, row_id, embedding_to_pgvector(emb))
    return len(rows)


async def re_embed_surface(
    conn: asyncpg.Connection,
    embedder: FastEmbedProvider,
    spec: dict[str, str],
    batch_size: int,
    dry_run: bool,
) -> int:
    sql = (
        f"SELECT {spec['id_col']}, {spec['source_col']} FROM {spec['name']} "
        f"WHERE {spec['source_col']} IS NOT NULL "
        f"AND length({spec['source_col']}) > 0"
    )
    rows = await conn.fetch(sql)
    total = len(rows)
    if total == 0:
        logger.info("surface=%s empty after filter; skipped", spec["name"])
        return 0
    logger.info(
        "surface=%s rows=%d batch=%d dry_run=%s",
        spec["name"], total, batch_size, dry_run,
    )
    touched = 0
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        touched += await re_embed_batch(conn, spec, embedder, batch, dry_run)
        logger.info(
            "surface=%s progress=%d/%d", spec["name"], touched, total,
        )
    return touched


async def reindex_surface(conn: asyncpg.Connection, index_name: str) -> None:
    logger.info("reindex begin: %s", index_name)
    t0 = time.monotonic()
    await conn.execute(f"REINDEX INDEX {index_name}")
    logger.info(
        "reindex done: %s elapsed=%.1fs",
        index_name, time.monotonic() - t0,
    )


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set; pass --db-url")
        return 2

    skip = set(args.skip or [])
    selected = [s for s in SURFACES if s["name"] not in skip]
    if not selected:
        logger.error("all surfaces skipped; nothing to do")
        return 2

    conn = await asyncpg.connect(db_url)
    try:
        await assert_pgvector(conn)
        counts = {s["name"]: await count_surface(conn, s) for s in selected}
        await confirm_or_abort(counts, args.yes)

        embedder = FastEmbedProvider()
        await embedder.warm()

        t_start = time.monotonic()
        for spec in selected:
            t0 = time.monotonic()
            touched = await re_embed_surface(
                conn, embedder, spec, args.batch_size, args.dry_run
            )
            logger.info(
                "surface=%s touched=%d elapsed=%.1fs",
                spec["name"], touched, time.monotonic() - t0,
            )
            if not args.dry_run and touched > 0:
                await reindex_surface(conn, spec["index_name"])
        logger.info(
            "complete: total_elapsed=%.1fs", time.monotonic() - t_start,
        )
    finally:
        await conn.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-url", default=None, help="overrides DATABASE_URL env var"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--yes", action="store_true", help="skip interactive confirmation"
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--skip", action="append",
        help="surface name to skip; repeatable",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
