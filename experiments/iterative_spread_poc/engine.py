import logging
from dataclasses import dataclass
from pathlib import Path

import asyncpg
from recollect.embeddings import FastEmbedProvider

from experiments.iterative_spread_poc.config import PocConfig

logger = logging.getLogger(__name__)


@dataclass
class StoredTrace:
    id: str
    content: str
    similarity: float = 0.0


class PocEngine:
    def __init__(self, config: PocConfig) -> None:
        self._config = config
        self._pool: asyncpg.Pool | None = None
        self._embedder = FastEmbedProvider(
            model_name=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )

    async def setup(self) -> None:
        """Create pool, run schema, warm embeddings."""
        logger.info("Setting up PocEngine with database %s", self._config.database_url)
        self._pool = await asyncpg.create_pool(self._config.database_url)

        schema_path = Path(__file__).parent / "schema.sql"
        schema_sql = schema_path.read_text()
        # Drop and recreate tables to handle dimension changes
        dim = str(self._config.embedding_dimensions)
        schema_sql = schema_sql.replace("vector(768)", f"vector({dim})")
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DROP TABLE IF EXISTS poc_iter_associations CASCADE;"
                "DROP TABLE IF EXISTS poc_iter_traces CASCADE;"
            )
            await conn.execute(schema_sql)
        logger.info("Schema applied (embedding dimensions=%s)", dim)

        await self._embedder.warm()
        logger.info("PocEngine setup complete")

    async def teardown(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PocEngine not set up; call setup() first")
        return self._pool

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return await self._embedder.generate_embedding(text)

    async def store(self, content: str) -> StoredTrace:
        """Embed and store content in poc_iter_traces. Return StoredTrace."""
        embedding = await self._embedder.generate_embedding(content)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO poc_iter_traces (content, embedding)"
                " VALUES ($1, $2) RETURNING id",
                content,
                embedding_str,
            )
        trace_id = str(row["id"])
        logger.info("Stored trace %s: %s", trace_id[:8], content[:60])
        return StoredTrace(id=trace_id, content=content)

    async def create_association(
        self,
        source_id: str,
        target_id: str,
        association_type: str = "entity",
        forward_strength: float = 0.8,
        backward_strength: float = 0.5,
    ) -> None:
        """Create an association between two traces."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO poc_iter_associations"
                " (source_trace_id, target_trace_id, association_type,"
                " forward_strength, backward_strength)"
                " VALUES ($1::uuid, $2::uuid, $3, $4, $5)"
                " ON CONFLICT DO NOTHING",
                source_id,
                target_id,
                association_type,
                forward_strength,
                backward_strength,
            )
        logger.info(
            "Created association %s -> %s (%s, fwd=%.2f, bwd=%.2f)",
            source_id[:8],
            target_id[:8],
            association_type,
            forward_strength,
            backward_strength,
        )
