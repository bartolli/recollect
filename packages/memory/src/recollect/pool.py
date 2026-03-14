"""Connection pool and schema management for PostgreSQL storage.

Owns the asyncpg connection pool and DDL execution. Sub-stores receive
the PoolManager instance and use it to acquire connections.
"""

from __future__ import annotations

import logging

import asyncpg

from recollect.config import config
from recollect.exceptions import StorageError

logger = logging.getLogger(__name__)

SCHEMA_SQL = """\
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT DEFAULT '',
    status TEXT DEFAULT 'active',
    summary_trace_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS memory_traces (
    id TEXT PRIMARY KEY,
    content TEXT,
    pattern JSONB DEFAULT '{}',
    context JSONB DEFAULT '{}',
    embedding vector(768),
    strength FLOAT DEFAULT 0.3,
    activation_count INTEGER DEFAULT 0,
    retrieval_count INTEGER DEFAULT 0,
    last_activation TIMESTAMPTZ,
    last_retrieval TIMESTAMPTZ,
    consolidated BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    decay_rate FLOAT DEFAULT 0.1,
    emotional_valence FLOAT DEFAULT 0.0,
    significance FLOAT DEFAULT 0.1,
    session_id TEXT,
    user_id TEXT,
    content_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(content, ''))
    ) STORED
);

CREATE TABLE IF NOT EXISTS associations (
    id TEXT PRIMARY KEY,
    source_trace_id TEXT REFERENCES memory_traces(id) ON DELETE CASCADE,
    target_trace_id TEXT REFERENCES memory_traces(id) ON DELETE CASCADE,
    association_type TEXT DEFAULT 'semantic',
    weight FLOAT DEFAULT 0.5,
    forward_strength FLOAT DEFAULT 0.5,
    backward_strength FLOAT DEFAULT 0.5,
    activation_count INTEGER DEFAULT 0,
    last_activation TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CHECK (source_trace_id != target_trace_id),
    pair_key TEXT GENERATED ALWAYS AS (
        LEAST(source_trace_id, target_trace_id) || '|' ||
        GREATEST(source_trace_id, target_trace_id) || '|' ||
        association_type
    ) STORED
);

CREATE TABLE IF NOT EXISTS trace_entities (
    entity_name TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    trace_id TEXT REFERENCES memory_traces(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_name, trace_id)
);

CREATE TABLE IF NOT EXISTS trace_concepts (
    concept TEXT NOT NULL,
    trace_id TEXT REFERENCES memory_traces(id) ON DELETE CASCADE,
    PRIMARY KEY (concept, trace_id)
);

CREATE TABLE IF NOT EXISTS persona_facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    content TEXT NOT NULL,
    source_trace_id TEXT REFERENCES memory_traces(id) ON DELETE SET NULL,
    confidence FLOAT DEFAULT 0.8,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    superseded_by TEXT REFERENCES persona_facts(id) ON DELETE SET NULL,
    status TEXT DEFAULT 'candidate',
    mention_count INTEGER DEFAULT 1,
    scope TEXT DEFAULT 'general',
    context_tags TEXT[] DEFAULT '{}',
    embedding vector(768),
    user_id TEXT
);

CREATE TABLE IF NOT EXISTS entity_relations (
    id TEXT PRIMARY KEY,
    source_entity TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    context TEXT DEFAULT '',
    weight FLOAT DEFAULT 0.5,
    source_trace_id TEXT REFERENCES memory_traces(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS concept_embeddings (
    id TEXT PRIMARY KEY,
    concept TEXT NOT NULL,
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recall_tokens (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    strength FLOAT DEFAULT 1.0,
    significance FLOAT DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activated_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS token_stamps (
    token_id TEXT REFERENCES recall_tokens(id) ON DELETE CASCADE,
    trace_id TEXT REFERENCES memory_traces(id) ON DELETE CASCADE,
    stamped_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (token_id, trace_id)
);

CREATE INDEX IF NOT EXISTS idx_traces_strength
    ON memory_traces (strength);
CREATE INDEX IF NOT EXISTS idx_traces_created
    ON memory_traces (created_at);
CREATE INDEX IF NOT EXISTS idx_assoc_source
    ON associations (source_trace_id);
CREATE INDEX IF NOT EXISTS idx_assoc_target
    ON associations (target_trace_id);
CREATE INDEX IF NOT EXISTS idx_trace_entities_name
    ON trace_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_trace_concepts_concept
    ON trace_concepts(concept);
CREATE INDEX IF NOT EXISTS idx_persona_facts_subject
    ON persona_facts(subject);
CREATE INDEX IF NOT EXISTS idx_persona_facts_spo
    ON persona_facts(subject, predicate, object);
CREATE INDEX IF NOT EXISTS idx_trace_entities_trgm
    ON trace_entities USING gin (entity_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_traces_content_fts
    ON memory_traces USING gin (content_tsv);
CREATE UNIQUE INDEX IF NOT EXISTS idx_assoc_pair_key
    ON associations(pair_key);
CREATE INDEX IF NOT EXISTS idx_entity_rel_source
    ON entity_relations(source_entity);
CREATE INDEX IF NOT EXISTS idx_entity_rel_target
    ON entity_relations(target_entity);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_rel_pair
    ON entity_relations(source_entity, relation, target_entity);
CREATE INDEX IF NOT EXISTS idx_concept_emb_owner
    ON concept_embeddings(owner_type, owner_id);

CREATE INDEX IF NOT EXISTS idx_persona_facts_status ON persona_facts(status);
CREATE INDEX IF NOT EXISTS idx_persona_facts_tags
    ON persona_facts USING gin (context_tags);

CREATE INDEX IF NOT EXISTS idx_traces_session ON memory_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_user ON memory_traces(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_facts_user ON persona_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_token_stamps_trace ON token_stamps(trace_id);
CREATE INDEX IF NOT EXISTS idx_token_stamps_token ON token_stamps(token_id);
"""

# HNSW index created separately -- requires pgvector 0.5+ but works on
# empty tables unlike IVFFlat.
VECTOR_INDEX_SQL = """\
CREATE INDEX IF NOT EXISTS idx_traces_embedding
    ON memory_traces USING hnsw (embedding vector_cosine_ops);
"""

FACT_VECTOR_INDEX_SQL = """\
CREATE INDEX IF NOT EXISTS idx_facts_embedding
    ON persona_facts USING hnsw (embedding vector_cosine_ops);
"""

CONCEPT_VECTOR_INDEX_SQL = """\
CREATE INDEX IF NOT EXISTS idx_concept_emb_hnsw
    ON concept_embeddings USING hnsw (embedding vector_cosine_ops);
"""


class PoolManager:
    """Manages the asyncpg connection pool and schema initialization."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url or config.database_url
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None

    async def get_pool(self) -> asyncpg.Pool[asyncpg.Record]:
        """Get the connection pool, creating the database if needed."""
        if self._pool is not None:
            return self._pool
        try:
            self._pool = await asyncpg.create_pool(
                self._database_url, min_size=2, max_size=10
            )
        except asyncpg.InvalidCatalogNameError:
            await self._create_database()
            self._pool = await asyncpg.create_pool(
                self._database_url, min_size=2, max_size=10
            )
        except Exception as exc:
            raise StorageError(f"Failed to create connection pool: {exc}") from exc
        return self._pool

    async def _create_database(self) -> None:
        """Create the database if it does not exist.

        Connects to the default 'postgres' database to issue CREATE DATABASE,
        then returns so the caller can retry the pool connection.
        """
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(self._database_url)
        db_name = parsed.path.lstrip("/")
        # Connect to the default 'postgres' database
        admin_url = urlunparse(parsed._replace(path="/postgres"))

        try:
            conn = await asyncpg.connect(admin_url)
            try:
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                logger.info("Created database %s", db_name)
            finally:
                await conn.close()
        except asyncpg.DuplicateDatabaseError:
            pass  # Race condition: another process created it
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to create database {db_name}: {exc}") from exc

    async def initialize(self) -> None:
        """Create schema tables and indexes."""
        try:
            pool = await self.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(SCHEMA_SQL)
                try:
                    await conn.execute(VECTOR_INDEX_SQL)
                    await conn.execute(FACT_VECTOR_INDEX_SQL)
                    await conn.execute(CONCEPT_VECTOR_INDEX_SQL)
                except asyncpg.UndefinedObjectError:
                    logger.exception(
                        "pgvector HNSW not available, skipping vector index"
                    )
            logger.info("Storage schema initialized")
        except StorageError:
            raise
        except asyncpg.PostgresError as exc:
            raise StorageError(f"Failed to initialize schema: {exc}") from exc

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Storage connection closed")

    @staticmethod
    def get_schema_sql() -> str:
        """Return the schema DDL for inspection/testing."""
        return SCHEMA_SQL
