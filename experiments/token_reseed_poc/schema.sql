CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS poc_token_stamps;
DROP TABLE IF EXISTS poc_recall_tokens;
DROP TABLE IF EXISTS poc_traces;

CREATE TABLE IF NOT EXISTS poc_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(768),
    significance FLOAT DEFAULT 0.5,
    emotional_valence FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_poc_traces_embedding
    ON poc_traces USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS poc_recall_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label TEXT NOT NULL,
    strength FLOAT DEFAULT 1.0,
    significance FLOAT DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT now(),
    last_activated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS poc_token_stamps (
    token_id UUID REFERENCES poc_recall_tokens(id) ON DELETE CASCADE,
    trace_id UUID REFERENCES poc_traces(id) ON DELETE CASCADE,
    stamped_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (token_id, trace_id)
);
