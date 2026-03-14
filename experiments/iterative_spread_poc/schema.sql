CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS poc_iter_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_poc_iter_traces_embedding
    ON poc_iter_traces USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS poc_iter_associations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_trace_id UUID REFERENCES poc_iter_traces(id) ON DELETE CASCADE,
    target_trace_id UUID REFERENCES poc_iter_traces(id) ON DELETE CASCADE,
    association_type TEXT NOT NULL DEFAULT 'entity',
    forward_strength FLOAT NOT NULL DEFAULT 0.8,
    backward_strength FLOAT NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (source_trace_id, target_trace_id)
);

CREATE INDEX IF NOT EXISTS idx_poc_iter_assoc_source
    ON poc_iter_associations (source_trace_id);
CREATE INDEX IF NOT EXISTS idx_poc_iter_assoc_target
    ON poc_iter_associations (target_trace_id);
