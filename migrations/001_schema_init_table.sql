-- 001_schema_init_table.sql
-- Initial schema matching app/db/model.py

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- Enums

CREATE TYPE document_status AS ENUM ('pending', 'processing', 'ready', 'failed');
CREATE TYPE source_type     AS ENUM ('pdf', 'pptx', 'markdown', 'text', 'json');
CREATE TYPE modality        AS ENUM ('text', 'table', 'image');
CREATE TYPE job_status      AS ENUM ('queued', 'running', 'done', 'failed');

-- Documents

CREATE TABLE documents (
    id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    filename      TEXT            NOT NULL,
    mime_type     VARCHAR(255)    NOT NULL,
    source_type   source_type     NOT NULL,
    size_bytes    BIGINT          NOT NULL,
    sha256        VARCHAR(64)     NOT NULL UNIQUE,
    storage_path  TEXT            NOT NULL,

    status        document_status NOT NULL DEFAULT 'pending',
    error         TEXT,
    num_chunks    INTEGER         NOT NULL DEFAULT 0,

    metadata      JSONB           NOT NULL DEFAULT '{}'::jsonb,

    created_at    TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX ix_documents_status ON documents (status);

-- Chunks

CREATE TABLE chunks (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),

    document_id      UUID          NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    parent_id        UUID          REFERENCES chunks (id) ON DELETE CASCADE,
    chunk_index      INTEGER       NOT NULL,

    modality         modality      NOT NULL,
    content          TEXT          NOT NULL,       -- text, table as markdown, or image caption
    image_path       TEXT,
    token_count      INTEGER,

    page             INTEGER,                      -- PDF page or PPTX slide
    heading_path     TEXT[],                       -- Markdown section path
    json_path        TEXT,                         -- e.g. $.items[3]

    embedding        HALFVEC(1024) NOT NULL,
    embedding_model  VARCHAR(255)  NOT NULL,       -- repo@revision, to detect stale vectors
    content_tsv      TSVECTOR      GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,

    metadata         JSONB         NOT NULL DEFAULT '{}'::jsonb,

    created_at       TIMESTAMPTZ   NOT NULL DEFAULT now(),

    CONSTRAINT uq_chunks_document_index UNIQUE (document_id, chunk_index)
);

CREATE INDEX ix_chunks_document_id ON chunks (document_id);
CREATE INDEX ix_chunks_modality    ON chunks (modality);
CREATE INDEX ix_chunks_content_tsv ON chunks USING gin (content_tsv);
CREATE INDEX ix_chunks_metadata    ON chunks USING gin (metadata jsonb_path_ops);

-- Binary-quantized HNSW index: fast candidate search, then rerank with the full halfvec.
-- Query must use the same expression, e.g.
--   ORDER BY binary_quantize(embedding)::bit(1024) <~> binary_quantize(:query)::bit(1024)
CREATE INDEX ix_chunks_embedding_bq ON chunks
    USING hnsw ((binary_quantize(embedding)::bit(1024)) bit_hamming_ops);

-- Ingestion jobs

CREATE TABLE ingestion_jobs (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),

    document_id  UUID         NOT NULL REFERENCES documents (id) ON DELETE CASCADE,

    status       job_status   NOT NULL DEFAULT 'queued',
    parser       VARCHAR(50)  NOT NULL DEFAULT 'fast',   -- "fast" or "docling"
    attempts     INTEGER      NOT NULL DEFAULT 0,
    error        TEXT,

    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,

    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX ix_ingestion_jobs_queue ON ingestion_jobs (status, created_at);

COMMIT;
