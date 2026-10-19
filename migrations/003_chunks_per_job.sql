-- 003_chunks_per_job.sql
-- Chunks belong to the ingestion job that produced them, so one document can keep chunks from
-- several parsers at once. Status, errors, counts and model names move from documents/chunks
-- to the job. Existing chunks are assigned to their document's latest finished job.

BEGIN;

-- Jobs

ALTER TABLE ingestion_jobs
    ADD COLUMN num_chunks             INTEGER      NOT NULL DEFAULT 0,
    ADD COLUMN embedding_model        VARCHAR(255),
    ADD COLUMN sparse_embedding_model TEXT;

CREATE INDEX ix_ingestion_jobs_document_parser ON ingestion_jobs (document_id, parser, created_at);

-- Chunks: link to their job

ALTER TABLE chunks ADD COLUMN job_id UUID REFERENCES ingestion_jobs (id) ON DELETE CASCADE;
ALTER TABLE chunks ADD COLUMN boxes  JSONB;

UPDATE chunks c
SET job_id = latest.id
FROM (
    SELECT DISTINCT ON (document_id) id, document_id
    FROM ingestion_jobs
    WHERE status = 'done'
    ORDER BY document_id, finished_at DESC NULLS LAST
) latest
WHERE latest.document_id = c.document_id;

DELETE FROM chunks WHERE job_id IS NULL;

UPDATE chunks SET boxes = metadata -> 'boxes' WHERE metadata ? 'boxes';

UPDATE ingestion_jobs j
SET num_chunks             = stats.num_chunks,
    embedding_model        = stats.embedding_model,
    sparse_embedding_model = stats.sparse_embedding_model
FROM (
    SELECT job_id, count(*) AS num_chunks, max(embedding_model) AS embedding_model, max(sparse_embedding_model) AS sparse_embedding_model
    FROM chunks
    GROUP BY job_id
) stats
WHERE stats.job_id = j.id;

ALTER TABLE chunks ALTER COLUMN job_id SET NOT NULL;

ALTER TABLE chunks DROP CONSTRAINT uq_chunks_document_index;
ALTER TABLE chunks ADD CONSTRAINT uq_chunks_job_index UNIQUE (job_id, chunk_index);
CREATE INDEX ix_chunks_job_id ON chunks (job_id);

DROP INDEX ix_chunks_metadata;
ALTER TABLE chunks
    DROP COLUMN parent_id,
    DROP COLUMN metadata,
    DROP COLUMN embedding_model,
    DROP COLUMN sparse_embedding_model;

-- Documents: the stored file only; ingestion state lives on jobs

DROP INDEX ix_documents_status;
ALTER TABLE documents
    DROP COLUMN status,
    DROP COLUMN error,
    DROP COLUMN num_chunks,
    DROP COLUMN updated_at;
DROP TYPE document_status;

CREATE INDEX ix_documents_metadata ON documents USING gin (metadata jsonb_path_ops);

COMMIT;
