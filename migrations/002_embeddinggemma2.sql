-- 002_embeddinggemma2.sql
-- Dense embeddings move from BGE-M3 (1024) to EmbeddingGemma 2 (768).
-- Vectors can't be converted, so chunks are deleted and every ingested document is queued again
-- with the parser it last used. Run the worker afterwards to re-embed.

BEGIN;

DROP INDEX ix_chunks_embedding_bq;

DELETE FROM chunks;

ALTER TABLE chunks ALTER COLUMN embedding TYPE halfvec(768);

CREATE INDEX ix_chunks_embedding_bq ON chunks
    USING hnsw ((binary_quantize(embedding)::bit(768)) bit_hamming_ops);

INSERT INTO ingestion_jobs (document_id, parser)
SELECT DISTINCT ON (document_id) document_id, parser
FROM ingestion_jobs
WHERE status = 'done'
ORDER BY document_id, finished_at DESC;

UPDATE documents SET status = 'pending', num_chunks = 0, error = NULL;

COMMIT;
