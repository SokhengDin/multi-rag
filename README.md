# Multi RAG

Research on encoding large multimodal documents with a VLM, and searching them with Postgres + pgvector alone.

> [!WARNING]
> **This is a research project, not production software.** The design, schema and this README will change as results come in. Don't depend on anything here staying stable.

The system is a multimodal RAG (retrieval-augmented generation) API. It ingests PDF, PPTX, Markdown, text and JSON files, splits them into text, table and image chunks, and searches them with hybrid retrieval: dense vectors, sparse vectors and full-text search in Postgres. Answers come from a local LLM or VLM running on CUDA, Apple Silicon (MPS or MLX) or CPU.

## Research questions

**RQ1. Does encoding large documents with a VLM improve retrieval over text-only parsing?**
Long PDFs and slide decks carry much of their meaning in figures, charts, tables and layout, which plain text extraction drops.

**RQ2. Can Postgres with pgvector be the only retrieval store for a multimodal hybrid RAG?**
No separate vector database (Qdrant, Milvus, Chroma) and no separate search engine (Elasticsearch, OpenSearch).

## RQ1: VLM encoding of large documents

Each document is ingested three ways and the same queries are run against each:

| Method | What becomes a chunk | `ingestion_jobs.parser` |
|---|---|---|
| Text-only (baseline) | Extracted text; images are skipped | `fast` |
| Layout-aware parsing | Text plus tables as Markdown, with page and heading structure | `docling` |
| VLM encoding | The above, plus a VLM caption for every figure, chart and slide (`modality = image`) | `docling` + VLM |

Questions within RQ1:

- **Quality:** do VLM captions raise recall@k and answer correctness, especially for questions whose answer is only in a figure or table?
- **Length:** does retrieval quality drop as documents get longer (10 → 100 → 500+ pages), and does VLM encoding slow that drop?
- **Cost:** how much ingestion time and memory does the VLM add per page, on CUDA, MPS and MLX, with and without quantization?
- **Structure:** do page numbers, heading paths and parent chunks (`page`, `heading_path`, `parent_id`) help ranking on long documents?

## RQ2: pgvector as the only store

Documents, chunks, vectors, full-text and metadata all live in one database and can be searched in one SQL query. What is being tried, and what each part is expected to show:

| Technique | pgvector / Postgres feature | Question |
|---|---|---|
| Half-precision dense vectors | `halfvec(1024)` | Is recall the same as `vector` at half the storage? |
| Binary-quantized HNSW index | `binary_quantize()` + `bit_hamming_ops` | How much recall is lost, and does re-ranking with the full vector win it back? |
| Sparse (SPLADE-style) search | `sparsevec(30522)` + HNSW `sparsevec_ip_ops` | Is it usable within the 1,000 non-zero limit, and does it beat full-text search? |
| Full-text search | `tsvector` + GIN index | How much does keyword search still add to hybrid results? |
| Hybrid fusion | RRF in a single SQL query | Is fusing in Postgres fast enough, compared to fusing in Python? |
| Metadata filters | `JSONB` + GIN, combined with vector search | Do filtered vector searches still return enough results? |

What to measure for each: recall@k against exact (no-index) search, query latency, index build time and size, and storage per chunk.

## Evaluation

| Metric | Measures | Used for |
|---|---|---|
| Recall@k, MRR, nDCG@10 | Whether the right chunks are retrieved, and how high | RQ1, RQ2 |
| Answer correctness | Whether the generated answer is right, judged against a reference | RQ1 |
| Ingestion time and peak memory per page | Cost of each encoding method | RQ1 |
| Query latency (p50 / p95), index size, build time | Cost of each retrieval setup | RQ2 |

The test set is a collection of long PDFs and slide decks, with questions labeled by where the answer lives: body text, table, or figure. That split is what shows whether VLM encoding helps where text-only parsing can't.

## Results

None yet. Results, the test set and the scripts to reproduce them will be added here.

## How retrieval works

### Dense search

Each chunk and each query becomes a normalized vector. Similarity is the cosine
between them; pgvector's `<=>` operator returns the cosine **distance**:

```math
\text{dist}(q, d) = 1 - \frac{q \cdot d}{\lVert q \rVert \, \lVert d \rVert}
```

### Binary-quantized index

Searching full vectors is expensive, so the HNSW index stores 1 bit per dimension:

```math
b_i = \begin{cases} 1 & \text{if } x_i > 0 \\ 0 & \text{otherwise} \end{cases}
```

Two binary vectors are compared with the Hamming distance, the number of differing bits:

```math
H(b_q, b_d) = \text{popcount}(b_q \oplus b_d)
```

A 1024-dim vector shrinks from 2 KB (`halfvec`) to 128 bytes. Search runs in two stages:
the binary index returns the top $N$ candidates by $H$, then those are re-ranked by the
exact cosine distance.

### Sparse search

A sparse vector has one weight per vocabulary token, almost all zero. The score is a dot
product over the tokens both vectors share:

```math
s(q, d) = \sum_{i \,\in\, \text{nz}(q) \,\cap\, \text{nz}(d)} q_i \, d_i
```

### Fusion (Reciprocal Rank Fusion)

Dense, sparse and full-text search each produce a ranking. Their scores aren't comparable,
so they are combined by rank instead of score:

```math
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}, \qquad k = 60
```

A document ranked high in several lists beats one ranked first in only one.

### Reranking

The top fused results are re-scored by a cross-encoder, which reads the query and chunk
together instead of comparing two precomputed vectors:

```math
\text{score}(q, d) = f_\theta([\,q \,;\, d\,])
```

It's slower but more accurate, so it only runs on the top candidates.

## Todo

- [x] App skeleton: config, logging, device selection, startup checks, middleware
- [x] Database schema and first migration
- [x] MinIO storage helpers
- [x] Model providers: LLM, VLM, embedders, reranker (PyTorch + MLX)
- [ ] Ingestion pipeline (`app/rag/ingestion/`)
- [ ] Retrieval pipeline (`app/rag/retrieval/`)
- [ ] Generation pipeline (`app/rag/generation/`)
- [ ] API routes (`app/api/`)
- [ ] Test set: long PDFs and slide decks, with questions labeled text / table / figure
- [ ] RQ1 evaluation: text-only vs layout-aware vs VLM encoding
- [ ] RQ2 benchmarks: recall, latency, index size for each pgvector setup

## Stack

| Part | Tool |
|---|---|
| API | FastAPI + Uvicorn |
| Database | Postgres 18 + pgvector (`halfvec` dense, `sparsevec` sparse, `tsvector` full-text) |
| File storage | MinIO (S3-compatible) |
| Models | Transformers / sentence-transformers (PyTorch), mlx-lm / mlx-vlm (Apple Silicon) |
| Config | `.env` read by `app/core/config.py` |

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
# 1. Install dependencies
uv sync

# 2. Create your .env and set passwords
cp .env.example .env

# 3. Start Postgres and MinIO
docker compose up -d

# 4. Create the tables (run once)
docker exec -i pg-vector psql -U postgres -d multi_rag -v ON_ERROR_STOP=1 < migrations/001_schema_init_table.sql

# 5. Run the API
uv run python main.py
```

- API docs: http://localhost:8000/docs
- MinIO console: http://localhost:9001 (log in with `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`)

On startup the log shows the detected device and whether the database and MinIO connections worked.

## Configuration

All settings live in `.env`. See [.env.example](.env.example) for the full list.

| Variable | Purpose |
|---|---|
| `API_PORT` | Port the API listens on |
| `DB_*` | Postgres connection. Also used by `docker-compose.yml` to create the database |
| `MINIO_*` | MinIO connection and bucket. `MINIO_SECRET_KEY` must be at least 8 characters |
| `DEVICE` | `auto`, `cuda`, `mps`, `cpu` or `mlx` (see below) |
| `QUANT_BITS` | Empty for full precision, or `4` / `8` |
| `LLM_MODEL_ID`, `VLM_MODEL_ID`, `EMBEDDING_MODEL_ID` | Hugging Face model IDs |

### Choosing a device

| `DEVICE` | Runs on | Use it for |
|---|---|---|
| `auto` | CUDA if present, else Apple GPU (MPS), else CPU | Most setups |
| `cuda` | NVIDIA GPU with PyTorch | Linux / Windows GPU servers |
| `mps` | Apple Silicon GPU with PyTorch | Macs |
| `mlx` | Apple Silicon with MLX | Macs; usually the fastest there. Use `mlx-community/...` model IDs |
| `cpu` | CPU with PyTorch | Debugging, CI |

If you ask for a device the machine doesn't have, the app stops with an error instead of quietly falling back to CPU.

`QUANT_BITS` needs an extra package: `uv add bitsandbytes` for CUDA, or `uv add optimum-quanto` for MPS / CPU. With MLX, pick a pre-quantized model instead.

## Project layout

```
main.py                 FastAPI app, startup checks, middleware
app/
  core/                 config, device selection, logging
  db/                   SQLAlchemy models and session
  schemas/              Pydantic In / Out schemas per table
  providers/            LLM, VLM, embedders and reranker (PyTorch + MLX)
  middleware/           access log and JSON 500 handler
  utils/                MinIO storage helpers
  rag/                  ingestion, retrieval, generation (to do)
  api/                  routes (to do)
migrations/             SQL migrations, applied in order
docker-compose.yml      Postgres (pgvector) and MinIO
```

## Database

Three tables, defined in [app/db/model.py](app/db/model.py) and created by [migrations/001_schema_init_table.sql](migrations/001_schema_init_table.sql):

- **documents**: one row per uploaded file, with its processing status. The file itself is stored in MinIO.
- **chunks**: pieces of a document (text, table or image caption) with their dense embedding, optional sparse embedding and full-text index.
- **ingestion_jobs**: the processing queue. A worker picks the oldest `queued` job.

When you change `model.py`, add a new numbered file in `migrations/` with the same change.

Changing `EMBEDDING_DIM` (1024) or `SPARSE_DIM` (30522) in `model.py` needs a migration and re-embedding every chunk.
