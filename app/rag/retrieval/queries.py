import json
from typing import Any

from pgvector import SparseVector
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.model import EMBEDDING_DIM, SPARSE_DIM
from app.schemas.search_schema import SearchFilters

QUERY_VECTOR  = f"CAST(:query_vector AS halfvec({EMBEDDING_DIM}))"
QUERY_SPARSE  = f"CAST(:query_sparse AS sparsevec({SPARSE_DIM}))"
# OR of the query words; plainto_tsquery alone ANDs them, which rarely matches a whole question
QUERY_TSQUERY = "replace(plainto_tsquery('simple', :query_text)::text, '&', '|')::tsquery"

# Must match the index expression in the migration, or the HNSW index is not used
BINARY_ORDER = f"binary_quantize(c.embedding)::bit({EMBEDDING_DIM}) <~> binary_quantize({QUERY_VECTOR})::bit({EMBEDDING_DIM})"

HIT_COLUMNS = """
    c.id AS chunk_id, c.document_id, d.filename, c.chunk_index, c.modality,
    c.content, c.page, c.heading_path, c.json_path, c.image_path, c.metadata->>'parser' AS parser
"""


def dense_search(
    session      : Session,
    query_vector : list[float],
    filters      : SearchFilters,
    top_k        : int,
    candidates   : int,
) -> list[dict]:
    """Binary-quantized HNSW finds candidates by Hamming distance, then the full halfvec re-ranks them by cosine."""
    where, params = _where(filters)
    _tune_hnsw(session, candidates)

    sql = f"""
        WITH dense AS (
            {_dense_ranked(where)}
        )
        SELECT {HIT_COLUMNS}, 1 - dense.distance AS score, dense.rank AS dense_rank
        FROM dense
        JOIN chunks    c ON c.id = dense.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY dense.rank
        LIMIT :top_k
    """
    params |= {"query_vector": str(query_vector), "candidates": candidates, "top_k": top_k}

    return [dict(row) for row in session.execute(text(sql), params).mappings()]


def hybrid_search(
    session      : Session,
    query_text   : str,
    query_vector : list[float],
    query_sparse : dict[int, float] | None,
    filters      : SearchFilters,
    top_k        : int,
    candidates   : int,
    rrf_k        : int = settings.RRF_K,
) -> list[dict]:
    """Dense, full-text and (when configured) sparse rankings, fused with Reciprocal Rank Fusion in one query."""
    where, params = _where(filters)
    _tune_hnsw(session, candidates)

    rankings = {
        "dense"    : _dense_ranked(where),
        "fulltext" : f"""
            SELECT id, row_number() OVER (ORDER BY score DESC) AS rank
            FROM (
                SELECT c.id, ts_rank_cd(c.content_tsv, {QUERY_TSQUERY}) AS score
                FROM chunks c
                WHERE c.content_tsv @@ {QUERY_TSQUERY} AND {where}
                ORDER BY score DESC
                LIMIT :candidates
            ) matches
        """,
    }
    if query_sparse:
        rankings["sparse"] = f"""
            SELECT id, row_number() OVER (ORDER BY score DESC) AS rank
            FROM (
                SELECT c.id, -(c.sparse_embedding <#> {QUERY_SPARSE}) AS score
                FROM chunks c
                WHERE c.sparse_embedding IS NOT NULL AND {where}
                ORDER BY c.sparse_embedding <#> {QUERY_SPARSE}
                LIMIT :candidates
            ) matches
        """
        params["query_sparse"] = SparseVector(query_sparse, SPARSE_DIM).to_text()

    ctes   = ",\n".join(f"{name} AS ({sql})" for name, sql in rankings.items())
    ranked = "\nUNION ALL\n".join(f"SELECT id, rank, '{name}' AS method FROM {name}" for name in rankings)
    ranks  = ", ".join(
        f"min(rank) FILTER (WHERE method = '{name}') AS {name}_rank" for name in ("dense", "fulltext", "sparse")
    )

    sql = f"""
        WITH {ctes},
        fused AS (
            SELECT id, sum(1.0 / (:rrf_k + rank)) AS score, {ranks}
            FROM ({ranked}) ranked
            GROUP BY id
            ORDER BY score DESC
            LIMIT :top_k
        )
        SELECT {HIT_COLUMNS}, fused.score, fused.dense_rank, fused.fulltext_rank, fused.sparse_rank
        FROM fused
        JOIN chunks    c ON c.id = fused.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY fused.score DESC
    """
    params |= {
        "query_text"   : query_text,
        "query_vector" : str(query_vector),
        "candidates"   : candidates,
        "top_k"        : top_k,
        "rrf_k"        : rrf_k,
    }

    return [dict(row) for row in session.execute(text(sql), params).mappings()]


def _dense_ranked(where: str) -> str:
    return f"""
        SELECT id, distance, row_number() OVER (ORDER BY distance) AS rank
        FROM (
            SELECT c.id, c.embedding <=> {QUERY_VECTOR} AS distance
            FROM chunks c
            WHERE {where}
            ORDER BY {BINARY_ORDER}
            LIMIT :candidates
        ) candidates
    """


def _where(filters: SearchFilters) -> tuple[str, dict[str, Any]]:
    clauses : list[str]      = []
    params  : dict[str, Any] = {}

    if filters.document_ids:
        clauses.append("c.document_id = ANY(CAST(:document_ids AS uuid[]))")
        params["document_ids"] = [str(i) for i in filters.document_ids]

    if filters.modalities:
        clauses.append("c.modality = ANY(CAST(:modalities AS modality[]))")
        params["modalities"] = [m.value for m in filters.modalities]

    metadata = {**(filters.metadata or {}), **({"parser": filters.parser} if filters.parser else {})}
    if metadata:
        clauses.append("c.metadata @> CAST(:metadata AS jsonb)")
        params["metadata"] = json.dumps(metadata)

    return " AND ".join(clauses) or "TRUE", params


def _tune_hnsw(session: Session, candidates: int) -> None:
    """ef_search caps how many rows HNSW returns (default 40). Iterative scan keeps going when filters drop rows."""
    session.execute(text("SELECT set_config('hnsw.ef_search', :ef, true), set_config('hnsw.iterative_scan', 'relaxed_order', true)"),
                    {"ef": str(max(candidates, 40))})
