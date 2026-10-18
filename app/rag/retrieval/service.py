import time

from PIL.Image import Image
from sqlalchemy.orm import Session

from app.providers.embedder import load_dense_embedder, load_reranker, load_sparse_embedder
from app.rag.retrieval.queries import dense_search, hybrid_search
from app.schemas.search_schema import SearchFilters, SearchHit, SearchIn, SearchOut
from app.utils import minio_storage as storage


def search(session: Session, request: SearchIn) -> SearchOut:
    started = time.perf_counter()

    embedder     = load_dense_embedder()
    query_vector = embedder.encode_query(request.query)
    # With reranking, fetch a larger first stage and let the cross-encoder pick the top_k
    first_stage  = max(request.top_k, request.rerank_top) if request.rerank else request.top_k

    if request.mode == "dense":
        rows = dense_search(session, query_vector, request.filters, first_stage, request.candidates)
    else:
        sparse       = load_sparse_embedder()
        query_sparse = sparse.encode_query(request.query) if sparse else None
        rows = hybrid_search(session, request.query, query_vector, query_sparse, request.filters, first_stage, request.candidates)

    hits = [_to_hit(row) for row in rows]
    if request.rerank and hits:
        hits = rerank(request.query, hits, request.top_k)

    return SearchOut(
        query    = request.query,
        mode     = request.mode,
        reranked = request.rerank,
        took_ms  = round((time.perf_counter() - started) * 1000, 2),
        hits     = hits,
    )


def search_by_image(
    session    : Session,
    image      : Image,
    text       : str | None,
    filters    : SearchFilters,
    top_k      : int,
    candidates : int,
) -> SearchOut:
    """Dense only: there is no full-text or sparse side to an image, and the reranker reads text only."""
    started = time.perf_counter()

    query_vector = load_dense_embedder().encode_image_query(image, text)
    rows         = dense_search(session, query_vector, filters, top_k, candidates)

    return SearchOut(
        query    = text or "",
        mode     = "dense",
        reranked = False,
        took_ms  = round((time.perf_counter() - started) * 1000, 2),
        hits     = [_to_hit(row) for row in rows],
    )


def rerank(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    """Cross-encoder reads query and chunk together; score becomes the reranker's, the first-stage ranks are kept."""
    ranked = load_reranker().rank(query, [hit.content for hit in hits], top_k=top_k)
    return [
        hits[r["corpus_id"]].model_copy(update={"score": float(r["score"]), "rerank_score": float(r["score"])})
        for r in ranked
    ]


def _to_hit(row: dict) -> SearchHit:
    image_path = row.pop("image_path")
    return SearchHit(**row, image_url=storage.presigned_url(image_path) if image_path else None)
