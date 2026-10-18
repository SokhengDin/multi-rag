import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.db.model import Modality
from app.schemas.ingestion_job_schema import Parser

# dense: vector search only | hybrid: dense + full-text (+ sparse if configured), fused with RRF
SearchMode = Literal["dense", "hybrid"]


class SearchFilters(BaseModel):
    document_ids : list[uuid.UUID] | None = None
    modalities   : list[Modality] | None  = None
    parser       : Parser | None          = None
    metadata     : dict[str, Any] | None  = Field(default=None, description="Chunks whose metadata contains these keys and values")


class SearchIn(BaseModel):
    query      : str           = Field(min_length=1, max_length=2000)
    mode       : SearchMode    = "hybrid"
    top_k      : int           = Field(default=10, ge=1, le=100)
    candidates : int           = Field(default=100, ge=1, le=1000, description="Results taken from each index before fusion")
    rerank     : bool          = False
    rerank_top : int           = Field(default=30, ge=1, le=200, description="Results passed to the reranker")
    filters    : SearchFilters = Field(default_factory=SearchFilters)


class SearchHit(BaseModel):
    chunk_id      : uuid.UUID
    document_id   : uuid.UUID
    filename      : str
    chunk_index   : int
    modality      : Modality
    content       : str
    page          : int | None
    heading_path  : list[str] | None
    json_path     : str | None
    parser        : str | None = None
    image_url     : str | None = None
    score         : float
    dense_rank    : int | None = None
    fulltext_rank : int | None = None
    sparse_rank   : int | None = None
    rerank_score  : float | None = None


class SearchOut(BaseModel):
    query    : str
    mode     : SearchMode
    reranked : bool
    took_ms  : float
    hits     : list[SearchHit]
