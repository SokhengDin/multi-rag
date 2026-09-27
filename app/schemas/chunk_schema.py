import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.db.model import EMBEDDING_DIM, Modality
from app.schemas.document_schema import METADATA_FIELD


class ChunkIn(BaseModel):
    document_id     : uuid.UUID
    parent_id       : uuid.UUID | None = None
    chunk_index     : int
    modality        : Modality
    content         : str
    image_path      : str | None = None
    token_count     : int | None = None
    page            : int | None = None
    heading_path    : list[str] | None = None
    json_path       : str | None = None
    embedding       : list[float] = Field(min_length=EMBEDDING_DIM, max_length=EMBEDDING_DIM)
    embedding_model : str
    metadata        : dict[str, Any] = METADATA_FIELD


class ChunkOut(BaseModel):
    """Embedding and tsvector are left out — too large and not useful to API clients."""
    model_config = ConfigDict(from_attributes=True)

    id              : uuid.UUID
    document_id     : uuid.UUID
    parent_id       : uuid.UUID | None
    chunk_index     : int
    modality        : Modality
    content         : str
    image_path      : str | None
    token_count     : int | None
    page            : int | None
    heading_path    : list[str] | None
    json_path       : str | None
    embedding_model : str
    metadata        : dict[str, Any] = METADATA_FIELD
    created_at      : datetime
