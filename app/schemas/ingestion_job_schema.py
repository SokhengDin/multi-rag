import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.db.model import JobStatus

# fast: text only | docling: layout-aware | vlm: docling + VLM captions for figures | page: every page as an image
Parser = Literal["fast", "docling", "vlm", "page"]


class IngestionJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id                     : uuid.UUID
    document_id            : uuid.UUID
    status                 : JobStatus
    parser                 : Parser
    attempts               : int
    error                  : str | None
    num_chunks             : int
    embedding_model        : str | None
    sparse_embedding_model : str | None
    started_at             : datetime | None
    finished_at            : datetime | None
    created_at             : datetime
    updated_at             : datetime


class IngestionSummary(BaseModel):
    """One per parser: the latest job's state, and how many chunks of that parser are searchable now."""
    parser      : Parser
    job_id      : uuid.UUID
    status      : JobStatus
    error       : str | None
    num_chunks  : int
    finished_at : datetime | None
