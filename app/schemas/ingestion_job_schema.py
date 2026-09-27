import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.db.model import JobStatus

Parser = Literal["fast", "docling"]


class IngestionJobIn(BaseModel):
    document_id : uuid.UUID
    parser      : Parser = "fast"


class IngestionJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id          : uuid.UUID
    document_id : uuid.UUID
    status      : JobStatus
    parser      : Parser
    attempts    : int
    error       : str | None
    started_at  : datetime | None
    finished_at : datetime | None
    created_at  : datetime
    updated_at  : datetime
