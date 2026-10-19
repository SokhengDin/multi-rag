import uuid
from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.db.model import SourceType
from app.schemas.ingestion_job_schema import IngestionSummary

# The ORM attribute is metadata_ ("metadata" is taken by SQLAlchemy); the API field is "metadata"
METADATA_FIELD = Field(default_factory=dict, validation_alias=AliasChoices("metadata_", "metadata"))


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id           : uuid.UUID
    filename     : str
    mime_type    : str
    source_type  : SourceType
    size_bytes   : int
    sha256       : str
    metadata     : dict[str, Any] = METADATA_FIELD
    ingestions   : list[IngestionSummary] = []
    created_at   : datetime
