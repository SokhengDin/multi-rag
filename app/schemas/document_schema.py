import uuid
from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.db.model import DocumentStatus, SourceType

# The ORM attribute is metadata_ ("metadata" is taken by SQLAlchemy); the API field is "metadata"
METADATA_FIELD = Field(default_factory=dict, validation_alias=AliasChoices("metadata_", "metadata"))


class DocumentIn(BaseModel):
    filename     : str
    mime_type    : str
    source_type  : SourceType
    size_bytes   : int
    sha256       : str = Field(min_length=64, max_length=64)
    storage_path : str
    metadata     : dict[str, Any] = METADATA_FIELD


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id           : uuid.UUID
    filename     : str
    mime_type    : str
    source_type  : SourceType
    size_bytes   : int
    sha256       : str
    storage_path : str
    status       : DocumentStatus
    error        : str | None
    num_chunks   : int
    metadata     : dict[str, Any] = METADATA_FIELD
    created_at   : datetime
    updated_at   : datetime
