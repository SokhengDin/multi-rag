import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.model import Modality
from app.schemas.ingestion_job_schema import Parser


def chunk_image_url(document_id: uuid.UUID, chunk_id: uuid.UUID) -> str:
    """API path that streams the chunk's image, relative to the API base URL."""
    return f"/documents/{document_id}/chunks/{chunk_id}/image"


class ChunkOut(BaseModel):
    """Embedding and tsvector are left out — too large and not useful to API clients."""
    model_config = ConfigDict(from_attributes=True)

    id           : uuid.UUID
    document_id  : uuid.UUID
    job_id       : uuid.UUID
    parser       : Parser
    chunk_index  : int
    modality     : Modality
    content      : str
    token_count  : int | None
    page         : int | None
    heading_path : list[str] | None
    json_path    : str | None
    boxes        : list[list[float]] | None
    image_url    : str | None = None
    created_at   : datetime
