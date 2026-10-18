import hashlib
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.model import DocumentModel, IngestionJobModel, SourceType
from app.schemas.ingestion_job_schema import Parser
from app.utils import minio_storage as storage

EXTENSIONS = {
    ".pdf"      : SourceType.PDF,
    ".pptx"     : SourceType.PPTX,
    ".md"       : SourceType.MARKDOWN,
    ".markdown" : SourceType.MARKDOWN,
    ".txt"      : SourceType.TEXT,
    ".json"     : SourceType.JSON,
}


def source_type_of(filename: str) -> SourceType:
    extension = Path(filename).suffix.lower()
    if extension not in EXTENSIONS:
        raise ValueError(f"Unsupported file type {extension!r}, use one of: {', '.join(EXTENSIONS)}")
    return EXTENSIONS[extension]


def submit_document(
    session  : Session,
    filename : str,
    data     : bytes,
    parser   : Parser = "fast",
    metadata : dict[str, Any] | None = None,
) -> IngestionJobModel:
    """Store the file and queue a job. A file already uploaded (same sha256) is reused, so it can be re-ingested with another parser."""
    filename    = Path(filename).name
    source_type = source_type_of(filename)
    sha256      = hashlib.sha256(data).hexdigest()

    document = session.scalars(select(DocumentModel).where(DocumentModel.sha256 == sha256)).first()

    if document is None:
        document_id  = uuid.uuid4()
        mime_type    = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        storage_path = storage.upload_bytes(f"documents/{document_id}/{filename}", data, mime_type)

        document = DocumentModel(
            id           = document_id,
            filename     = filename,
            mime_type    = mime_type,
            source_type  = source_type,
            size_bytes   = len(data),
            sha256       = sha256,
            storage_path = storage_path,
            metadata_    = metadata or {},
        )
        session.add(document)

    return queue_job(session, document, parser)


def queue_job(session: Session, document: DocumentModel, parser: Parser) -> IngestionJobModel:
    job = IngestionJobModel(document=document, parser=parser)
    session.add(job)
    session.commit()
    session.refresh(job)

    return job


def delete_document(session: Session, document: DocumentModel) -> None:
    """Chunks and jobs go with the row (ON DELETE CASCADE); the file and its images go from MinIO."""
    storage.delete_prefix(f"documents/{document.id}/")
    session.delete(document)
    session.commit()
