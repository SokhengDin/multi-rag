import hashlib
import mimetypes
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.model import ChunkModel, DocumentModel, IngestionJobModel, SourceType
from app.schemas.document_schema import DocumentOut
from app.schemas.ingestion_job_schema import IngestionSummary, Parser
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
    """A file already uploaded (same sha256) is reused, so it can be re-ingested with another parser."""
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


def describe_documents(session: Session, documents: list[DocumentModel]) -> list[DocumentOut]:
    """Documents with, per parser, the latest job and the number of chunks searchable now."""
    ids = [d.id for d in documents]

    latest_jobs = session.scalars(
        select(IngestionJobModel)
        .where(IngestionJobModel.document_id.in_(ids))
        .distinct(IngestionJobModel.document_id, IngestionJobModel.parser)
        .order_by(IngestionJobModel.document_id, IngestionJobModel.parser, IngestionJobModel.created_at.desc())
    )
    live_chunks = {
        (document_id, parser): count
        for document_id, parser, count in session.execute(
            select(ChunkModel.document_id, IngestionJobModel.parser, func.count())
            .join(IngestionJobModel, IngestionJobModel.id == ChunkModel.job_id)
            .where(ChunkModel.document_id.in_(ids))
            .group_by(ChunkModel.document_id, IngestionJobModel.parser)
        )
    }

    summaries: dict[uuid.UUID, list[IngestionSummary]] = {}
    for job in latest_jobs:
        summaries.setdefault(job.document_id, []).append(IngestionSummary(
            parser      = job.parser,
            job_id      = job.id,
            status      = job.status,
            error       = job.error,
            num_chunks  = live_chunks.get((job.document_id, job.parser), 0),
            finished_at = job.finished_at,
        ))

    return [
        DocumentOut.model_validate(d).model_copy(update={"ingestions": summaries.get(d.id, [])})
        for d in documents
    ]


@lru_cache(maxsize=8)
def _stored_bytes(storage_path: str) -> bytes:
    return storage.download_bytes(storage_path)


def render_page(document: DocumentModel, page: int, dpi: int) -> bytes:
    """PNG of one PDF page. Raises IndexError if the page doesn't exist."""
    import pymupdf

    with pymupdf.open(stream=_stored_bytes(document.storage_path), filetype="pdf") as pdf:
        if not 1 <= page <= pdf.page_count:
            raise IndexError(f"{document.filename} has {pdf.page_count} pages")
        return pdf[page - 1].get_pixmap(dpi=dpi).tobytes("png")


def delete_document(session: Session, document: DocumentModel) -> None:
    """Chunks and jobs go with the row (ON DELETE CASCADE); the file and its images go from MinIO."""
    storage.delete_prefix(f"documents/{document.id}/")
    session.delete(document)
    session.commit()
