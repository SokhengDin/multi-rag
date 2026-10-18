import uuid

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.model import ChunkModel, DocumentModel, DocumentStatus, Modality
from app.db.session import get_session
from app.rag.ingestion.service import delete_document, queue_job, submit_document
from app.schemas.base_schema import RESPONSE_SCHEMA
from app.schemas.chunk_schema import ChunkOut
from app.schemas.document_schema import DocumentOut
from app.schemas.ingestion_job_schema import IngestionJobOut, Parser
from app.utils import minio_storage as storage

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", status_code=201, response_model=RESPONSE_SCHEMA[IngestionJobOut])
def upload_document(
    file    : UploadFile = File(...),
    parser  : Parser     = Form("fast"),
    session : Session    = Depends(get_session),
):
    """Store the file and queue it for ingestion. The same file uploaded again reuses the stored document."""
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        job = submit_document(session, file.filename or "", data, parser)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return RESPONSE_SCHEMA(status=201, message="Document queued for ingestion", data=IngestionJobOut.model_validate(job))


@router.get("", response_model=RESPONSE_SCHEMA[list[DocumentOut]])
def list_documents(
    status  : DocumentStatus | None = None,
    limit   : int                   = Query(50, ge=1, le=200),
    offset  : int                   = Query(0, ge=0),
    session : Session               = Depends(get_session),
):
    stmt = select(DocumentModel).order_by(DocumentModel.created_at.desc()).limit(limit).offset(offset)
    if status:
        stmt = stmt.where(DocumentModel.status == status)

    documents = [DocumentOut.model_validate(d) for d in session.scalars(stmt)]
    return RESPONSE_SCHEMA(status=200, message="OK", data=documents)


@router.get("/{document_id}", response_model=RESPONSE_SCHEMA[DocumentOut])
def get_document(document_id: uuid.UUID, session: Session = Depends(get_session)):
    document = _get_or_404(session, document_id)
    return RESPONSE_SCHEMA(status=200, message="OK", data=DocumentOut.model_validate(document))


@router.get("/{document_id}/file", response_class=RedirectResponse)
def open_document_file(document_id: uuid.UUID, session: Session = Depends(get_session)):
    """Redirects to the stored file. Add #page=N to open a PDF at that page."""
    document = _get_or_404(session, document_id)
    return RedirectResponse(storage.presigned_url(document.storage_path))


@router.delete("/{document_id}", response_model=RESPONSE_SCHEMA[None])
def remove_document(document_id: uuid.UUID, session: Session = Depends(get_session)):
    delete_document(session, _get_or_404(session, document_id))
    return RESPONSE_SCHEMA(status=200, message="Document deleted")


@router.post("/{document_id}/ingest", status_code=201, response_model=RESPONSE_SCHEMA[IngestionJobOut])
def reingest_document(
    document_id : uuid.UUID,
    parser      : Parser  = Body("fast", embed=True),
    session     : Session = Depends(get_session),
):
    """Queue the document again, e.g. with another parser. Its chunks are replaced when the job runs."""
    job = queue_job(session, _get_or_404(session, document_id), parser)
    return RESPONSE_SCHEMA(status=201, message="Document queued for ingestion", data=IngestionJobOut.model_validate(job))


@router.get("/{document_id}/chunks", response_model=RESPONSE_SCHEMA[list[ChunkOut]])
def list_chunks(
    document_id : uuid.UUID,
    modality    : Modality | None = None,
    limit       : int             = Query(100, ge=1, le=500),
    offset      : int             = Query(0, ge=0),
    session     : Session         = Depends(get_session),
):
    _get_or_404(session, document_id)

    stmt = (
        select(ChunkModel)
        .where(ChunkModel.document_id == document_id)
        .order_by(ChunkModel.chunk_index)
        .limit(limit)
        .offset(offset)
    )
    if modality:
        stmt = stmt.where(ChunkModel.modality == modality)

    chunks = [ChunkOut.model_validate(c) for c in session.scalars(stmt)]
    return RESPONSE_SCHEMA(status=200, message="OK", data=chunks)


def _get_or_404(session: Session, document_id: uuid.UUID) -> DocumentModel:
    document = session.get(DocumentModel, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return document
