import json
import uuid
from urllib.parse import quote

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from minio.error import S3Error
from sqlalchemy import select
from sqlalchemy.orm import Session, contains_eager

from app.db.model import ChunkModel, DocumentModel, IngestionJobModel, Modality, SourceType
from app.db.session import get_session
from app.rag.ingestion.service import delete_document, describe_documents, queue_job, render_page, submit_document
from app.schemas.base_schema import RESPONSE_SCHEMA
from app.schemas.chunk_schema import ChunkOut, chunk_image_url
from app.schemas.document_schema import DocumentOut
from app.schemas.ingestion_job_schema import IngestionJobOut, Parser
from app.utils import minio_storage as storage

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", status_code=201, response_model=RESPONSE_SCHEMA[IngestionJobOut])
def upload_document(
    file     : UploadFile = File(...),
    parser   : Parser     = Form("fast"),
    metadata : str | None = Form(None, description='JSON object stored on the document and usable as a search filter, e.g. {"project": "churn"}'),
    session  : Session    = Depends(get_session),
):
    """Store the file and queue it for ingestion. The same file uploaded again reuses the stored document."""
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        values = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"metadata is not valid JSON: {e}")
    if not isinstance(values, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")

    try:
        job = submit_document(session, file.filename or "", data, parser, values)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return RESPONSE_SCHEMA(status=201, message="Document queued for ingestion", data=IngestionJobOut.model_validate(job))


@router.get("", response_model=RESPONSE_SCHEMA[list[DocumentOut]])
def list_documents(
    limit   : int     = Query(50, ge=1, le=200),
    offset  : int     = Query(0, ge=0),
    session : Session = Depends(get_session),
):
    documents = session.scalars(select(DocumentModel).order_by(DocumentModel.created_at.desc()).limit(limit).offset(offset)).all()
    return RESPONSE_SCHEMA(status=200, message="OK", data=describe_documents(session, list(documents)))


@router.get("/{document_id}", response_model=RESPONSE_SCHEMA[DocumentOut])
def get_document(document_id: uuid.UUID, session: Session = Depends(get_session)):
    document = _get_or_404(session, document_id)
    return RESPONSE_SCHEMA(status=200, message="OK", data=describe_documents(session, [document])[0])


@router.get("/{document_id}/file", response_class=StreamingResponse)
def open_document_file(document_id: uuid.UUID, session: Session = Depends(get_session)):
    """The original file, shown inline. Add #page=N to open a PDF at that page."""
    document = _get_or_404(session, document_id)
    return _stream(document.storage_path, document.filename)


@router.get("/{document_id}/chunks/{chunk_id}/image", response_class=StreamingResponse)
def get_chunk_image(document_id: uuid.UUID, chunk_id: uuid.UUID, session: Session = Depends(get_session)):
    """The figure or page image of an image chunk."""
    chunk = session.get(ChunkModel, chunk_id)
    if chunk is None or chunk.document_id != document_id or not chunk.image_path:
        raise HTTPException(status_code=404, detail=f"No image for chunk {chunk_id}")
    return _stream(chunk.image_path)


@router.get("/{document_id}/pages/{page}", response_class=Response)
def get_page_image(
    document_id : uuid.UUID,
    page        : int,
    dpi         : int     = Query(110, ge=36, le=300),
    session     : Session = Depends(get_session),
):
    """A PDF page as PNG, to draw search-hit boxes on."""
    document = _get_or_404(session, document_id)
    if document.source_type != SourceType.PDF:
        raise HTTPException(status_code=400, detail="Page images are only available for PDFs")

    try:
        png = render_page(document, page, dpi)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


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
    """Queue the document again. When the job finishes it replaces the chunks of the same parser; other parsers' chunks stay."""
    job = queue_job(session, _get_or_404(session, document_id), parser)
    return RESPONSE_SCHEMA(status=201, message="Document queued for ingestion", data=IngestionJobOut.model_validate(job))


@router.get("/{document_id}/chunks", response_model=RESPONSE_SCHEMA[list[ChunkOut]])
def list_chunks(
    document_id : uuid.UUID,
    parser      : Parser | None   = None,
    modality    : Modality | None = None,
    limit       : int             = Query(100, ge=1, le=500),
    offset      : int             = Query(0, ge=0),
    session     : Session         = Depends(get_session),
):
    _get_or_404(session, document_id)

    stmt = (
        select(ChunkModel)
        .join(IngestionJobModel, IngestionJobModel.id == ChunkModel.job_id)
        .options(contains_eager(ChunkModel.job))
        .where(ChunkModel.document_id == document_id)
        .order_by(IngestionJobModel.parser, ChunkModel.chunk_index)
        .limit(limit)
        .offset(offset)
    )
    if parser:
        stmt = stmt.where(IngestionJobModel.parser == parser)
    if modality:
        stmt = stmt.where(ChunkModel.modality == modality)

    chunks = [
        ChunkOut.model_validate(c).model_copy(update={"image_url": chunk_image_url(c.document_id, c.id) if c.image_path else None})
        for c in session.scalars(stmt)
    ]
    return RESPONSE_SCHEMA(status=200, message="OK", data=chunks)


def _get_or_404(session: Session, document_id: uuid.UUID) -> DocumentModel:
    document = session.get(DocumentModel, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return document


def _stream(object_name: str, filename: str | None = None) -> StreamingResponse:
    try:
        chunks, content_type, size = storage.stream_object(object_name)
    except S3Error as e:
        if e.code == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Stored file not found")
        raise

    headers = {"Content-Length": str(size), "Cache-Control": "private, max-age=86400"}
    if filename:
        headers["Content-Disposition"] = f"inline; filename*=UTF-8''{quote(filename)}"
    return StreamingResponse(chunks, media_type=content_type, headers=headers)
