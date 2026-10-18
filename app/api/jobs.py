import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.model import IngestionJobModel, JobStatus
from app.db.session import get_session
from app.schemas.base_schema import RESPONSE_SCHEMA
from app.schemas.ingestion_job_schema import IngestionJobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=RESPONSE_SCHEMA[list[IngestionJobOut]])
def list_jobs(
    status      : JobStatus | None = None,
    document_id : uuid.UUID | None = None,
    limit       : int              = Query(50, ge=1, le=200),
    session     : Session          = Depends(get_session),
):
    stmt = select(IngestionJobModel).order_by(IngestionJobModel.created_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(IngestionJobModel.status == status)
    if document_id:
        stmt = stmt.where(IngestionJobModel.document_id == document_id)

    jobs = [IngestionJobOut.model_validate(j) for j in session.scalars(stmt)]
    return RESPONSE_SCHEMA(status=200, message="OK", data=jobs)


@router.get("/{job_id}", response_model=RESPONSE_SCHEMA[IngestionJobOut])
def get_job(job_id: uuid.UUID, session: Session = Depends(get_session)):
    job = session.get(IngestionJobModel, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return RESPONSE_SCHEMA(status=200, message="OK", data=IngestionJobOut.model_validate(job))
