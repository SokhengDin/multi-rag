import time
from datetime import datetime, timezone

from minio.error import S3Error
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from urllib3.exceptions import HTTPError

from app import logger
from app.db.model import IngestionJobModel, JobStatus
from app.db.session import SessionLocal
from app.rag.ingestion.pipeline import IngestionPipeline

POLL_SECONDS        = 5
MAX_ATTEMPTS        = 3
RETRY_DELAY_SECONDS = 10

# Storage or database outages; a bad file fails the same way every time, so it is not retried
RETRYABLE = (S3Error, OperationalError, HTTPError, ConnectionError, TimeoutError)


def claim_job(session: Session) -> IngestionJobModel | None:
    """Take the oldest queued job. SKIP LOCKED lets several workers run side by side."""
    job = session.scalars(
        select(IngestionJobModel)
        .where(IngestionJobModel.status == JobStatus.QUEUED)
        .order_by(IngestionJobModel.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()

    if job is None:
        return None

    job.status      = JobStatus.RUNNING
    job.attempts   += 1
    job.error       = None
    job.started_at  = datetime.now(timezone.utc)
    job.finished_at = None
    session.commit()

    return job


def process_job(pipeline: IngestionPipeline, session: Session, job: IngestionJobModel) -> None:
    started = time.perf_counter()

    try:
        replaced = pipeline.run(session, job)
        job.status      = JobStatus.DONE
        job.finished_at = datetime.now(timezone.utc)
        session.commit()
        logger.info(f"Ingested {job.document.filename} ({job.parser}): {job.num_chunks} chunks in {time.perf_counter() - started:.1f}s")
    except Exception as e:
        session.rollback()
        retry = isinstance(e, RETRYABLE) and job.attempts < MAX_ATTEMPTS
        logger.exception(f"Ingestion {'will retry' if retry else 'failed'} for job {job.id} (attempt {job.attempts}/{MAX_ATTEMPTS})")
        if retry:
            time.sleep(RETRY_DELAY_SECONDS * job.attempts)

        job.status      = JobStatus.QUEUED if retry else JobStatus.FAILED
        job.error       = str(e)
        job.finished_at = None if retry else datetime.now(timezone.utc)
        session.commit()
        return

    try:
        pipeline.remove_images(job.document_id, replaced)
    except Exception:
        logger.exception(f"Could not delete images of replaced jobs {replaced}")


def run_worker(poll_seconds: int = POLL_SECONDS) -> None:
    pipeline = IngestionPipeline()
    logger.info("Ingestion worker started")

    while True:
        try:
            with SessionLocal() as session:
                job = claim_job(session)
                if job is not None:
                    process_job(pipeline, session, job)
                    continue
        except Exception:
            logger.exception("Worker loop error, retrying")

        time.sleep(poll_seconds)


if __name__ == "__main__":
    run_worker()
