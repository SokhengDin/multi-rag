import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import logger
from app.db.model import DocumentStatus, IngestionJobModel, JobStatus
from app.db.session import SessionLocal
from app.rag.ingestion.pipeline import IngestionPipeline

POLL_SECONDS = 5


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

    job.status          = JobStatus.RUNNING
    job.attempts       += 1
    job.error           = None
    job.started_at      = datetime.now(timezone.utc)
    job.document.status = DocumentStatus.PROCESSING
    session.commit()

    return job


def process_job(pipeline: IngestionPipeline, session: Session, job: IngestionJobModel) -> None:
    started = time.perf_counter()

    try:
        num_chunks = pipeline.run(session, job)
        job.status           = JobStatus.DONE
        job.finished_at      = datetime.now(timezone.utc)
        job.document.status  = DocumentStatus.READY
        job.document.error   = None
        session.commit()
        logger.info(f"Ingested {job.document.filename} ({job.parser}): {num_chunks} chunks in {time.perf_counter() - started:.1f}s")
    except Exception as e:
        session.rollback()
        logger.exception(f"Ingestion failed for job {job.id}")
        job.status           = JobStatus.FAILED
        job.error            = str(e)
        job.finished_at      = datetime.now(timezone.utc)
        job.document.status  = DocumentStatus.FAILED
        job.document.error   = str(e)
        session.commit()


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
