import io
import uuid

from PIL.Image import Image
from pgvector import SparseVector
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app import logger
from app.core.config import settings
from app.db.model import SPARSE_DIM, ChunkModel, IngestionJobModel, Modality
from app.providers.embedder import load_dense_embedder, load_sparse_embedder
from app.providers.vlm import LocalVLM, MLXVLM, load_vlm
from app.rag.ingestion.captioner import describe_image
from app.rag.ingestion.chunker import Chunk, chunk_elements, count_tokens
from app.rag.ingestion.parsers import parse
from app.utils import minio_storage as storage


class IngestionPipeline:
    """Download -> parse -> chunk -> caption (vlm only) -> embed -> replace this parser's chunks of the document."""

    def __init__(self) -> None:
        self.embedder = load_dense_embedder()
        self.sparse   = load_sparse_embedder()
        self._vlm     : LocalVLM | MLXVLM | None = None

    @property
    def vlm(self) -> LocalVLM | MLXVLM:
        if self._vlm is None:
            logger.info(f"Loading VLM {settings.VLM_MODEL_ID}")
            self._vlm = load_vlm()
        return self._vlm

    def run(self, session: Session, job: IngestionJobModel) -> list[uuid.UUID]:
        """Adds the job's chunks and deletes the previous ones of the same parser, uncommitted.
        Returns the replaced jobs, whose images remove_images() deletes after the commit."""
        document  = job.document
        tokenizer = self.embedder.tokenizer

        data     = storage.download_bytes(document.storage_path)
        elements = parse(data, document.filename, document.source_type, job.parser)
        chunks   = chunk_elements(elements, tokenizer, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        images = [c for c in chunks if c.modality == Modality.IMAGE] if job.parser == "vlm" else []
        for number, chunk in enumerate(images, start=1):
            logger.info(f"Captioning image {number}/{len(images)} of {document.filename}")
            chunk.content     = describe_image(self.vlm, chunk.image, chunk.content)
            chunk.token_count = count_tokens(tokenizer, chunk.content)

        # An image with no text (e.g. a page that is only a figure) is still embedded from the image
        chunks = [c for c in chunks if c.content.strip() or c.image is not None]
        if not chunks:
            raise ValueError(f"No content extracted from {document.filename}")

        texts  = [c.content for c in chunks]
        dense  = self.embedder.encode_documents([self.embedder.document_input(c.content, _title(c), c.image) for c in chunks])
        sparse = self.sparse.encode_documents(texts) if self.sparse else [None] * len(chunks)

        replaced = list(session.scalars(
            select(IngestionJobModel.id).where(
                IngestionJobModel.document_id == document.id,
                IngestionJobModel.parser      == job.parser,
                IngestionJobModel.id          != job.id,
            )
        ))
        session.execute(delete(ChunkModel).where(ChunkModel.job_id.in_(replaced)))
        session.add_all(
            self._to_row(job, index, chunk, vector, sparse_vector)
            for index, (chunk, vector, sparse_vector) in enumerate(zip(chunks, dense, sparse))
        )

        job.num_chunks             = len(chunks)
        job.embedding_model        = self.embedder.model_id
        job.sparse_embedding_model = self.sparse.model_id if self.sparse else None

        return replaced

    @staticmethod
    def remove_images(document_id: uuid.UUID, job_ids: list[uuid.UUID]) -> None:
        for job_id in job_ids:
            storage.delete_prefix(_image_prefix(document_id, job_id))

    def _to_row(
        self,
        job           : IngestionJobModel,
        index         : int,
        chunk         : Chunk,
        vector        : list[float],
        sparse_vector : dict[int, float] | None,
    ) -> ChunkModel:
        return ChunkModel(
            document_id  = job.document_id,
            job_id       = job.id,
            chunk_index  = index,
            modality     = chunk.modality,
            content      = chunk.content,
            image_path   = _upload_image(job, index, chunk.image) if chunk.image else None,
            token_count  = chunk.token_count,
            page         = chunk.page,
            heading_path = chunk.heading_path,
            json_path    = chunk.json_path,
            boxes        = chunk.boxes,

            embedding        = vector,
            sparse_embedding = SparseVector(sparse_vector, SPARSE_DIM) if sparse_vector else None,
        )


def _image_prefix(document_id: uuid.UUID, job_id: uuid.UUID) -> str:
    return f"documents/{document_id}/jobs/{job_id}/"


def _upload_image(job: IngestionJobModel, index: int, image: Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return storage.upload_bytes(f"{_image_prefix(job.document_id, job.id)}{index}.png", buffer.getvalue(), "image/png")


def _title(chunk: Chunk) -> str | None:
    return " > ".join(chunk.heading_path) if chunk.heading_path else None
