import io

from PIL.Image import Image
from pgvector import SparseVector
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app import logger
from app.core.config import settings
from app.db.model import SPARSE_DIM, ChunkModel, DocumentModel, IngestionJobModel, Modality
from app.providers.embedder import load_dense_embedder, load_sparse_embedder
from app.providers.vlm import LocalVLM, MLXVLM, load_vlm
from app.rag.ingestion.captioner import describe_image
from app.rag.ingestion.chunker import Chunk, chunk_elements, count_tokens
from app.rag.ingestion.parsers import parse
from app.utils import minio_storage as storage


class IngestionPipeline:
    """Download -> parse -> chunk -> caption (vlm only) -> embed -> replace the document's chunks."""

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

    def run(self, session: Session, job: IngestionJobModel) -> int:
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

        session.execute(delete(ChunkModel).where(ChunkModel.document_id == document.id))
        session.add_all(
            self._to_row(document, job.parser, index, chunk, vector, sparse_vector)
            for index, (chunk, vector, sparse_vector) in enumerate(zip(chunks, dense, sparse))
        )
        document.num_chunks = len(chunks)

        return len(chunks)

    def _to_row(
        self,
        document      : DocumentModel,
        parser        : str,
        index         : int,
        chunk         : Chunk,
        vector        : list[float],
        sparse_vector : dict[int, float] | None,
    ) -> ChunkModel:
        return ChunkModel(
            document_id  = document.id,
            chunk_index  = index,
            modality     = chunk.modality,
            content      = chunk.content,
            image_path   = self._upload_image(document, index, chunk.image) if chunk.image else None,
            token_count  = chunk.token_count,
            page         = chunk.page,
            heading_path = chunk.heading_path,
            json_path    = chunk.json_path,

            embedding       = vector,
            embedding_model = self.embedder.model_id,

            sparse_embedding       = SparseVector(sparse_vector, SPARSE_DIM) if sparse_vector else None,
            sparse_embedding_model = self.sparse.model_id if self.sparse else None,

            metadata_ = {"parser": parser, **({"boxes": chunk.boxes} if chunk.boxes else {})},
        )

    @staticmethod
    def _upload_image(document: DocumentModel, index: int, image: Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return storage.upload_bytes(f"documents/{document.id}/images/{index}.png", buffer.getvalue(), "image/png")


def _title(chunk: Chunk) -> str | None:
    return " > ".join(chunk.heading_path) if chunk.heading_path else None
