import enum
import uuid
from datetime import datetime
from typing   import Any

from pgvector.sqlalchemy import HALFVEC, SPARSEVEC
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import mapped_column, DeclarativeBase, Mapped, relationship

# EmbeddingGemma 2 = 768. Changing it needs a migration and re-embedding :), think carefully
EMBEDDING_DIM = 768
# SPLADE-style sparse vectors use the BERT vocab size.
SPARSE_DIM    = 30522

class Base(DeclarativeBase):
    pass


class SourceType(str, enum.Enum):
    PDF      = "pdf"
    PPTX     = "pptx"
    MARKDOWN = "markdown"
    TEXT     = "text"
    JSON     = "json"


class Modality(str, enum.Enum):
    TEXT  = "text"
    TABLE = "table"
    IMAGE = "image"


class JobStatus(str, enum.Enum):
    QUEUED  = "queued"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


def pg_enum(
    enum_cls: type[enum.Enum],
    name: str
) -> Enum:
    return Enum(enum_cls, name=name, values_callable=lambda members: [m.value for m in members])


class DocumentModel(Base):
    """The stored file. Its ingestion state lives on its jobs, one set of chunks per parser."""
    __tablename__ = "documents"

    id      : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))

    filename: Mapped[str]       = mapped_column(Text, nullable=False)
    mime_type: Mapped[str]      = mapped_column(String(255))
    source_type: Mapped[SourceType] = mapped_column(pg_enum(SourceType, "source_type"))
    size_bytes: Mapped[int]     = mapped_column(BigInteger)
    sha256     : Mapped[str]    = mapped_column(String(64), unique=True)
    storage_path: Mapped[str]   = mapped_column(Text)

    metadata_    : Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default=text("'{}'::jsonb"))   # set at upload, filterable in search

    chunks  : Mapped[list["ChunkModel"]]        = relationship(back_populates="document", cascade="all, delete-orphan", passive_deletes=True)
    jobs    : Mapped[list["IngestionJobModel"]] = relationship(back_populates="document", cascade="all, delete-orphan", passive_deletes=True)

    created_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_documents_metadata", "metadata", postgresql_using="gin", postgresql_ops={"metadata": "jsonb_path_ops"}),
    )


class IngestionJobModel(Base):
    __tablename__ = "ingestion_jobs"

    id      : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))

    document_id : Mapped[uuid.UUID]  = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))

    status      : Mapped[JobStatus]  = mapped_column(pg_enum(JobStatus, "job_status"), server_default=JobStatus.QUEUED.value)
    parser      : Mapped[str]        = mapped_column(String(50), server_default="fast")   # "fast", "docling", "vlm" or "page"
    attempts    : Mapped[int]        = mapped_column(Integer, server_default="0")
    error       : Mapped[str | None] = mapped_column(Text)

    num_chunks             : Mapped[int]        = mapped_column(Integer, server_default="0")
    embedding_model        : Mapped[str | None] = mapped_column(String(255))   # repo@revision, to detect stale vectors
    sparse_embedding_model : Mapped[str | None] = mapped_column(Text)

    started_at  : Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at : Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document    : Mapped["DocumentModel"]    = relationship(back_populates="jobs")
    chunks      : Mapped[list["ChunkModel"]] = relationship(back_populates="job", passive_deletes=True)

    created_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_ingestion_jobs_queue", "status", "created_at"),
        Index("ix_ingestion_jobs_document_parser", "document_id", "parser", "created_at"),
    )


class ChunkModel(Base):
    __tablename__ = "chunks"

    id      : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))

    document_id : Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    job_id      : Mapped[uuid.UUID] = mapped_column(ForeignKey("ingestion_jobs.id", ondelete="CASCADE"))
    chunk_index : Mapped[int]       = mapped_column(Integer)

    modality    : Mapped[Modality]   = mapped_column(pg_enum(Modality, "modality"))
    content     : Mapped[str]        = mapped_column(Text)          # text, table as markdown, or image caption
    image_path  : Mapped[str | None] = mapped_column(Text)
    token_count : Mapped[int | None] = mapped_column(Integer)

    page         : Mapped[int | None]       = mapped_column(Integer)      # PDF page or PPTX slide
    heading_path : Mapped[list[str] | None] = mapped_column(ARRAY(Text))  # Markdown section path
    json_path    : Mapped[str | None]       = mapped_column(Text)         # e.g. $.items[3]
    boxes        : Mapped[list[list[float]] | None] = mapped_column(JSONB)  # [[x0, y0, x1, y1], ...] page fractions, top-left origin

    embedding        : Mapped[Any]        = mapped_column(HALFVEC(EMBEDDING_DIM))
    content_tsv      : Mapped[Any | None] = mapped_column(TSVECTOR, Computed("to_tsvector('simple', content)", persisted=True))
    sparse_embedding : Mapped[Any | None] = mapped_column(SPARSEVEC(SPARSE_DIM))

    document    : Mapped["DocumentModel"]     = relationship(back_populates="chunks")
    job         : Mapped["IngestionJobModel"] = relationship(back_populates="chunks")

    created_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("job_id", "chunk_index", name="uq_chunks_job_index"),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_job_id",      "job_id"),
        Index("ix_chunks_modality",    "modality"),
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        # The HNSW indexes on binary_quantize(embedding) and sparse_embedding are written by hand in the migrations.
    )

    @property
    def parser(self) -> str:
        return self.job.parser
