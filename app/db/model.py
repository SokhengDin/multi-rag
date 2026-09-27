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

# BGE-M3 / Qwen3-Embedding-0.6B = 1024. Changing it needs a migration and re-embedding :), thin carefully
EMBEDDING_DIM = 1024
# SPLADE-style sparse vectors use the BERT vocab size.
SPARSE_DIM    = 30522

class Base(DeclarativeBase):
    pass


class DocumentStatus(str, enum.Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    READY      = "ready"
    FAILED     = "failed"


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
    __tablename__ = "documents"

    id      : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))

    filename: Mapped[str]       = mapped_column(Text, nullable=False)
    mime_type: Mapped[str]      = mapped_column(String(255))
    source_type: Mapped[SourceType] = mapped_column(pg_enum(SourceType, "source_type"))
    size_bytes: Mapped[int]     = mapped_column(BigInteger)
    sha256     : Mapped[str]    = mapped_column(String(64), unique=True)
    storage_path: Mapped[str]   = mapped_column(Text)

    status  : Mapped[DocumentStatus] = mapped_column(pg_enum(DocumentStatus, "document_status"), server_default=DocumentStatus.PENDING.value, index=True)
    error   : Mapped[str | None] = mapped_column(Text)
    num_chunks: Mapped[int] = mapped_column(Integer, server_default="0")

    metadata_    : Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default=text("'{}'::jsonb"))

    chunks  : Mapped[list["ChunkModel"]]        = relationship(back_populates="document", cascade="all, delete-orphan", passive_deletes=True)
    jobs    : Mapped[list["IngestionJobModel"]] = relationship(back_populates="document", cascade="all, delete-orphan", passive_deletes=True)

    created_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ChunkModel(Base):
    __tablename__ = "chunks"

    id      : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))

    document_id : Mapped[uuid.UUID]        = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    parent_id   : Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"))
    chunk_index : Mapped[int]              = mapped_column(Integer)

    modality    : Mapped[Modality]   = mapped_column(pg_enum(Modality, "modality"))
    content     : Mapped[str]        = mapped_column(Text)          # text, table as markdown, or image caption
    image_path  : Mapped[str | None] = mapped_column(Text)
    token_count : Mapped[int | None] = mapped_column(Integer)

    page         : Mapped[int | None]       = mapped_column(Integer)      # PDF page or PPTX slide
    heading_path : Mapped[list[str] | None] = mapped_column(ARRAY(Text))  # Markdown section path
    json_path    : Mapped[str | None]       = mapped_column(Text)         # e.g. $.items[3]

    embedding       : Mapped[Any] = mapped_column(HALFVEC(EMBEDDING_DIM))
    embedding_model : Mapped[str] = mapped_column(String(255))            # repo@revision, to detect stale vectors
    content_tsv     : Mapped[Any] = mapped_column(TSVECTOR, Computed("to_tsvector('simple', content)", persisted=True))

    sparse_embedding       : Mapped[Any | None] = mapped_column(SPARSEVEC(SPARSE_DIM))
    sparse_embedding_model : Mapped[str | None] = mapped_column(Text)

    metadata_    : Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default=text("'{}'::jsonb"))

    document    : Mapped["DocumentModel"] = relationship(back_populates="chunks")

    created_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_modality",    "modality"),
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        Index("ix_chunks_metadata",    "metadata",    postgresql_using="gin", postgresql_ops={"metadata": "jsonb_path_ops"}),
        # The HNSW index on binary_quantize(embedding) is written by hand in the migration.
    )


class IngestionJobModel(Base):
    __tablename__ = "ingestion_jobs"

    id      : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))

    document_id : Mapped[uuid.UUID]  = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))

    status      : Mapped[JobStatus]  = mapped_column(pg_enum(JobStatus, "job_status"), server_default=JobStatus.QUEUED.value)
    parser      : Mapped[str]        = mapped_column(String(50), server_default="fast")   # "fast" or "docling"
    attempts    : Mapped[int]        = mapped_column(Integer, server_default="0")
    error       : Mapped[str | None] = mapped_column(Text)

    started_at  : Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at : Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document    : Mapped["DocumentModel"] = relationship(back_populates="jobs")

    created_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_ingestion_jobs_queue", "status", "created_at"), 
    )
