import threading
from functools import cache
from typing import Any

import torch
from PIL.Image import Image
from sentence_transformers import SentenceTransformer, CrossEncoder, SparseEncoder

from app import logger
from app.core.config import settings
from app.core.device import check_device, check_torch_device, check_dtype
from app.db.model import EMBEDDING_DIM, SPARSE_DIM

# EmbeddingGemma 2 formats. Text queries get QUERY_TEMPLATE from the model's own "query" prompt.
DOCUMENT_TEMPLATE = "title: {title} | text: {text}"
QUERY_TEMPLATE    = "task: search result | query: {query}"
IMAGE_TOKEN       = "<|image|>"


def _load_kwargs(revision: str | None) -> dict[str, Any]:
    device = check_torch_device()
    return {
        "device"       : device,
        "revision"     : revision,
        "model_kwargs" : {"dtype": _embedding_dtype(device)},
    }


def _embedding_dtype(device: str) -> torch.dtype:
    """EmbeddingGemma 2 overflows in float16 (silent NaNs), so use bfloat16 where it runs, else float32."""
    dtype = check_dtype(device)
    if dtype != torch.float16:
        return dtype
    return torch.bfloat16 if device == "mps" else torch.float32


def _model_id(model_id: str, revision: str | None) -> str:
    return f"{model_id}@{revision}" if revision else model_id


def _check_dim(model_id: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise RuntimeError(f"{model_id} outputs {actual} dimensions, but the database expects {expected}.")


def _document_input(text: str, title: str | None, image: Image | None) -> str | dict:
    text = DOCUMENT_TEMPLATE.format(title=title or "none", text=text)
    if image is not None:
        return {"text": f"{text} {IMAGE_TOKEN}", "image": [image]}
    return text

def _to_dicts(embeddings) -> list[dict[int, float]]:
    """Batch sparse tensor -> one {index: value} dict per row."""
    embeddings   = embeddings.coalesce()
    rows, cols   = embeddings.indices()
    values       = embeddings.values()
    result       = [{} for _ in range(embeddings.shape[0])]

    for row, col, value in zip(rows.tolist(), cols.tolist(), values.tolist()):
        result[row][col] = value

    return result


class DenseEmbedder:
    """One vector per chunk. Text, and images if the model supports them."""
    def __init__(
        self,
        model_id       : str,
        expected_dim   : int,
        revision       : str | None            = None,
        truncate_dim   : int | None            = None,
        batch_size     : int                   = 32,
        max_seq_length : int                   = 8192,
        config_kwargs  : dict[str, Any] | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_id, truncate_dim=truncate_dim, config_kwargs=config_kwargs, **_load_kwargs(revision))
        self.model.max_seq_length = max_seq_length

        self.tokenizer  = self.model.tokenizer
        self.model_id   = _model_id(model_id, revision)
        self.dimension  = self.model.get_embedding_dimension()
        self.batch_size = batch_size

        _check_dim(model_id, self.dimension, expected_dim)

        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"Dense embedder ready: {self.model_id} on {self.model.device} ({self.model.dtype}), {self.dimension} dims, {params:.0f}M params")

    def supports(self, modality: str) -> bool:
        return self.model.supports(modality)

    def document_input(self, text: str, title: str | None = None, image: Image | None = None) -> str | dict:
        """An image is embedded together with its caption, in one vector."""
        return _document_input(text, title, image if self.supports("image") else None)

    def encode_documents(self, inputs: list[str | dict]) -> list[list[float]]:
        """inputs come from document_input(), already formatted, so the default document prompt is turned off."""
        vectors = self.model.encode_document(inputs, prompt="", batch_size=self.batch_size, normalize_embeddings=True)
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        vector = self.model.encode_query(query, normalize_embeddings=True)
        return vector.tolist()

    def encode_image_query(self, image: Image, text: str | None = None) -> list[float]:
        """An image alone takes no prefix; text sent with it gets the query prefix, then the image."""
        inputs = {"text": f"{QUERY_TEMPLATE.format(query=text)} {IMAGE_TOKEN}", "image": [image]} if text else image
        vectors = self.model.encode([inputs], prompt="", normalize_embeddings=True)
        return vectors[0].tolist()


class MLXDenseEmbedder:
    """Same interface as DenseEmbedder, on MLX. Use an mlx-community model ID."""

    def __init__(self, model_id: str, expected_dim: int, batch_size: int = 32) -> None:
        from mlx_vlm.embedding_loader import load_embedding_model
        from mlx_vlm.utils import get_model_path
        from transformers import AutoProcessor

        path = get_model_path(model_id)
        self.model      = load_embedding_model(path)
        self.processor  = AutoProcessor.from_pretrained(path)
        self.tokenizer  = self.processor.tokenizer
        self.model_id   = model_id
        self.batch_size = batch_size
        # MLX evaluation is not thread-safe, and FastAPI runs sync routes in a thread pool
        self._lock      = threading.Lock()

        self.dimension = len(self._embed(["dimension check"])[0])
        _check_dim(model_id, self.dimension, expected_dim)

        logger.info(f"Dense embedder ready: {self.model_id} on mlx, {self.dimension} dims")

    def supports(self, modality: str) -> bool:
        return modality in ("text", "image")

    def document_input(self, text: str, title: str | None = None, image: Image | None = None) -> str | dict:
        return _document_input(text, title, image)

    def encode_documents(self, inputs: list[str | dict]) -> list[list[float]]:
        """Text is batched; text+image inputs go one at a time."""
        vectors : list[list[float] | None] = [None] * len(inputs)
        texts   = [(i, x) for i, x in enumerate(inputs) if isinstance(x, str)]

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            for (i, _), vector in zip(batch, self._embed([x for _, x in batch])):
                vectors[i] = vector

        for i, x in enumerate(inputs):
            if isinstance(x, dict):
                vectors[i] = self._embed([x["text"]], x["image"])[0]

        return vectors

    def encode_query(self, query: str) -> list[float]:
        return self._embed([QUERY_TEMPLATE.format(query=query)])[0]

    def encode_image_query(self, image: Image, text: str | None = None) -> list[float]:
        if text:
            return self._embed([f"{QUERY_TEMPLATE.format(query=text)} {IMAGE_TOKEN}"], [image])[0]
        return self._embed(None, [image])[0]

    def _embed(self, texts: list[str] | None, images: list[Image] | None = None) -> list[list[float]]:
        import mlx.core as mx

        with self._lock:
            inputs = self.processor(text=texts, images=images, padding=True, return_tensors="np")
            output = self.model(**{key: mx.array(value) for key, value in inputs.items()}).text_embeds
            return output.astype(mx.float32).tolist()


class SparseEmbedder:
    """Vocabulary-sized vector, mostly zeros. Returned as {token_index: weight}."""

    def __init__(
        self,
        model_id        : str,
        expected_dim    : int,
        revision        : str | None = None,
        max_active_dims : int        = 256,
        batch_size      : int        = 32,
    ) -> None:
        self.model = SparseEncoder(model_id, max_active_dims=max_active_dims, **_load_kwargs(revision))

        self.model_id   = _model_id(model_id, revision)
        self.dimension  = self.model.get_embedding_dimension()
        self.batch_size = batch_size

        _check_dim(model_id, self.dimension, expected_dim)

    def encode_documents(self, texts: list[str]) -> list[dict[int, float]]:
        embeddings = self.model.encode_document(texts, batch_size=self.batch_size)
        return _to_dicts(embeddings)

    def encode_query(self, query: str) -> dict[int, float]:
        embeddings = self.model.encode_query([query])
        return _to_dicts(embeddings)[0]

    def explain(self, sparse: dict[int, float], top_k: int = 10) -> list[tuple[str, float]]:
        """Which tokens carry the weight. Useful for debugging retrieval."""
        tokens = self.model.tokenizer.convert_ids_to_tokens(list(sparse))
        pairs  = sorted(zip(tokens, sparse.values()), key=lambda pair: pair[1], reverse=True)
        return pairs[:top_k]

class Reranker:
    """Scores (query, chunk) pairs directly. Not stored, runs on the top results only."""

    def __init__(self, model_id: str, revision: str | None = None, batch_size: int = 32) -> None:
        self.model      = CrossEncoder(model_id, **_load_kwargs(revision))
        self.model_id   = _model_id(model_id, revision)
        self.batch_size = batch_size

    def rank(self, query: str, documents: list[str], top_k: int | None = None) -> list[dict]:
        """Returns [{"corpus_id": index_in_documents, "score": float}, ...], best first."""
        return self.model.rank(query, documents, top_k=top_k, batch_size=self.batch_size)


@cache
def load_dense_embedder() -> DenseEmbedder | MLXDenseEmbedder:
    if check_device() == "mlx":
        return MLXDenseEmbedder(settings.EMBEDDING_MODEL_ID, EMBEDDING_DIM)
    return DenseEmbedder(settings.EMBEDDING_MODEL_ID, EMBEDDING_DIM, config_kwargs={"audio_config": None})


@cache
def load_sparse_embedder() -> SparseEmbedder | None:
    return SparseEmbedder(settings.SPARSE_MODEL_ID, SPARSE_DIM) if settings.SPARSE_MODEL_ID else None


@cache
def load_reranker() -> Reranker:
    return Reranker(settings.RERANKER_MODEL_ID)
