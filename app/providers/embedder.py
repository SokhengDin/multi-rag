from typing import Any

from sentence_transformers import SentenceTransformer, CrossEncoder, SparseEncoder
from app.core.device import check_device, check_dtype
from app.db.model import EMBEDDING_DIM

def _load_kwargs(revision: str | None) -> dict[str, Any]:
    device = check_device()
    return {
        "device"       : device,
        "revision"     : revision,
        "model_kwargs" : {"dtype": check_dtype(device)},
    }


def _model_id(model_id: str, revision: str | None) -> str:
    return f"{model_id}@{revision}" if revision else model_id


def _check_dim(model_id: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise RuntimeError(f"{model_id} outputs {actual} dimensions, but the database expects {expected}.")

def _to_dicts(embeddings) -> list[dict[int, float]]:
    """Batch sparse tensor -> one {index: value} dict per row."""
    embeddings   = embeddings.coalesce()
    rows, cols   = embeddings.indices()
    values       = embeddings.values()
    result       = [{} for _ in range(embeddings.shape[0])]

    for row, col, value in zip(rows.tolist(), cols.tolist(), values.tolist()):
        result[row][col] = value

    return result

# Dense
class DenseEmbedder:
    """One vector per chunk. Text, and images if the model supports them."""
    def __init__(
        self,
        model_id     : str,
        expected_dim : int,
        revision     : str | None = None,
        truncate_dim : int | None = None,
        batch_size   : int        = 32,
    ) -> None:
        self.model = SentenceTransformer(model_id, truncate_dim=truncate_dim, **_load_kwargs(revision))

        self.model_id   = _model_id(model_id, revision)
        self.dimension  = self.model.get_embedding_dimension()
        self.batch_size = batch_size

        _check_dim(model_id, self.dimension, expected_dim)

    def supports(self, modality: str) -> bool:
        return self.model.supports(modality)

    def encode_documents(self, inputs: list[Any]) -> list[list[float]]:
        vectors = self.model.encode_document(inputs, batch_size=self.batch_size, normalize_embeddings=True)
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        vector = self.model.encode_query(query, normalize_embeddings=True)
        return vector.tolist()


# Sparse

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