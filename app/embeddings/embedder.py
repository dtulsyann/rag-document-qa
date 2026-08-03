"""
Embeddings.

Thin wrapper around sentence-transformers so the rest of the codebase
never touches the model directly -- swapping embedding models later
(e.g. for an experiment) means changing one line in config.py, not
hunting through the codebase.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Cached so we only load the model weights once per process.
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Used for both chunks (ingestion) and queries."""
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
