"""
Two-stage retrieval with a cross-encoder reranker.

After the initial retrieval (dense or hybrid) returns a broad set of
candidates, this module re-scores each (question, chunk) pair using a
cross-encoder (ms-marco-MiniLM-L-6-v2). The cross-encoder reads question
and chunk text jointly, producing a much more accurate relevance score
than cosine similarity alone. Results are then sorted by this score and
the top_k are returned.

The model is loaded lazily and cached so the first call pays the load
cost, but subsequent calls within the same process are instant.
"""
from functools import lru_cache
from app.config import RERANKER_MODEL_NAME


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return chunks
    model = _get_reranker()
    pairs = [(question, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [c for c, _ in scored[:top_k]]
