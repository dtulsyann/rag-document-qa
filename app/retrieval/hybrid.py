"""
Hybrid search (Dense + BM25 via Reciprocal Rank Fusion).

Runs dense (semantic) retrieval and BM25 (keyword) retrieval separately,
then fuses the two ranked lists with Reciprocal Rank Fusion (RRF). A chunk
that ranks well in EITHER list gets boosted; one that ranks well in BOTH
rises to the top.

BM25 needs the full chunk corpus to build its index -- this is cached per
collection_name so it's only built once, not on every query.
"""
from functools import lru_cache
from rank_bm25 import BM25Okapi

from app.embeddings.embedder import embed_query
from app.vectorstore.chroma_store import query as vector_query, get_all_chunks

RRF_K = 60  # standard RRF damping constant


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@lru_cache(maxsize=8)
def _get_bm25_index(collection_name: str):
    """Cached per collection. Returns (bm25_index, chunks_list)."""
    chunks = get_all_chunks(collection_name)
    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


def _bm25_search(question: str, collection_name: str, k: int) -> list[dict]:
    bm25, chunks = _get_bm25_index(collection_name)
    scores = bm25.get_scores(_tokenize(question))

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:k]]


def _dense_search(question: str, collection_name: str, k: int) -> list[dict]:
    query_vector = embed_query(question)
    return vector_query(query_vector, k=k, collection_name=collection_name)


def retrieve(question: str, k: int = 5, collection_name: str = "baseline") -> list[dict]:
    # Pull a slightly larger candidate pool from each method before fusing,
    # so RRF has enough overlap/signal to work with.
    pool_size = max(k * 3, 15)

    dense_results = _dense_search(question, collection_name, pool_size)
    bm25_results = _bm25_search(question, collection_name, pool_size)

    # Reciprocal Rank Fusion
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
        chunk_lookup[cid] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
        chunk_lookup[cid] = chunk

    fused_order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_lookup[cid] for cid, _ in fused_order[:k]]
