"""
Vector store.

Thin wrapper around ChromaDB. Notice we're using Chroma's raw client
API directly (not a LangChain VectorStore abstraction) -- ids, embeddings,
documents, and metadata all stay explicit and visible.

`collection_name` lets each experiment (baseline / parent_child / hybrid /
reranker) keep its own isolated set of vectors, so experiments never
contaminate each other's index.
"""
import chromadb
from app.config import VECTOR_DB_DIR

_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)


def get_collection(collection_name: str = "baseline"):
    return _client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict], vectors: list[list[float]],
                collection_name: str = "baseline"):
    """
    Args:
        chunks: list of chunk dicts (must include chunk_id + metadata fields)
        vectors: embeddings, same order/length as chunks
    """
    collection = get_collection(collection_name)
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "filename": c["filename"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                # parent_id is only present for parent-child chunks;
                # Chroma metadata can't store None cleanly across all
                # versions, so default to "" when absent.
                "parent_id": c.get("parent_id", ""),
            }
            for c in chunks
        ],
    )


def query(query_vector: list[float], k: int = 5,
          collection_name: str = "baseline") -> list[dict]:
    """
    Returns list of dicts: [{"chunk_id", "text", "filename", "page_number",
    "chunk_index", "parent_id", "score"}, ...] ordered by relevance.
    """
    collection = get_collection(collection_name)
    results = collection.query(query_embeddings=[query_vector], n_results=k)

    out = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        out.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "filename": meta["filename"],
            "page_number": meta["page_number"],
            "chunk_index": meta["chunk_index"],
            "parent_id": meta.get("parent_id", ""),
            # Chroma returns cosine distance; convert to a similarity score
            "score": 1 - results["distances"][0][i],
        })
    return out


def reset_collection(collection_name: str = "baseline"):
    """Useful when re-ingesting during development/experiments."""
    try:
        _client.delete_collection(collection_name)
    except Exception:
        pass
