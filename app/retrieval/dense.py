"""
Phase 4: Baseline dense retrieval (pure vector similarity search).
"""
from app.embeddings.embedder import embed_query
from app.vectorstore.chroma_store import query as vector_query


def retrieve(question: str, k: int = 5, collection_name: str = "baseline") -> list[dict]:
    query_vector = embed_query(question)
    return vector_query(query_vector, k=k, collection_name=collection_name)
