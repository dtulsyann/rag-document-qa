"""
Pipeline orchestrator.

This is the SINGLE SOURCE OF TRUTH for "ask a question, get an answer".
Both app/main.py (the live FastAPI app) and evaluation/evaluator.py
(the eval harness) call answer_question() -- retrieval logic is never
duplicated between them, so eval metrics are guaranteed to reflect what
a real user actually experiences.

Swapping retrieval strategy (dense/hybrid/reranker) for an experiment is
done via PipelineConfig, not by branching code here.
"""
from dataclasses import dataclass, field

from app.config import PipelineConfig
from app.retrieval.dense import retrieve as dense_retrieve
from app.generation.prompt_builder import build_prompt
from app.generation.llm import generate
from app.generation.citations import build_citations, extract_cited_source_numbers


@dataclass
class RAGResult:
    answer: str
    citations: list[dict]
    retrieved_chunks: list[dict]
    cited_source_numbers: list[int] = field(default_factory=list)


def _retrieve_chunks(question: str, config: PipelineConfig) -> list[dict]:
    """
    Dispatches to the correct retrieval strategy based on config.
    Experiments 2 (hybrid) and 3 (reranker) plug in here later --
    for now only dense retrieval (the baseline) is wired up.
    """
    if config.search_strategy == "dense":
        chunks = dense_retrieve(question, k=config.top_k, collection_name=config.collection_name)
    elif config.search_strategy == "hybrid":
        from app.retrieval.hybrid import retrieve as hybrid_retrieve
        chunks = hybrid_retrieve(question, k=config.top_k, collection_name=config.collection_name)
    else:
        raise ValueError(f"Unknown search_strategy: {config.search_strategy}")

    if config.use_reranker:
        from app.retrieval.reranker import rerank
        chunks = rerank(question, chunks, top_k=config.top_k)

    return chunks


def answer_question(question: str, config: PipelineConfig = None) -> RAGResult:
    config = config or PipelineConfig()

    chunks = _retrieve_chunks(question, config)
    prompt = build_prompt(question, chunks)
    answer_text = generate(prompt, model=config.llm_model)

    citations = build_citations(chunks)
    cited_numbers = extract_cited_source_numbers(answer_text)

    return RAGResult(
        answer=answer_text,
        citations=citations,
        retrieved_chunks=chunks,
        cited_source_numbers=cited_numbers,
    )


def ingest_pdf(pdf_path: str, config: PipelineConfig = None):
    """
    Full ingestion: PDF -> pages -> chunks -> embeddings -> vector store.
    Dispatches chunking strategy based on config.
    """
    from app.ingestion.pdf_loader import extract_pages
    from app.embeddings.embedder import embed_texts
    from app.vectorstore.chroma_store import add_chunks

    config = config or PipelineConfig()
    pages = extract_pages(pdf_path)

    if config.chunking_strategy == "fixed":
        from app.chunking.fixed import chunk_pages
        chunks = chunk_pages(pages)
    else:
        raise ValueError(f"Unknown chunking_strategy: {config.chunking_strategy}")

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    add_chunks(chunks, vectors, collection_name=config.collection_name)

    return len(chunks)
