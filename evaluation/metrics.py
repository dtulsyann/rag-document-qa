"""
Phase 6: Retrieval evaluation metrics.

Hand-rolled (not RAGAS) so every formula is transparent and explainable.
A "match" between a retrieved chunk and a ground-truth source is defined
as same filename + same page number (a chunk is considered relevant if
it comes from the page the ground truth points to).
"""


def _is_match(chunk: dict, relevant_sources: list[dict]) -> bool:
    for src in relevant_sources:
        if chunk["filename"] == src["filename"] and chunk["page_number"] == src["page"]:
            return True
    return False


def hit_at_k(retrieved_chunks: list[dict], relevant_sources: list[dict]) -> int:
    """1 if ANY retrieved chunk matches a ground-truth source, else 0."""
    return int(any(_is_match(c, relevant_sources) for c in retrieved_chunks))


def reciprocal_rank(retrieved_chunks: list[dict], relevant_sources: list[dict]) -> float:
    """1 / rank of the first matching chunk (0 if none match). Used for MRR."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if _is_match(chunk, relevant_sources):
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_chunks: list[dict], relevant_sources: list[dict]) -> float:
    """Fraction of all ground-truth relevant sources that were retrieved."""
    if not relevant_sources:
        return 0.0
    matched_sources = set()
    for src in relevant_sources:
        for chunk in retrieved_chunks:
            if chunk["filename"] == src["filename"] and chunk["page_number"] == src["page"]:
                matched_sources.add((src["filename"], src["page"]))
                break
    return len(matched_sources) / len(relevant_sources)


def aggregate_metrics(per_question_results: list[dict]) -> dict:
    """
    Args:
        per_question_results: list of {"hit", "reciprocal_rank", "recall"}
    Returns:
        dict of averaged metrics across the whole dataset
    """
    n = len(per_question_results)
    if n == 0:
        return {"hit_rate": 0.0, "mrr": 0.0, "avg_recall": 0.0, "n_questions": 0}

    return {
        "hit_rate": sum(r["hit"] for r in per_question_results) / n,
        "mrr": sum(r["reciprocal_rank"] for r in per_question_results) / n,
        "avg_recall": sum(r["recall"] for r in per_question_results) / n,
        "n_questions": n,
    }
