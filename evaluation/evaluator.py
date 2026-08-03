"""
Phase 6: Evaluation harness.

Reuses app.pipeline.answer_question() -- the exact same function the live
FastAPI app calls. This guarantees eval numbers reflect real user behavior;
retrieval logic is never reimplemented here.

Usage:
    python -m evaluation.evaluator --config baseline
"""
import json
import os
from datetime import datetime, timezone

from app.config import PipelineConfig
from app.pipeline import answer_question
from evaluation.metrics import hit_at_k, reciprocal_rank, recall_at_k, aggregate_metrics

DATASET_PATH = "evaluation/dataset.json"
RESULTS_DIR = "experiments/results"


def load_dataset(path: str = DATASET_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_evaluation(config: PipelineConfig, experiment_name: str,
                    dataset_path: str = DATASET_PATH) -> dict:
    dataset = load_dataset(dataset_path)
    per_question_results = []

    for item in dataset:
        result = answer_question(item["question"], config=config)
        retrieved = result.retrieved_chunks
        relevant = item["relevant_sources"]

        per_question_results.append({
            "id": item["id"],
            "question": item["question"],
            "hit": hit_at_k(retrieved, relevant),
            "reciprocal_rank": reciprocal_rank(retrieved, relevant),
            "recall": recall_at_k(retrieved, relevant),
            "answer": result.answer,
            "retrieved_pages": [
                {"filename": c["filename"], "page": c["page_number"]} for c in retrieved
            ],
        })

    aggregate = aggregate_metrics(per_question_results)

    report = {
        "experiment_name": experiment_name,
        "config": {
            "chunking_strategy": config.chunking_strategy,
            "search_strategy": config.search_strategy,
            "use_reranker": config.use_reranker,
            "top_k": config.top_k,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate_metrics": aggregate,
        "per_question_results": per_question_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{experiment_name}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== {experiment_name} ===")
    print(f"Hit Rate: {aggregate['hit_rate']:.2%}")
    print(f"MRR:      {aggregate['mrr']:.3f}")
    print(f"Recall:   {aggregate['avg_recall']:.2%}")
    print(f"Saved -> {out_path}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline",
                         help="Name for this run, used as the results filename")
    parser.add_argument("--collection", default="baseline",
                         help="Chroma collection to query against")
    parser.add_argument("--search-strategy", default="dense",
                         choices=["dense", "hybrid"],
                         help="Retrieval strategy: dense or hybrid")
    parser.add_argument("--chunking", default="fixed",
                         choices=["fixed", "parent_child"],
                         help="Chunking strategy used during ingestion")
    parser.add_argument("--reranker", action="store_true",
                         help="Enable cross-encoder reranking")
    args = parser.parse_args()

    config = PipelineConfig(
        collection_name=args.collection,
        search_strategy=args.search_strategy,
        chunking_strategy=args.chunking,
        use_reranker=args.reranker,
    )
    run_evaluation(config, experiment_name=args.experiment)
