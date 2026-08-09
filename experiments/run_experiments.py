"""
Run all experiments through the same evaluation harness.

Each experiment config gets its own Chroma collection (so vectors don't
mix between experiments) and its own results JSON file.

Before running this, ingest the SAME test PDF(s) into each collection
matching the config below, e.g.:

    from app.pipeline import ingest_pdf
    from app.config import PipelineConfig
    ingest_pdf("data/raw/test.pdf", PipelineConfig(collection_name="baseline"))
    ingest_pdf("data/raw/test.pdf", PipelineConfig(collection_name="hybrid", search_strategy="hybrid"))
    ingest_pdf("data/raw/test.pdf", PipelineConfig(collection_name="reranker"))  # same chunks as baseline

Usage:
    python -m experiments.run_experiments
"""
from app.config import PipelineConfig
from evaluation.evaluator import run_evaluation

EXPERIMENTS = [
    ("baseline", PipelineConfig(
        collection_name="baseline",
        chunking_strategy="fixed",
        search_strategy="dense",
        use_reranker=False,
    )),
    ("exp1_hybrid_search", PipelineConfig(
        collection_name="hybrid",
        chunking_strategy="fixed",
        search_strategy="hybrid",
        use_reranker=False,
    )),
    ("exp2_reranker", PipelineConfig(
        collection_name="reranker",
        chunking_strategy="fixed",
        search_strategy="dense",
        use_reranker=True,
    )),
]


def run_all():
    reports = {}
    for name, config in EXPERIMENTS:
        reports[name] = run_evaluation(config, experiment_name=name)
    return reports


if __name__ == "__main__":
    run_all()
