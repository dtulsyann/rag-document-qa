"""
Central configuration for the RAG pipeline.
Keep every tunable parameter here so experiments only need to change
values in one place (or override via the ExperimentConfig in experiments/).
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Paths
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
VECTOR_DB_DIR = f"{DATA_DIR}/vector_db"

# Embedding
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking (baseline defaults)
CHUNK_SIZE = 500          # tokens
CHUNK_OVERLAP = 50        # tokens
PARENT_CHUNK_SIZE = 1500  # tokens, used only in parent-child experiment

# Retrieval
TOP_K = 5
RERANK_CANDIDATE_K = 20   # how many candidates to pull before reranking

# Reranker
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Generation
LLM_MODEL_EVAL = "claude-haiku-4-5-20251001"   # cheap, used for eval loops
LLM_MODEL_DEMO = "claude-sonnet-4-6"            # higher quality, used for demo/app
MAX_TOKENS = 1000


@dataclass
class PipelineConfig:
    """
    Bundles every swappable knob of the pipeline into one object.
    Pass a different PipelineConfig into pipeline.answer_question()
    to run an experiment without touching the core pipeline code.
    """
    chunking_strategy: str = "fixed"      # "fixed" | "parent_child" | "semantic"
    search_strategy: str = "dense"        # "dense" | "hybrid"
    use_reranker: bool = False
    top_k: int = TOP_K
    llm_model: str = LLM_MODEL_EVAL
    collection_name: str = "baseline"     # keeps each experiment's vectors isolated
