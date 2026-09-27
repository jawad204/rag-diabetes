"""
Every path and shared constant in one place.

WHY THIS EXISTS: paths used to be written as relative strings like
"data/niddk_chunks_final.jsonl", which resolve against whatever directory you
happened to run from. Run a script from inside a subfolder and it broke.
Here they're derived from this file's own location, so they're correct no
matter where the process starts.
"""

from pathlib import Path

# config.py -> rag_diabetes/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ENV_FILE = PROJECT_ROOT / ".env"

# --- ingestion outputs, in pipeline order ---
PAGES_PATH = DATA_DIR / "niddk_pages.jsonl"          # scrape.py writes
CHUNKS_PATH = DATA_DIR / "niddk_chunks.jsonl"        # chunk.py writes
CHUNKS_FINAL_PATH = DATA_DIR / "niddk_chunks_final.jsonl"  # metadata.py writes

# --- evaluation ---
EVAL_SET_1 = DATA_DIR / "eval_questions.json"
EVAL_SET_2 = DATA_DIR / "eval_questions_set2.json"

# --- chunking ---
CHUNK_SIZE = 600          # CHARACTERS, not tokens
CHUNK_OVERLAP = 75
MIN_BODY_CHARS = 40       # drop header-only / ToC stub chunks

# --- embeddings ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# bge models score better when the QUERY is prefixed. Documents are NOT.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --- vector store ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "niddk_diabetes_chunks"

# --- retrieval ---
CANDIDATES_PER_RETRIEVER = 20
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
TOP_K = 5

# --- reranking (built, measured, NOT used - see README) ---
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_CANDIDATES = 25

# --- generation ---
LLM_MODEL = "claude-haiku-4-5-20251001"
TEMPERATURE = 0           # deterministic: stick to sources, comparable re-runs
MAX_TOKENS = 1024
