"""
Sanity check for the embedding model before we commit to embedding and
indexing all 808 chunks: load BAAI/bge-small-en-v1.5, embed a handful of
sample chunks from different categories, embed a few realistic patient
questions, and check that the *right* chunk scores highest for each
question via cosine similarity.

Run: python niddk_embed_test.py

(First run downloads the model from Hugging Face - a few hundred MB,
cached afterward so later runs are instant.)
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/niddk_chunks_final.jsonl")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models score noticeably better when the QUERY (not the documents)
# is prefixed with this instruction at search time. Documents are NOT
# prefixed - only queries, at search time.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# A few realistic patient questions, each paired with the category we'd
# expect the top match to come from - lets us check "did the right kind
# of chunk win" rather than just eyeballing raw scores.
TEST_QUERIES = [
    ("What foods should I avoid if I have type 2 diabetes?", "type-2"),
    ("Is gestational diabetes dangerous for my baby?", "gestational"),
    ("What is the A1C test and what do the results mean?", "diagnosis"),
    ("How can diabetes affect my feet?", "complications"),
    ("aaaaaa","gestational")
]


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    # pick a manageable sample: up to 3 chunks per category, so the test
    # set has variety without embedding all 808 chunks yet
    sample = []
    seen_per_category: dict[str, int] = {}
    for c in chunks:
        cat = c["metadata"]["category"]
        if seen_per_category.get(cat, 0) < 3:
            sample.append(c)
            seen_per_category[cat] = seen_per_category.get(cat, 0) + 1
    print(f"Using a sample of {len(sample)} chunks across {len(seen_per_category)} categories\n")

    print(f"Loading {MODEL_NAME} (first run downloads the model)...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    doc_texts = [c["content"] for c in sample]
    doc_vectors = model.encode(doc_texts, normalize_embeddings=True)
    print(f"Embedded {len(doc_vectors)} sample chunks, dimension = {doc_vectors.shape[1]}\n")

    print("=" * 70)
    print("QUERY -> TOP 3 MATCHES")
    print("=" * 70)
    for query, expected_category in TEST_QUERIES:
        query_vector = model.encode(QUERY_INSTRUCTION + query, normalize_embeddings=True)
        sims = [cosine_sim(query_vector, dv) for dv in doc_vectors]
        ranked = sorted(zip(sims, sample), key=lambda x: -x[0])[:3]

        print(f"\nQuery: {query}")
        print(f"(expected category: {expected_category})")
        for score, chunk in ranked:
            hit = "OK" if chunk["metadata"]["category"] == expected_category else "  "
            print(f"  [{hit}] {score:.3f}  ({chunk['metadata']['category']:>13s})  {chunk['content'][:80]!r}")
