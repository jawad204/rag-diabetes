"""
Hybrid retriever: combines the dense (Qdrant + bge embeddings) and sparse
(BM25 keyword) retrievers into one via LangChain's EnsembleRetriever.

ROLE IN THE PIPELINE: this is the RECALL stage. Its job is to get the right
chunks somewhere into a wide candidate pool (CANDIDATES_PER_RETRIEVER=20
each), not to order them perfectly - ordering is handled downstream by the
cross-encoder in niddk_rerank_retriever.py, which can only reorder what
this stage hands it. Run this file standalone to inspect the pool itself.

How the merge works - EnsembleRetriever uses Reciprocal Rank Fusion (RRF).
It does NOT average the two retrievers' scores: BM25 scores (~9.0) and
cosine similarities (~0.75) are on totally different scales, so averaging
them would be meaningless. Instead RRF discards raw scores and uses only
each document's RANK in each list:

    rrf_score(doc) = sum over retrievers of  weight / (k + rank_in_that_list)

so a doc ranked highly by both retrievers wins, while a doc found by only
one retriever still contributes. That's what lets us fuse two systems
whose scores aren't comparable.

Reuses the components already built and tested:
  - BGEEmbeddings   from niddk_index_qdrant.py
  - preprocess      from niddk_bm25_test.py  (the fixed tokenizer)

Requires the Qdrant container to be running.
Run: python niddk_hybrid_retriever.py
"""

from pathlib import Path

from langchain_qdrant import QdrantVectorStore

# EnsembleRetriever moved between packages across LangChain versions,
# so try the known locations rather than pinning one import path.
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover
    from langchain_classic.retrievers import EnsembleRetriever

from rag_diabetes.retrieval.sparse import build_bm25, load_documents, preprocess
from rag_diabetes.retrieval.vector_store import (
    COLLECTION_NAME,
    QDRANT_URL,
    BGEEmbeddings,
)

from rag_diabetes.config import (
    CANDIDATES_PER_RETRIEVER,
    CHUNKS_FINAL_PATH,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    TOP_K,
)

CHUNKS_PATH = CHUNKS_FINAL_PATH

# How many candidates each retriever contributes before fusion.
# This is now a RECALL stage feeding the reranker (see
# niddk_rerank_retriever.py), so we pull wide: the reranker can only
# reorder what this stage hands it, and anything missed here is gone
# for good. FINAL_K is only used when running THIS file standalone.

FINAL_K = TOP_K
# Relative trust in each retriever. Must sum to 1.0.
# 0.7/0.3 favours the dense side: patient questions are usually
# paraphrases ("is it dangerous for my baby") rather than exact medical
# terminology, which is where embeddings beat keyword matching. BM25
# still carries 0.3 so exact terms (A1C, "feet", "type 2") stay findable.


TEST_QUERIES = [
    "What foods should I avoid if I have type 2 diabetes?",
    "Is gestational diabetes dangerous for my baby?",
    "What is the A1C test and what do the results mean?",
    "How can diabetes affect my feet?",
]


def build_hybrid_retriever(
    chunks_path: Path = CHUNKS_PATH,
) -> EnsembleRetriever:
    """Build the dense + sparse ensemble retriever.

    Note: EnsembleRetriever has no `k` setting of its own - it returns the
    full fused list. Trim to the number you want at the call site.
    """
    # --- sparse side: BM25 over the chunks, rebuilt in memory each run ---
    documents = load_documents(chunks_path)
    bm25 = build_bm25(documents, k=CANDIDATES_PER_RETRIEVER)

    # --- dense side: the existing Qdrant collection ---
    embedder = BGEEmbeddings()
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    dense = vector_store.as_retriever(
        search_kwargs={"k": CANDIDATES_PER_RETRIEVER}
    )

    ensemble = EnsembleRetriever(
        retrievers=[dense, bm25],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )
    return ensemble


if __name__ == "__main__":
    print("Building hybrid retriever (dense + BM25)...")
    retriever = build_hybrid_retriever()
    print(
        f"Ready. weights: dense={DENSE_WEIGHT} sparse={SPARSE_WEIGHT}, "
        f"{CANDIDATES_PER_RETRIEVER} candidates each -> top {FINAL_K}\n"
    )

    print("=" * 74)
    print("HYBRID (DENSE + BM25) RESULTS")
    print("=" * 74)
    for query in TEST_QUERIES:
        results = retriever.invoke(query)[:FINAL_K]
        print(f"\nQuery: {query}")
        print(f"  bm25 tokens -> {preprocess(query)}")
        for rank, doc in enumerate(results, 1):
            cat = doc.metadata["category"]
            print(f"  {rank}. ({cat:>13s})  {doc.page_content[:82]!r}")
