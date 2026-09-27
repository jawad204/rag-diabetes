"""
Two-stage retrieval: hybrid (recall) -> cross-encoder reranker (precision).

Stage 1 - RECALL: the hybrid retriever (dense 0.7 + BM25 0.3, fused by RRF)
returns a wide candidate pool. Its job is only to make sure the right
chunks are SOMEWHERE in that pool - not to order them well.

Stage 2 - PRECISION: a cross-encoder reranks that pool and we keep the top 5.

Why a cross-encoder is more accurate than the bi-encoder used for search:
  - bi-encoder (bge-small): embeds the query and the document SEPARATELY,
    then compares the two vectors. The document was embedded at index time,
    long before this query existed, so the two never interact.
  - cross-encoder (bge-reranker-base): feeds query + document through the
    model TOGETHER, so every query token can attend to every document
    token. Much more accurate, but far too slow to run over all 808 chunks
    - which is why it only runs on the ~25 survivors of stage 1.

First run downloads bge-reranker-base from Hugging Face (~1.1 GB), cached
afterwards. Requires the Qdrant container to be running.

Run: python niddk_rerank_retriever.py
"""

from sentence_transformers import CrossEncoder

from rag_diabetes.retrieval.sparse import preprocess
from rag_diabetes.retrieval.hybrid import build_hybrid_retriever

RERANKER_MODEL = "BAAI/bge-reranker-base"

# How many hybrid candidates to rerank. Bigger = better recall but slower,
# since the cross-encoder scores every (query, doc) pair one by one.
RERANK_CANDIDATES = 25

# How many to return after reranking.
FINAL_K = 5

TEST_QUERIES = [
    "What foods should I avoid if I have type 2 diabetes?",
    "Is gestational diabetes dangerous for my baby?",
    "What is the A1C test and what do the results mean?",
    "How can diabetes affect my feet?",
]


class RerankedRetriever:
    """Hybrid retrieval followed by cross-encoder reranking."""

    def __init__(self, final_k: int = FINAL_K, candidates: int = RERANK_CANDIDATES):
        self.hybrid = build_hybrid_retriever()
        self.cross_encoder = CrossEncoder(RERANKER_MODEL, device="cpu")
        self.final_k = final_k
        self.candidates = candidates

    def invoke(self, query: str):
        """Return (document, rerank_score) pairs, best first."""
        # stage 1: wide recall
        pool = self.hybrid.invoke(query)[: self.candidates]
        if not pool:
            return []

        # stage 2: score every (query, document) pair jointly
        pairs = [(query, doc.page_content) for doc in pool]
        scores = self.cross_encoder.predict(pairs)

        ranked = sorted(zip(scores, pool), key=lambda x: -x[0])
        return [(doc, float(score)) for score, doc in ranked[: self.final_k]]


if __name__ == "__main__":
    print("Building hybrid retriever (stage 1)...")
    retriever = RerankedRetriever()
    print(f"Loaded reranker {RERANKER_MODEL} (stage 2)")
    print(f"Ready. hybrid -> {RERANK_CANDIDATES} candidates -> reranked -> top {FINAL_K}\n")

    print("=" * 78)
    print("HYBRID + CROSS-ENCODER RERANK")
    print("=" * 78)
    for query in TEST_QUERIES:
        # show what stage 1 produced, so you can see what reranking changed
        pool = retriever.hybrid.invoke(query)[:RERANK_CANDIDATES]
        pool_ids = [d.metadata["chunk_id"] for d in pool]

        results = retriever.invoke(query)

        print(f"\nQuery: {query}")
        print(f"  bm25 tokens -> {preprocess(query)}")
        print(f"  stage 1 pool: {len(pool)} candidates")
        for rank, (doc, score) in enumerate(results, 1):
            was = pool_ids.index(doc.metadata["chunk_id"]) + 1
            move = f"was #{was}"
            cat = doc.metadata["category"]
            print(f"  {rank}. [{score:6.2f}] ({cat:>13s}, {move:>7s})  {doc.page_content[:70]!r}")
