"""
LangChain-native version of the reranking stage.

Same two-stage idea as niddk_rerank_retriever.py (hybrid recall ->
cross-encoder precision), but built from LangChain components instead of
a hand-rolled class:

    ContextualCompressionRetriever
        base_retriever  = the hybrid EnsembleRetriever   (gets candidates)
        base_compressor = CrossEncoderReranker           (trims to top_n)

"Compression" is LangChain's word for "shrink the candidate list before it
reaches the LLM" - here that shrinking is done by reranking and keeping the
best TOP_N.

Why this version instead of the hand-rolled one:
  - it is a single composable retriever, so it drops straight into a chain
    when we add the generation step
  - LangSmith traces LangChain components automatically; it cannot see a
    custom class

What it costs you: the per-document rerank scores and the "was #N"
movement are hidden inside the compressor. Keep niddk_rerank_retriever.py
around as the diagnostic version when you want to SEE what reranking did.

Requires the Qdrant container to be running.
Run: python niddk_rerank_lc.py
"""

from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# these moved between packages across LangChain versions
try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:  # pragma: no cover
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from rag_diabetes.retrieval.hybrid import build_hybrid_retriever

RERANKER_MODEL = "BAAI/bge-reranker-base"
TOP_N = 5

TEST_QUERIES = [
    "What foods should I avoid if I have type 2 diabetes?",
    "Is gestational diabetes dangerous for my baby?",
    "What is the A1C test and what do the results mean?",
    "How can diabetes affect my feet?",
]


def build_reranking_retriever(top_n: int = TOP_N) -> ContextualCompressionRetriever:
    hybrid = build_hybrid_retriever()
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid,
    )


if __name__ == "__main__":
    print("Building LangChain-native reranking retriever...")
    retriever = build_reranking_retriever()
    print(f"Ready. hybrid -> rerank ({RERANKER_MODEL}) -> top {TOP_N}\n")

    print("=" * 78)
    print("CONTEXTUAL COMPRESSION RETRIEVER (hybrid + cross-encoder)")
    print("=" * 78)
    for query in TEST_QUERIES:
        results = retriever.invoke(query)
        print(f"\nQuery: {query}")
        for rank, doc in enumerate(results, 1):
            cat = doc.metadata["category"]
            print(f"  {rank}. ({cat:>13s})  {doc.page_content[:82]!r}")
