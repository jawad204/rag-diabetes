"""
BM25 (sparse / keyword) retriever over the same 808 chunks.

IMPORTANT - why the custom preprocess_func:
LangChain's BM25Retriever defaults to `text.split()` for tokenization,
which means NO lowercasing, NO punctuation stripping, and NO stopword
removal. On this corpus that badly breaks retrieval:

  - the query "...type 2 diabetes?" produces the token 'diabetes?'
    (with the question mark), which appears in only 74/808 chunks,
    while plain 'diabetes' appears in 430 - so the single most
    important query term silently fails to match most of the corpus
  - stopwords like 'have' (328 chunks), 'if' (199) and 'type' (221)
    become scoring terms, so chunks that merely stack common words
    outrank chunks that are actually on-topic
  - 'Diabetes' and 'diabetes' are treated as different terms

So we pass our own tokenizer: lowercase, strip punctuation, drop
stopwords. Verified to fix both failure modes on this data.

Run: python niddk_bm25_test.py
"""

import json
import re
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

CHUNKS_PATH = Path("data/niddk_chunks_final.jsonl")
TOP_K = 3

# Deliberately conservative: only function words and question words.
# Note we do NOT drop 'type' or any digits - "type 1" / "type 2" are
# meaningful medical terms in this corpus, not noise.
STOPWORDS = set(
    """a an the and or but if of to in on at for with without from by as
    is are was were be been being have has had do does did
    i you he she it we they me my your his her its our their
    this that these those what which who whom when where why how
    can could should would may might will shall must there here about
    into over under again further then once all any both each few more
    most other some such no nor not only own same so than too very just now""".split()
)


def preprocess(text: str) -> list[str]:
    """Lowercase, keep only alphanumeric tokens, drop stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


TEST_QUERIES = [
    "What foods should I avoid if I have type 2 diabetes?",
    "Is gestational diabetes dangerous for my baby?",
    "What is the A1C test and what do the results mean?",
    "How can diabetes affect my feet?",
]


def load_documents(path: Path) -> list[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            docs.append(
                Document(
                    page_content=chunk["content"],
                    metadata={**chunk["metadata"], "chunk_id": chunk["id"]},
                )
            )
    return docs


def build_bm25(documents: list[Document], k: int = TOP_K) -> BM25Retriever:
    bm25 = BM25Retriever.from_documents(documents, preprocess_func=preprocess)
    bm25.k = k
    return bm25


if __name__ == "__main__":
    documents = load_documents(CHUNKS_PATH)
    print(f"Loaded {len(documents)} chunks from {CHUNKS_PATH}")

    bm25 = build_bm25(documents)
    print(f"Built BM25 index (custom tokenizer), returning top {TOP_K} per query\n")

    print("=" * 70)
    print("BM25 (KEYWORD) RESULTS")
    print("=" * 70)
    for query in TEST_QUERIES:
        results = bm25.invoke(query)
        print(f"\nQuery: {query}")
        print(f"  tokens -> {preprocess(query)}")
        for doc in results:
            cat = doc.metadata["category"]
            print(f"  ({cat:>13s})  {doc.page_content[:88]!r}")
