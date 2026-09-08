"""
Embeds all chunks in niddk_chunks_final.jsonl with bge-small-en-v1.5 and
indexes them into a Qdrant collection (running via Docker on localhost:6333).

Requires: `docker run -d --name qdrant-diabetes -p 6333:6333 -p 6334:6334 \
           -v qdrant_storage:/qdrant/storage qdrant/qdrant` already running.

Run: python niddk_index_qdrant.py
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/niddk_chunks_final.jsonl")
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "niddk_diabetes_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbeddings(Embeddings):
    """Thin LangChain-compatible wrapper around sentence-transformers,
    so we call the exact same model/API that already worked in
    niddk_embed_test.py - no dependency on langchain's own HF wrapper
    classes, whose exact API can vary by version."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            QUERY_INSTRUCTION + text, normalize_embeddings=True
        )
        return vector.tolist()


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_documents(chunks: list[dict]) -> list[Document]:
    docs = []
    for chunk in chunks:
        docs.append(
            Document(
                page_content=chunk["content"],
                metadata={**chunk["metadata"], "chunk_id": chunk["id"]},
            )
        )
    return docs


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    documents = to_documents(chunks)

    print(f"Loading {MODEL_NAME}...")
    embedder = BGEEmbeddings()

    print(f"Embedding + indexing {len(documents)} chunks into Qdrant collection '{COLLECTION_NAME}'...")
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedder,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,  # wipe + recreate the collection each run, for a clean re-index
    )
    print("Indexing complete.")

    # sanity check: run one real query end-to-end through Qdrant
    print("\n" + "=" * 70)
    print("SANITY CHECK QUERY")
    print("=" * 70)
    query = "What foods should I avoid if I have type 2 diabetes?"
    results = vector_store.similarity_search_with_score(query, k=3)
    print(f"\nQuery: {query}\n")
    for doc, score in results:
        print(f"score={score:.3f}  category={doc.metadata['category']:>13s}  {doc.page_content[:90]!r}")
