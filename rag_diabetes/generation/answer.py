"""
The generation step - the G in RAG. This is the entry point for the whole
project: ask(question) -> a patient-friendly answer with NIDDK citations,
or a refusal if the question is outside the corpus.

    question
      |
      +-- GATE: best cosine score below RELEVANCE_THRESHOLD?  -> refuse, no LLM call
      |
      +-- retrieve (hybrid, no reranking)
            -> build prompt
            -> Claude
            -> answer + sources

TWO LAYERS OF SCOPE CONTROL - neither is enough alone
-----------------------------------------------------
1. The cosine gate below. Free (the score is already computed), instant, and
   it never argues. Measured on 10 in-scope and 10 out-of-scope questions:
   at 0.70 it passed 10/10 real questions and blocked 9/10 off-topic ones.

2. The prompt's SCOPE section (see niddk_prompt.py). Catches what the gate
   misses - questions that LOOK like diabetes vocabulary but aren't covered.
   "Side effects of chemotherapy" scored 0.7521 and sails through the gate,
   but Claude reading five actual diabetes chunks can see they say nothing
   about chemotherapy.

Why not a cross-encoder gate: tested, and it was far worse (0.497 overlap vs
cosine's 0.044). It rates "best treatment for a broken arm" 0.93 against
"cover a blister, cut, or sore with a bandage", because it judges whether text
READS LIKE a response, not whether it's the right topic.

RETRIEVAL is hybrid only, no reranking. Measured across two independent
labelled question sets (38 questions):

    hybrid only      page recall 0.944 / 0.950,  page MRR 0.780 / 0.758
    hybrid + rerank  page recall 0.833 / 0.900,  page MRR 0.667 / 0.747

SETUP
-----
    pip install langchain-anthropic

Create a file called `.env` in the PROJECT ROOT (next to data/ and
docs_loading/, not inside docs_loading/) containing:

    ANTHROPIC_API_KEY=sk-ant-...

`.env.example` shows the format. `.env` itself is in .gitignore and must
never be committed - a key pushed to GitHub is a compromised key.

Cost: ~850 input + ~350 output tokens per question, roughly $0.003 on Haiku.
Refused questions cost nothing - the gate runs before the API call.

Requires the Qdrant container to be running.

Run:  python niddk_answer.py
      python niddk_answer.py "Can diabetes damage my kidneys?"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from rag_diabetes.retrieval.hybrid import build_hybrid_retriever
from rag_diabetes.retrieval.vector_store import BGEEmbeddings
from rag_diabetes.config import (
    COLLECTION_NAME,
    ENV_FILE,
    LLM_MODEL,
    MAX_TOKENS,
    QDRANT_URL,
    TEMPERATURE,
    TOP_K,
)
from rag_diabetes.generation.guardrails import (
    REFUSAL,
    RELEVANCE_THRESHOLD,
    TOPICS,
    is_meta_question,
    passes_relevance_gate,
)
from rag_diabetes.generation.prompt import build_prompt

# Load .env from the project root. Explicit path rather than bare
# load_dotenv() so it works no matter which directory you run from.
load_dotenv(ENV_FILE)

MODEL = LLM_MODEL

DEFAULT_QUESTION = "How can diabetes affect my feet?"


class DiabetesRAG:
    """Retrieval + generation, with an out-of-scope gate. Build once, ask many."""

    def __init__(self, model: str = MODEL, top_k: int = TOP_K):
        self.embedder = BGEEmbeddings()
        # the hybrid retriever does the actual retrieval...
        self.retriever = build_hybrid_retriever()
        # ...but it returns RRF scores, which are rank-based and say nothing
        # about absolute relevance. So we keep a direct handle on the vector
        # store purely to read the raw cosine score for the gate.
        self.store = QdrantVectorStore.from_existing_collection(
            embedding=self.embedder,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
        )
        self.llm = ChatAnthropic(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        self.top_k = top_k

    def relevance_score(self, question: str) -> float:
        """Best cosine similarity between the question and any chunk."""
        hits = self.store.similarity_search_with_score(question, k=1)
        return float(hits[0][1]) if hits else 0.0

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)[: self.top_k]

    def ask(self, question: str) -> dict:
        """Answer a question, or refuse if it falls outside the corpus."""
        # "what can I ask?" is about the assistant, not about diabetes - it
        # would score low and be refused. Answer it directly instead.
        if is_meta_question(question):
            return {
                "question": question,
                "answer": TOPICS,
                "refused": False,
                "reason": "meta question - answered from the topic list",
                "relevance_score": 1.0,
                "sources": [],
                "prompt": None,
            }

        score = self.relevance_score(question)

        # --- gate: refuse before spending an API call ---
        if not passes_relevance_gate(score):
            return {
                "question": question,
                "answer": REFUSAL,
                "refused": True,
                "reason": f"relevance {score:.4f} below threshold {RELEVANCE_THRESHOLD}",
                "relevance_score": score,
                "sources": [],
                "prompt": None,
            }

        docs = self.retrieve(question)
        prompt = build_prompt(question, docs)
        response = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": response.content,
            "refused": False,
            "reason": None,
            "relevance_score": score,
            "sources": [
                {
                    "n": i,
                    "title": d.metadata.get("title", "NIDDK"),
                    "url": d.metadata.get("url", ""),
                    "category": d.metadata.get("category", ""),
                }
                for i, d in enumerate(docs, 1)
            ],
            "prompt": prompt,  # kept so you can inspect exactly what was sent
        }


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not found.")
        print(f"  Expected a .env file at: {ENV_FILE}")
        print("  containing:  ANTHROPIC_API_KEY=sk-ant-...")
        print("  (see .env.example)")
        sys.exit(1)

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    print("Building retriever...")
    rag = DiabetesRAG()
    print(f"Model: {MODEL}   gate: cosine >= {RELEVANCE_THRESHOLD}\n")

    print("=" * 74)
    print(f"Q: {question}")
    print("=" * 74)

    result = rag.ask(question)

    print(f"\n(relevance {result['relevance_score']:.4f})")
    if result["refused"]:
        print(f"REFUSED - {result['reason']}\n")
        print(result["answer"])
        sys.exit(0)

    print()
    print(result["answer"])
    print()
    print("-" * 74)
    print("SOURCES RETRIEVED")
    print("-" * 74)
    for s in result["sources"]:
        print(f"  [{s['n']}] ({s['category']}) {s['title']}")
        print(f"      {s['url']}")
