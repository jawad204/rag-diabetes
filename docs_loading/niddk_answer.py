"""
The generation step - the G in RAG. This is the entry point for the whole
project: ask(question) -> a patient-friendly answer with NIDDK citations.

    retrieve (hybrid)  ->  build prompt  ->  Claude  ->  answer + sources

Retrieval is hybrid only, no reranking. Measured across two independent
labelled question sets (38 questions):

    hybrid only      page recall 0.944 / 0.950,  page MRR 0.780 / 0.758
    hybrid + rerank  page recall 0.833 / 0.900,  page MRR 0.667 / 0.747

SETUP
-----
    pip install langchain-anthropic

Create a file called `.env` in the PROJECT ROOT (next to data/ and
docs_loading/, not inside docs_loading/) containing:

    ANTHROPIC_API_KEY=sk-ant-...

`.env.example` shows the format. `.env` itself is listed in .gitignore and
must never be committed - a key pushed to GitHub is a compromised key, and
scrapers find them within minutes.

load_dotenv() below reads that file into the environment at startup, so the
key never appears in your code.

Cost: ~850 input + ~350 output tokens per question. On Haiku that is
roughly $0.003 per question - about 8 cents to answer 30.

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

from niddk_hybrid_retriever import build_hybrid_retriever
from niddk_prompt import SYSTEM_INSTRUCTIONS, build_prompt

# Load .env from the project root. Explicit path rather than bare
# load_dotenv() so it works no matter which directory you run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# claude-haiku-4-5-20251001 is the cheap/fast option and is plenty for
# summarising retrieved text. Swap to "claude-sonnet-5" if answers feel thin.
MODEL = "claude-haiku-4-5-20251001"

TOP_K = 5

# 0 = as deterministic as the model gets. We want the answer to stick to the
# sources, not to be creative - and it makes re-runs comparable when you are
# iterating on the prompt.
TEMPERATURE = 0

MAX_TOKENS = 1024

DEFAULT_QUESTION = "How can diabetes affect my feet?"


class DiabetesRAG:
    """Retrieval + generation. Build once, ask many times."""

    def __init__(self, model: str = MODEL, top_k: int = TOP_K):
        self.retriever = build_hybrid_retriever()
        self.llm = ChatAnthropic(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        self.top_k = top_k

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)[: self.top_k]

    def ask(self, question: str) -> dict:
        """Answer a question. Returns the answer text and the sources used."""
        docs = self.retrieve(question)
        prompt = build_prompt(question, docs)

        response = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": response.content,
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
        print(f"  Expected a .env file at: {PROJECT_ROOT / '.env'}")
        print("  containing:  ANTHROPIC_API_KEY=sk-ant-...")
        print("  (see .env.example)")
        sys.exit(1)

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    print("Building retriever...")
    rag = DiabetesRAG()
    print(f"Model: {MODEL}\n")

    print("=" * 74)
    print(f"Q: {question}")
    print("=" * 74)

    result = rag.ask(question)

    print()
    print(result["answer"])
    print()
    print("-" * 74)
    print("SOURCES RETRIEVED")
    print("-" * 74)
    for s in result["sources"]:
        print(f"  [{s['n']}] ({s['category']}) {s['title']}")
        print(f"      {s['url']}")
