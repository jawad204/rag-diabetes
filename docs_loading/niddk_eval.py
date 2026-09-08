"""
Retrieval evaluation: measures recall@k and MRR over a labelled question set,
and compares HYBRID ALONE against HYBRID + RERANKING.

That comparison is the point. Right now you don't know whether reranking
helps overall - it clearly helped the feet and foods queries and clearly
hurt the gestational one. This turns that into a number.

THE TWO METRICS
---------------
recall@k : did any correct chunk appear in the top k?  1 or 0 per question,
           then averaged. Answers "did we find it at all?"

MRR      : 1 / (position of the first correct chunk), or 0 if it never
           appeared, then averaged. Answers "did we put it near the top?"
           #1 -> 1.00,  #2 -> 0.50,  #3 -> 0.33,  #5 -> 0.20

Both are needed: recall can't tell #1 from #5, MRR can.

CHUNK LEVEL vs PAGE LEVEL
-------------------------
Each metric is reported twice, because exact-chunk labels are too strict.
The diabetic-kidney-disease page has 18 chunks and several of them answer
"Can diabetes damage my kidneys?" perfectly - but the label names only one.
Returning a different chunk from that same page is a good result, and
chunk-level scoring calls it a total miss.

  chunk level : did we return the exact labelled chunk?   (strict)
  page level  : did we return ANY chunk from the right page?  (fair)

Read them together. A low chunk score with a high page score means
retrieval is finding the right document and the labels are just picky.
Both low means retrieval genuinely went to the wrong topic.

Requires the Qdrant container to be running.

Run: python niddk_eval.py                             (default question set)
     python niddk_eval.py data/eval_questions_set2.json   (a different set)

Set 2 is an independent set covering areas set 1 never touched (hypoglycemia,
heart/stroke, eye, dental, sexual/bladder, neuropathy, healthy living, islet
transplant). Running both is how you check a finding wasn't a fluke of one
sample - 18 or 20 questions is small enough that a single set can mislead.
"""

import json
import sys
from pathlib import Path

from niddk_hybrid_retriever import build_hybrid_retriever
from niddk_rerank_lc import build_reranking_retriever

EVAL_PATH = Path("eval_questions.json")
K = 5


# ---------------------------------------------------------------------------
# THE METRICS
#
# These two functions are the whole of "model evaluation" - it really is just
# arithmetic. Worth trying to write them yourself before reading mine: cover
# the bodies, and go from the description above. Each is about three lines.
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """1.0 if any correct chunk is in the top k, else 0.0."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in top_k for rid in relevant_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """1 / position of the first correct chunk (1-indexed), or 0 if absent."""
    for position, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


# ---------------------------------------------------------------------------


def page_of(chunk_id: str) -> str:
    """Chunk ids look like '<url>#chunk-7', so the page is everything
    before the '#'."""
    return chunk_id.split("#")[0]


def load_questions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(retriever, questions: list[dict], k: int = K) -> dict:
    """Run every question through a retriever, scoring at chunk AND page level."""
    chunk_recalls, chunk_rrs = [], []
    page_recalls, page_rrs = [], []
    per_question = []

    for item in questions:
        docs = retriever.invoke(item["question"])[:k]
        retrieved_ids = [d.metadata["chunk_id"] for d in docs]
        retrieved_pages = [page_of(cid) for cid in retrieved_ids]
        relevant_pages = [page_of(cid) for cid in item["relevant_ids"]]

        cr = recall_at_k(retrieved_ids, item["relevant_ids"], k)
        crr = reciprocal_rank(retrieved_ids, item["relevant_ids"], k)
        pr = recall_at_k(retrieved_pages, relevant_pages, k)
        prr = reciprocal_rank(retrieved_pages, relevant_pages, k)

        chunk_recalls.append(cr)
        chunk_rrs.append(crr)
        page_recalls.append(pr)
        page_rrs.append(prr)

        per_question.append(
            {
                "question": item["question"],
                "chunk_position": (int(round(1 / crr)) if crr > 0 else None),
                "page_position": (int(round(1 / prr)) if prr > 0 else None),
            }
        )

    n = len(questions)
    return {
        "chunk_recall": sum(chunk_recalls) / n,
        "chunk_mrr": sum(chunk_rrs) / n,
        "page_recall": sum(page_recalls) / n,
        "page_mrr": sum(page_rrs) / n,
        "per_question": per_question,
    }


if __name__ == "__main__":
    # optional first argument: a different question set
    eval_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL_PATH
    questions = load_questions(eval_path)
    print(f"Loaded {len(questions)} labelled questions from {eval_path}\n")

    print("Building retrievers...")
    hybrid_only = build_hybrid_retriever()
    with_rerank = build_reranking_retriever(top_n=K)
    print()

    configs = [("hybrid only", hybrid_only), ("hybrid + rerank", with_rerank)]
    results = {}

    for name, retriever in configs:
        print(f"Evaluating: {name} ...")
        results[name] = evaluate(retriever, questions)

    # ---- summary ----
    print("\n" + "=" * 74)
    print(f"RESULTS  (k={K})")
    print("=" * 74)
    print(f"{'config':<20s} {'chunk recall':>13s} {'chunk MRR':>11s} {'page recall':>13s} {'page MRR':>11s}")
    print("-" * 74)
    for name, _ in configs:
        r = results[name]
        print(
            f"{name:<20s} {r['chunk_recall']:>13.3f} {r['chunk_mrr']:>11.3f} "
            f"{r['page_recall']:>13.3f} {r['page_mrr']:>11.3f}"
        )

    # ---- per-question, so you can see WHICH questions changed ----
    fmt = lambda p: f"#{p}" if p else "miss"
    for level, key in [("CHUNK", "chunk_position"), ("PAGE", "page_position")]:
        print("\n" + "=" * 74)
        print(f"PER QUESTION - {level} LEVEL (position of first correct hit)")
        print("=" * 74)
        print(f"{'question':<46s} {'hybrid':>8s} {'rerank':>8s}")
        print("-" * 74)
        for i, item in enumerate(questions):
            a = results["hybrid only"]["per_question"][i][key]
            b = results["hybrid + rerank"]["per_question"][i][key]
            flag = ""
            if a and b and b < a:
                flag = "  better"
            elif (a and not b) or (a and b and b > a):
                flag = "  WORSE"
            print(f"{item['question'][:44]:<46s} {fmt(a):>8s} {fmt(b):>8s}{flag}")