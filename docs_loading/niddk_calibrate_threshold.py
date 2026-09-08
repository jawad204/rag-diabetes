"""
Finds the cosine-similarity cutoff that separates questions your corpus CAN
answer from questions it CANNOT.

WHY: the pipeline always returns 5 chunks, no matter what you ask. Ask about
a broken arm and you still get 5 diabetes chunks back. The prompt tells the
model to refuse in that case, but that is a request, not a rule - a model can
be talked around it. A threshold cannot: if the best chunk isn't similar
enough, we never call the LLM at all.

HOW: run known IN-SCOPE and known OUT-OF-SCOPE questions, print the best
cosine score for each, and look at the gap between the two groups. The
threshold goes in that gap.

Note we do NOT use the hybrid RRF score. RRF is built from rank positions
(~0.01 range, always similar) and says nothing about absolute relevance - a #1
result is #1 whether it's a great match or the least-bad of 808 bad ones.

TWO CANDIDATE GATES, measured side by side:

  cosine  - the bi-encoder score from Qdrant. Free (already computed), but it
            compares a pre-made summary of the chunk against the question, so
            it can be fooled by shared vocabulary. Measured first and it did
            NOT separate cleanly: "side effects of chemotherapy" scored 0.7521
            (it matched generic "higher chance of getting infections" text)
            while the real question "I get shaky and sweaty between meals"
            scored only 0.7078.

  cross-encoder - reads the question and the chunk TOGETHER, so it judges
            whether the text actually answers the question rather than whether
            the words look alike. Slower (a model pass per pair) but far
            sharper. Note the sigmoid saturation that ruined RANKING is
            harmless here: a gate only needs above-or-below a line, and
            saturation pushes clear matches to ~1.0 and clear misses to ~0.

Requires the Qdrant container to be running.
Run: python niddk_calibrate_threshold.py
"""

from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

from niddk_index_qdrant import COLLECTION_NAME, QDRANT_URL, BGEEmbeddings

RERANKER_MODEL = "BAAI/bge-reranker-base"

# How many chunks the cross-encoder looks at before deciding. We take the best
# of these - if none of the top few are relevant, nothing deeper will be.
GATE_CANDIDATES = 5

# Questions the corpus SHOULD be able to answer.
IN_SCOPE = [
    "How can diabetes affect my feet?",
    "What is the A1C test?",
    "Can diabetes damage my kidneys?",
    "Is gestational diabetes dangerous for my baby?",
    "What are the symptoms of type 1 diabetes?",
    "Why do my gums bleed when I brush?",
    "I get shaky and sweaty between meals. What is that?",
    "Should I stop smoking if I'm diabetic?",
    "Can you transplant cells to cure type 1 diabetes?",
    "What A1C number should I be aiming for?",
]

# Questions the corpus should REFUSE. A mix on purpose:
#   - completely unrelated
#   - medical but not diabetes
#   - diabetes-adjacent but genuinely not in these 34 pages
OUT_OF_SCOPE = [
    "What is the best treatment for a broken arm?",
    "How do I change the oil in my car?",
    "What are the symptoms of appendicitis?",
    "Who won the World Cup in 2022?",
    "How do I treat a sprained ankle?",
    "What is the recommended dose of ibuprofen for a headache?",
    "How do I bake sourdough bread?",
    "What are the side effects of chemotherapy?",
    "Can you write me a Python function to sort a list?",
    "What is the capital of Australia?",
]


def score_question(
    store: QdrantVectorStore, cross_encoder: CrossEncoder, question: str
) -> tuple[float, float, str]:
    """Return (best cosine, best cross-encoder score, snippet of best chunk)."""
    hits = store.similarity_search_with_score(question, k=GATE_CANDIDATES)
    if not hits:
        return 0.0, 0.0, ""

    best_cosine = float(hits[0][1])

    pairs = [(question, doc.page_content) for doc, _ in hits]
    ce_scores = cross_encoder.predict(pairs)
    best_idx = int(max(range(len(ce_scores)), key=lambda i: ce_scores[i]))
    best_ce = float(ce_scores[best_idx])

    snippet = hits[best_idx][0].page_content[:44].replace("\n", " ")
    return best_cosine, best_ce, snippet


def report(label: str, lo_in: float, hi_out: float) -> None:
    print(f"\n  {label}")
    print(f"    lowest in-scope   : {lo_in:.4f}")
    print(f"    highest out-scope : {hi_out:.4f}")
    if lo_in > hi_out:
        mid = (lo_in + hi_out) / 2
        print(f"    CLEAN SEPARATION - gap {lo_in - hi_out:.4f}, threshold {mid:.4f}")
    else:
        print(f"    OVERLAP of {hi_out - lo_in:.4f} - no cutoff works cleanly")


if __name__ == "__main__":
    print("Connecting to Qdrant...")
    store = QdrantVectorStore.from_existing_collection(
        embedding=BGEEmbeddings(),
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    print(f"Loading {RERANKER_MODEL} ...")
    cross_encoder = CrossEncoder(RERANKER_MODEL, device="cpu")
    print()

    results = {}
    for label, questions in [("IN SCOPE", IN_SCOPE), ("OUT OF SCOPE", OUT_OF_SCOPE)]:
        print("=" * 82)
        print(f"{label}    (cosine / cross-encoder)")
        print("=" * 82)
        cos_scores, ce_scores = [], []
        for q in questions:
            cos, ce, snippet = score_question(store, cross_encoder, q)
            cos_scores.append(cos)
            ce_scores.append(ce)
            print(f"  cos {cos:.4f} | ce {ce:.4f}   {q[:40]:<42s} -> {snippet!r}")
        results[label] = {"cosine": cos_scores, "cross_encoder": ce_scores}
        print()

    print("=" * 82)
    print("WHERE TO PUT THE THRESHOLD")
    print("=" * 82)
    for metric in ("cosine", "cross_encoder"):
        report(
            metric,
            min(results["IN SCOPE"][metric]),
            max(results["OUT OF SCOPE"][metric]),
        )
    print()
    print("  Use whichever separates cleanly. If both do, prefer cosine - it is")
    print("  already computed, so the gate costs nothing extra.")
