"""
Enriches each chunk in niddk_chunks.jsonl with extra metadata:
  - category   : coarse topic bucket derived from the source URL
  - source     : constant "NIDDK" (future-proofing for a second source)
  - word_count : cheap length proxy, useful later for context budgeting

Run: python niddk_metadata.py
"""

import json
from pathlib import Path

INPUT_PATH = Path("data/niddk_chunks.jsonl")
OUTPUT_PATH = Path("data/niddk_chunks_final.jsonl")

SOURCE = "NIDDK"

# Checked in order, first match wins. Each rule is a URL substring ->
# category label. Ordered most-specific first (e.g. gestational sub-pages
# before the general "what-is-diabetes" bucket) so a page only ever
# matches the most precise rule that applies to it.
CATEGORY_RULES = [
    ("/what-is-diabetes/type-1-diabetes", "type-1"),
    ("/what-is-diabetes/type-2-diabetes", "type-2"),
    ("/what-is-diabetes/prediabetes-insulin-resistance", "prediabetes"),
    ("/what-is-diabetes/gestational", "gestational"),
    ("/what-is-diabetes/monogenic-neonatal-mellitus-mody", "monogenic"),
    ("/risk-factors-type-2-diabetes", "type-2"),
    ("/preventing-type-2-diabetes", "prevention"),
    ("/tests-diagnosis", "diagnosis"),
    ("/diagnostic-tests/a1c-test", "diagnosis"),
    ("/managing-diabetes", "management"),
    ("/insulin-medicines-treatments", "management"),
    ("/healthy-living-with-diabetes", "management"),
    ("/preventing-problems", "complications"),
    ("/diabetes-pregnancy", "pregnancy"),
    ("/financial-help-diabetes-care", "financial"),
    ("/what-is-diabetes", "overview"),  # catch-all for the root page itself
    ("/symptoms-causes", "overview"),
]


def categorize(url: str) -> str:
    for substring, category in CATEGORY_RULES:
        if substring in url:
            return category
    return "uncategorized"


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def enrich(chunks: list[dict]) -> list[dict]:
    for chunk in chunks:
        url = chunk["metadata"]["url"]
        chunk["metadata"]["category"] = categorize(url)
        chunk["metadata"]["source"] = SOURCE
        chunk["metadata"]["word_count"] = len(chunk["content"].split())
    return chunks


if __name__ == "__main__":
    chunks = load_chunks(INPUT_PATH)
    print(f"Loaded {len(chunks)} chunks from {INPUT_PATH}")

    chunks = enrich(chunks)

    # sanity check: report how many chunks landed in each category, and
    # flag anything that fell through to "uncategorized" so you can catch
    # a URL the rules don't cover
    from collections import Counter

    counts = Counter(c["metadata"]["category"] for c in chunks)
    print("\nChunks per category:")
    for category, n in sorted(counts.items(), key=lambda x: -x[1]):
        flag = "  <-- check this" if category == "uncategorized" else ""
        print(f"  {category:15s} {n:4d}{flag}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\nSample enriched chunk:")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2)[:600])
