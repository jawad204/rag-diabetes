"""
Splits each scraped NIDDK page into topic-scoped chunks.

Stage 1: MarkdownHeaderTextSplitter splits each page on its markdown headers,
so each chunk stays scoped to one section (and the header text becomes metadata).

Stage 2: any resulting section that's still longer than CHUNK_SIZE gets
further split by RecursiveCharacterTextSplitter, with the header/page
metadata carried over to every sub-chunk.

Run: python niddk_chunk.py
"""

import json
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

INPUT_PATH = Path("data/niddk_pages.jsonl")
OUTPUT_PATH = Path("data/niddk_chunks.jsonl")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 75

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def load_pages(path: Path) -> list[dict]:
    pages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep header text inside page_content too
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []

    for page in pages:
        # Stage 1: split on headers. Each result is a Document with
        # page_content + metadata like {"Header 2": "What are the symptoms?"}
        header_docs = header_splitter.split_text(page["content"])

        for header_doc in header_docs:
            # attach page-level metadata (the header splitter has no idea
            # which page it's working on)
            base_metadata = {
                "url": page["url"],
                "title": page["title"],
                **header_doc.metadata,
            }

            if len(header_doc.page_content) <= CHUNK_SIZE:
                all_chunks.append(
                    {"content": header_doc.page_content, "metadata": base_metadata}
                )
            else:
                # Stage 2: this section is still too long, split further.
                sub_texts = size_splitter.split_text(header_doc.page_content)
                for sub_text in sub_texts:
                    all_chunks.append(
                        {"content": sub_text, "metadata": dict(base_metadata)}
                    )

    # give every chunk a stable id + index within its source page,
    # useful later for citations / debugging
    per_url_counter: dict[str, int] = {}
    for chunk in all_chunks:
        url = chunk["metadata"]["url"]
        idx = per_url_counter.get(url, 0)
        chunk["metadata"]["chunk_index"] = idx
        chunk["id"] = f"{url}#chunk-{idx}"
        per_url_counter[url] = idx + 1

    return all_chunks


if __name__ == "__main__":
    pages = load_pages(INPUT_PATH)
    print(f"Loaded {len(pages)} pages from {INPUT_PATH}")

    chunks = chunk_pages(pages)
    print(f"Produced {len(chunks)} chunks (avg {len(chunks) / len(pages):.1f} per page)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved to {OUTPUT_PATH}")

    # sanity check: print a couple of chunks so you can eyeball quality
    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS")
    print("=" * 60)
    for c in chunks[:3]:
        print(f"\n--- {c['id']} ---")
        print(f"metadata: {c['metadata']}")
        print(f"content ({len(c['content'])} chars):")
        print(c["content"][:300])