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

from rag_diabetes.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    MIN_BODY_CHARS,
    PAGES_PATH,
)

INPUT_PATH = PAGES_PATH
OUTPUT_PATH = CHUNKS_PATH

# chunks whose content is nothing but a header line / ToC stub (e.g. a
# section that's empty in the source, or "**On this page:**" leftovers)
# get dropped after chunking. See MIN_BODY_CHARS below.

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def body_length(text: str) -> int:
    """Length of the content with markdown header lines (#, ##, ...) removed.

    A chunk that's nothing but a header line has body_length 0 even though
    len(text) isn't - that's exactly the case we want to catch and drop.
    """
    non_header_lines = [
        line for line in text.split("\n") if not line.lstrip().startswith("#")
    ]
    return len(" ".join(non_header_lines).strip())


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

    # drop chunks that are just a header line with no real body text -
    # e.g. a genuinely empty section in the source page, or a "**On this
    # page:**" / "View or Print All Sections" ToC leftover. These carry no
    # retrievable information (any real content for that header, if it
    # exists, is captured in the sibling chunks that share the same
    # metadata, so nothing is lost by dropping them).
    before = len(all_chunks)
    all_chunks = [c for c in all_chunks if body_length(c["content"]) >= MIN_BODY_CHARS]
    dropped = before - len(all_chunks)
    if dropped:
        print(f"Dropped {dropped} header-only/empty stub chunks (body < {MIN_BODY_CHARS} chars)")

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