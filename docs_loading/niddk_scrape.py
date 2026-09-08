"""
Fetches each NIDDK page and extracts clean article text (stripping nav,
footer, share buttons, etc.) using trafilatura.

Run: python scrape_niddk.py
"""

import json
import time
from pathlib import Path

import requests
import trafilatura
from tqdm import tqdm

from niddk_urls import NIDDK_DIABETES_URLS

OUTPUT_PATH = Path("data/niddk_pages.jsonl")
HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio RAG project; educational use)"}


def scrape_all():
    records = []

    for url in tqdm(NIDDK_DIABETES_URLS, desc="Scraping"):
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"  SKIP (status {resp.status_code}): {url}")
            continue

        content = trafilatura.extract(
            resp.text,
            output_format="markdown",
            include_links=False,
            include_images=False,
        )
        if not content:
            print(f"  SKIP (no content extracted): {url}")
            continue

        meta = trafilatura.extract_metadata(resp.text)
        title = meta.title if meta and meta.title else url.split("/")[-1]

        records.append({"url": url, "title": title, "content": content})
        time.sleep(1)  # be polite to a government server

    return records


if __name__ == "__main__":
    records = scrape_all()
    print(f"\nScraped {len(records)} / {len(NIDDK_DIABETES_URLS)} pages successfully.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved to {OUTPUT_PATH}")

    if records:
        print("\n" + "=" * 60)
        print(f"SAMPLE PAGE: {records[0]['title']}")
        print("=" * 60)
        print(records[0]["content"][:1500])