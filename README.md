# Diabetes RAG — patient-friendly answers grounded in NIDDK sources

A retrieval-augmented generation system that answers diabetes questions in plain language,
using only the National Institute of Diabetes and Digestive and Kidney Diseases
([NIDDK](https://www.niddk.nih.gov/health-information/diabetes)) patient-education pages,
with a citation for every claim.

It refuses anything it can't ground in those pages — including medical questions that aren't
about diabetes. That constraint is the point of the project, and it's enforced in two
independent layers.

```
question
   │
   ├─▶ meta-question?  ──────────────▶ list of supported topics
   │
   ├─▶ relevance gate (cosine ≥ 0.60)
   │        └─ below ─────────────────▶ refusal   (no retrieval, no API call, no cost)
   │
   ├─▶ hybrid retrieval  ─ dense (Qdrant / bge-small) 0.7
   │                     └ sparse (BM25)              0.3   ─▶ RRF ─▶ top 5 chunks
   │
   └─▶ prompt with SCOPE rules ─▶ Claude Haiku 4.5 ─▶ cited answer
                                        └─ off-topic despite passing the gate ─▶ refusal
```

**Corpus:** 34 NIDDK pages → 808 chunks.

---

## Measured results

Retrieval was evaluated on **two independent, non-overlapping labelled question sets**
(18 and 20 questions), written in patient phrasing rather than copied from page headers.

| configuration | set 1 page recall@5 | set 1 page MRR | set 2 page recall@5 | set 2 page MRR |
|---|---|---|---|---|
| **hybrid only** (shipped) | **0.944** | **0.780** | **0.950** | **0.758** |
| hybrid + cross-encoder rerank | 0.833 | 0.667 | 0.900 | 0.747 |

Metric implementations (recall@k, MRR) are unit-tested against seven hand-computed cases.

Scope control was verified end to end:

| test question | relevance score | outcome |
|---|---|---|
| "What is the capital of Australia?" | 0.4280 | refused at the gate — no API call |
| "What are the side effects of chemotherapy?" | 0.7521 | passed the gate, refused by the prompt |

---

## Findings worth reading

These are the results that changed the design. Each one is measured, not assumed.

### 1. LangChain's default BM25 tokenizer was silently destroying sparse retrieval

`BM25Retriever`'s default `preprocess_func` is `text.split()` — no lowercasing, no punctuation
stripping, no stopword removal. In this corpus that meant the query token `'diabetes?'`
(with the question mark attached) matched **74 of 808 chunks**, while `'diabetes'` matched
**430**. Meanwhile stopwords like `'have'` (328 chunks) dominated the scoring.

Replaced with: lowercase → `re.findall(r"[a-z0-9]+")` → drop stopwords. Digits and the word
`type` are deliberately **kept**, because "type 1" and "type 2" are the most important
distinction in the corpus.

Verified by reimplementing BM25 by hand in NumPy and comparing term-document counts directly.

### 2. Reranking looked great on spot checks and lost on measurement

A `bge-reranker-base` cross-encoder over the top 25 hybrid candidates looked convincing on four
hand-picked queries. Across 38 labelled questions it **lost 3 of the 4 comparisons**, while
adding a 1.1 GB model and seconds of latency per query.

It's built, it's committed (`retrieval/rerank.py`), and it's **switched off**. This is the
clearest argument in the project for having an eval set: spot checks said ship it, measurement
said don't.

*Suspected cause, untested:* `bge-reranker-base` squashes scores through a sigmoid, so all
strongly-relevant chunks collapse to 1.00 and the ordering becomes an arbitrary tiebreak.
Raw logits would likely fix it.

### 3. A cross-encoder makes a bad relevance gate

Before settling on cosine similarity for the input gate, a cross-encoder was tested as the
scorer. It was far worse: **0.497 score overlap** between in-scope and out-of-scope questions,
versus cosine's **0.044**.

The failure is instructive. It scored *"best treatment for a broken arm"* at **0.93** against
*"cover a blister, cut, or sore with a bandage"* — because a cross-encoder judges whether a
passage **reads like a response** to a query, not whether it's on the right **topic**.

### 4. Read page-level recall, not chunk-level

Chunk-level recall swung from 0.667 to 0.450 between the two question sets **on the identical
pipeline**, purely because set 2 labelled deeper mid-page chunks. A metric that moves 0.22
based on which chunk a human happened to pick is measuring labelling, not retrieval.
Page-level recall stayed at 0.944 / 0.950 across both.

### 5. One guardrail layer isn't enough

The chemotherapy question scores **0.7521** — above the gate — because diabetes pages contain
generic text about *"a higher chance of getting infections"*. Cosine similarity cannot tell
those topics apart. Claude reading the actual retrieved chunks can, and refuses it.

Cosine catches the cheap cases for free; the prompt catches the semantically-close ones.
Neither layer alone is sufficient.

### 6. `0.60` is not "60%"

The relevance threshold is a **cosine similarity**, not a percentage or a confidence. It was
calibrated empirically on 10 in-scope vs 10 out-of-scope questions
(`tests/calibrate_threshold.py`), then lowered from 0.70 to 0.60 after 0.70 over-refused real
questions. The number is specific to this embedding model and this corpus — **re-run the
calibration if either changes.**

---

## Setup

Requires Python 3.11+ and Docker.

```bash
git clone git@github.com:jawad204/rag-diabetes.git
cd rag-diabetes

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
pip install -e .                 # makes `rag_diabetes` importable from anywhere
```

Start the vector database:

```bash
docker run -d --name qdrant-diabetes -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Dashboard at <http://localhost:6333/dashboard>.

Create `.env` at the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

> The key must be **workspace-scoped**. An organization-scoped / Admin API key does *not* work
> with the Messages API and fails with a misleading `401 API key is invalid`.

## Running it

```bash
# one-time: build the vector index (34 pages are already scraped into data/)
python -m rag_diabetes.retrieval.vector_store

# the web app
uvicorn rag_diabetes.api.main:app --reload
# → http://127.0.0.1:8000     (API docs at /docs)
```

Rebuilding the corpus from scratch is optional — the scraped output is committed:

```bash
python -m rag_diabetes.ingestion.scrape     # 34 URLs  → data/niddk_pages.jsonl
python -m rag_diabetes.ingestion.chunk      # 841 → 808 chunks after stub filtering
python -m rag_diabetes.ingestion.metadata   # adds category / source / word_count
```

## Evaluation

```bash
python -m tests.eval_retrieval                                   # question set 1
python -m tests.eval_retrieval data/eval_questions_set2.json     # question set 2
python -m tests.calibrate_threshold                              # relevance gate calibration
```

---

## Project layout

Modules are grouped by **when they run**, not by what they're named.

```
rag_diabetes/
├── config.py            all tunable constants in one place; paths derived from
│                        __file__ so nothing depends on the working directory
│
├── ingestion/           runs once, offline
│   ├── urls.py          34 curated NIDDK URLs
│   ├── scrape.py        requests + trafilatura → markdown, nav/footer/images stripped
│   ├── chunk.py         header-aware split, then 600-char split, then stub filter
│   └── metadata.py      category (12 URL-derived buckets), source, word_count
│
├── retrieval/           runs per question
│   ├── vector_store.py  bge-small-en-v1.5 embeddings → Qdrant
│   ├── sparse.py        BM25 with the corrected tokenizer
│   ├── hybrid.py        RRF fusion, 0.7 dense / 0.3 sparse
│   ├── rerank.py        cross-encoder reranker — built, measured, OFF
│   └── rerank_debug.py  hand-rolled version that prints scores and rank movement
│
├── generation/          runs per question
│   ├── guardrails.py    relevance gate, refusal text, meta-question handling
│   ├── prompt.py        SCOPE rules + citation format + grounding instructions
│   └── answer.py        DiabetesRAG.ask() — the single entry point
│
└── api/
    ├── main.py          FastAPI; the pipeline loads once at startup, not per request
    └── static/          vanilla HTML / CSS / JS frontend

tests/                   measurement only — never imported by the pipeline
data/                    scraped pages, chunks, and two labelled eval question sets
```

### Design notes

- **Chunking** is two-stage: `MarkdownHeaderTextSplitter` first, so every chunk stays scoped to
  one section and its headers survive as metadata; then `RecursiveCharacterTextSplitter`
  (600 characters, 75 overlap) only on sections that are still too long. A `body_length`
  filter then drops header-only and table-of-contents stubs — 841 chunks in, 808 out.
- **RRF** fuses the two retrievers by `weight / (60 + rank)`, summed. Raw scores are discarded
  because BM25 (~9.0) and cosine (~0.75) aren't on comparable scales. The `+60` flattens
  within-list differences, so appearing in **both** lists matters more than ranking high in one.
- **Embeddings** use `BAAI/bge-small-en-v1.5` with the query-side instruction prefix
  (`"Represent this sentence for searching relevant passages: "`) applied to queries only,
  never to documents — that's how the model was trained.
- Dense retrieval covers the vocabulary gaps that matter for patients: *"Why do my gums bleed
  when I brush?"* returns the right chunk at #1, even though the page says "mouth problems";
  *"I'm leaking urine"* hits "bladder problems".

---

## Known limitations

1. **Answer quality is unmeasured.** Retrieval has 38 labelled questions behind it; generation
   has a handful of hand-inspected outputs. The next step is an LLM-as-judge pass scoring
   faithfulness, citation accuracy, refusal behaviour and readability — spot-checked by hand to
   confirm the judge isn't nonsense.
2. **Near-duplicate chunks waste top-5 slots.** One run returned 5 sources that were only 3
   distinct pages. Reranking does not fix this — a cross-encoder scores each chunk in isolation.
   It needs its own dedup pass over the ranked list.
3. **The output guardrail isn't wired in.** `find_invalid_citations()` exists in
   `guardrails.py` and is never called.
4. **Chunk-level eval labels are too sparse.** Labelling 2–3 acceptable chunks per question
   would make the chunk-level metric meaningful.
5. **Corpus gaps are not a retrieval problem.** Only 16 chunks contain the word "avoid" —
   NIDDK frames nutrition positively. No amount of tuning retrieves content that isn't there.
6. **Single-turn only.** No follow-up questions, no conversation memory.

## Stack

LangChain · Qdrant (Docker) · sentence-transformers (`bge-small-en-v1.5`) · rank_bm25 ·
Claude Haiku 4.5 · FastAPI · trafilatura · vanilla JS

## Disclaimer

Educational portfolio project. Not medical advice. Answers are generated from public NIDDK
patient-education material and may be incomplete or wrong — talk to a healthcare professional.
