# Project structure

```
rag-diabetes/
├── data/                          # unchanged: jsonl corpus + eval sets
├── rag_diabetes/
│   ├── config.py                  # ALL paths and constants
│   ├── ingestion/                 # one-time setup, run in order
│   │   ├── urls.py                # 34 NIDDK URLs
│   │   ├── scrape.py              # -> data/niddk_pages.jsonl
│   │   ├── chunk.py               # -> data/niddk_chunks.jsonl   (808 chunks)
│   │   └── metadata.py            # -> data/niddk_chunks_final.jsonl
│   ├── retrieval/
│   │   ├── vector_store.py        # BGEEmbeddings + Qdrant index/connect
│   │   ├── sparse.py              # BM25 + the fixed tokenizer
│   │   ├── hybrid.py              # EnsembleRetriever, RRF, 0.7/0.3
│   │   ├── rerank.py              # LangChain-native (measured, NOT used)
│   │   └── rerank_debug.py        # hand-rolled, prints scores + "was #N"
│   ├── generation/
│   │   ├── prompt.py              # SCOPE rules + source formatting
│   │   ├── guardrails.py          # threshold, meta questions, refusal,
│   │   │                          #   citation validation
│   │   └── answer.py              # DiabetesRAG - the entry point
│   └── api/
│       ├── main.py                # FastAPI only, no HTML
│       └── static/
│           ├── index.html
│           ├── style.css
│           └── app.js
└── tests/
    ├── eval_retrieval.py          # recall@5 + MRR, chunk and page level
    ├── calibrate_threshold.py     # finds the relevance cutoff
    └── embed_test.py              # embedding sanity check
```

## Running things

All commands from the **project root**.

```bash
# one-time setup (in order)
python -m rag_diabetes.ingestion.scrape
python -m rag_diabetes.ingestion.chunk
python -m rag_diabetes.ingestion.metadata
python -m rag_diabetes.retrieval.vector_store     # embeds + indexes into Qdrant

# ask a question
python -m rag_diabetes.generation.answer "Can diabetes damage my kidneys?"

# web app
uvicorn rag_diabetes.api.main:app --reload

# evaluation
python -m tests.eval_retrieval
python -m tests.eval_retrieval data/eval_questions_set2.json
python -m tests.calibrate_threshold
```

`python -m` matters: it runs the file as part of the package, so
`from rag_diabetes...` imports resolve. Running `python rag_diabetes/api/main.py`
directly will fail with ModuleNotFoundError.

## What changed and why

- **`config.py`** — paths were relative strings resolved against the current
  working directory, so scripts broke depending on where you ran them. Paths
  are now derived from the package location.
- **`guardrails.py`** — scope control was constants scattered through
  `answer.py`. It's an architectural layer, so it's now a named module.
- **No `sys.path` hack** — the API used to insert its own folder onto the path.
  Real package imports made that unnecessary.
- **Frontend split into `static/`** — the page was a Python string. Now it's
  html/css/js, editable with syntax highlighting.
- **Dropped the `niddk_` prefix** — redundant inside a package named
  `rag_diabetes`.
