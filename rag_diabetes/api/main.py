"""
FastAPI service wrapping the RAG pipeline, plus a small static frontend.

    GET  /            -> the ask-a-question page (static/index.html)
    GET  /static/*    -> css + js
    POST /ask         -> {"question": "..."} -> answer + sources
    GET  /health      -> liveness + whether the pipeline is loaded

WHY THE LIFESPAN HANDLER:
Building DiabetesRAG loads the bge embedding model and rebuilds the BM25 index
over all 808 chunks - several seconds. That happens ONCE when the server
starts, not per request. Building it inside the endpoint would make every
question pay that cost.

SETUP
-----
    pip install fastapi uvicorn
    # .env at the project root must contain ANTHROPIC_API_KEY
    # the Qdrant container must be running

RUN (from the project root)
---
    uvicorn rag_diabetes.api.main:app --reload
    # then open http://127.0.0.1:8000
    # interactive API docs at /docs

--reload restarts on file changes, which re-loads the models each time.
Drop it once you stop editing.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_diabetes.generation.answer import DiabetesRAG

STATIC_DIR = Path(__file__).resolve().parent / "static"

# populated once at startup by the lifespan handler
rag: DiabetesRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    print("Loading RAG pipeline (embedding model + BM25 index + Qdrant)...")
    rag = DiabetesRAG()
    print("Ready.")
    yield
    rag = None


app = FastAPI(
    title="Diabetes RAG",
    description="Patient-friendly diabetes answers grounded in NIDDK sources.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Source(BaseModel):
    n: int
    title: str
    url: str
    category: str


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    relevance_score: float
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": rag is not None}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = rag.ask(req.question)
    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        refused=result["refused"],
        relevance_score=result["relevance_score"],
        sources=[Source(**s) for s in result["sources"]],
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
