"""
Builds the prompt for the generation step - and just PRINTS it.

No LLM is called here on purpose. This lets you see exactly what context
gets sent, with your real retrieved chunks, before committing to any
provider or API key. Once the prompt looks right, sending it to a model
is about five more lines.

The prompt has three parts:
  1. INSTRUCTIONS - the rules the model must follow
  2. SOURCES     - your top 5 retrieved chunks, pasted in as text
  3. QUESTION    - what the patient asked

Part 2 is the whole point of RAG: the model has never seen your NIDDK
content, so you paste it in and tell the model to answer from that alone.

Requires the Qdrant container to be running.
Run: python niddk_prompt.py
"""

from langchain_core.documents import Document

from niddk_hybrid_retriever import build_hybrid_retriever

SYSTEM_INSTRUCTIONS = """You answer diabetes questions for patients.

Rules:
- Use ONLY the numbered sources below. Do not add anything from your own
  knowledge, even if you are confident it is correct.
- If the sources do not answer the question, say so plainly and stop.
  Do not fill the gap with general knowledge.
- Write in plain language a non-medical reader understands. Short
  sentences. Explain any medical term you have to use.
- Cite the source number after each claim, like [2].
- End with: "This is general information, not medical advice. Talk to
  your health care professional about your own situation."
"""

TEST_QUESTION = "How can diabetes affect my feet?"

# How many retrieved chunks to put in the prompt.
TOP_K = 5


def format_sources(docs: list[Document]) -> str:
    """Turn retrieved chunks into a numbered SOURCES block.

    The title + url come from the metadata built back in
    niddk_metadata.py - this is where that metadata earns its keep,
    because it becomes the citation the patient can click.
    """
    blocks = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "NIDDK")
        url = doc.metadata.get("url", "")
        blocks.append(f"[{i}] {title}\n{url}\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_prompt(question: str, docs: list[Document]) -> str:
    return (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"{'=' * 70}\n"
        f"SOURCES\n"
        f"{'=' * 70}\n\n"
        f"{format_sources(docs)}\n\n"
        f"{'=' * 70}\n"
        f"QUESTION\n"
        f"{'=' * 70}\n\n"
        f"{question}\n"
    )


if __name__ == "__main__":
    print("Building retriever...")
    # Hybrid only - NO reranking. Measured on the 18-question eval set:
    #   hybrid only      page recall 0.944   page MRR 0.780
    #   hybrid + rerank  page recall 0.833   page MRR 0.667
    # Reranking made retrieval worse on every metric, so it is off.
    # See issue 1 in the project notes (score saturation) before re-enabling.
    retriever = build_hybrid_retriever()

    print(f"Retrieving for: {TEST_QUESTION}\n")
    docs = retriever.invoke(TEST_QUESTION)[:TOP_K]

    prompt = build_prompt(TEST_QUESTION, docs)

    print("#" * 70)
    print("# THIS IS WHAT WOULD BE SENT TO THE LLM")
    print("#" * 70)
    print()
    print(prompt)
    print("#" * 70)
    print(f"# prompt length: {len(prompt)} characters, {len(docs)} sources")
    print("#" * 70)
