
# RAG-Powered Research Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions using a Ready Tensor publications dataset.
It uses LangChain + ChromaDB + HuggingFace embeddings and Google Gemini (gemini-1.5-flash-latest).
A simple Streamlit UI is included.

## Quickstart

1. Clone repo
2. Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`
3. Install dependencies:
Run:
## Structure
- `project1.py` — Streamlit app + RAG pipeline (entrypoint)
- `data/project_1_publications.json` — dataset
- `src/` — helper modules (loader, retriever, pipeline)
- `docs/` — diagrams & screenshots
