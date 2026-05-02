# PDF Research Assistant

A local-first chat app for asking questions about your PDFs, with every claim
cited back to the source filename and page number.

Upload PDFs in the Streamlit sidebar; the app extracts text, chunks it
(500 tokens / 50 overlap), embeds with `all-MiniLM-L6-v2`, and persists vectors
in a local ChromaDB store. Each query retrieves the top-5 relevant chunks and
passes them to Claude with a strict system prompt that forbids outside
knowledge — if the answer isn't in the retrieved context, the assistant says so
instead of guessing.

## Features
- Multi-PDF upload with per-page metadata (filename + page number)
- Persistent vector store (`chroma_db/`) — survives restarts
- Conversation memory across turns
- Inline `[filename, p.N]` citations on every factual claim
- "Sources" expander under each answer showing the exact retrieved chunks
- One-click "Clear knowledge base" reset

## Stack
- **Orchestration:** LangChain
- **Vector store:** ChromaDB (local, persistent)
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2`
- **LLM:** Anthropic Claude (`claude-sonnet-4-5`)
- **PDF parsing:** pypdf
- **UI:** Streamlit

## Quick start
```bash
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY
streamlit run app.py
