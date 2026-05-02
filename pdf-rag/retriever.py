"""Chroma retrieval: top-k similarity search over the persisted vector store."""

from __future__ import annotations

from langchain_core.documents import Document

from ingest import get_vectorstore

DEFAULT_K = 5


def retrieve(query: str, k: int = DEFAULT_K) -> list[Document]:
    """Return the top-k most relevant chunks for the query."""
    if not query.strip():
        return []
    return get_vectorstore().similarity_search(query, k=k)


def collection_size() -> int:
    """Number of vectors currently in the store. 0 means empty / reset."""
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0
