"""PDF ingestion: parse, chunk, embed, and store in Chroma with source metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def parse_pdf(file_path: str | Path, filename: str | None = None) -> list[Document]:
    """Extract one Document per page with filename + page metadata."""
    path = Path(file_path)
    name = filename or path.name
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": name, "page": page_num},
            )
        )
    return docs


def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(list(docs))


def ingest_pdf(file_path: str | Path, filename: str | None = None) -> int:
    """Parse, chunk, embed, and store. Returns number of chunks added."""
    pages = parse_pdf(file_path, filename=filename)
    if not pages:
        return 0
    chunks = chunk_documents(pages)
    if not chunks:
        return 0
    store = get_vectorstore()
    store.add_documents(chunks)
    return len(chunks)


def reset_knowledge_base() -> None:
    """Drop all vectors from the collection."""
    store = get_vectorstore()
    store.delete_collection()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-pdf> [<path-to-pdf> ...]")
        sys.exit(1)

    total = 0
    for arg in sys.argv[1:]:
        n = ingest_pdf(arg)
        print(f"  {arg}: {n} chunks")
        total += n
    print(f"Done. {total} chunks added to {CHROMA_DIR}/")
