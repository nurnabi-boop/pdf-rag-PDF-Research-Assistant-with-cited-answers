"""Streamlit chat UI for the PDF Research Assistant."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from chat import answer
from ingest import ingest_pdf, reset_knowledge_base
from retriever import collection_size, retrieve

load_dotenv()

st.set_page_config(page_title="PDF Research Assistant", page_icon=":books:", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: role, content, sources?
if "ingested" not in st.session_state:
    st.session_state.ingested = set()


def _ingest_uploads(uploaded_files) -> None:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.ingested]
    if not new_files:
        return
    progress = st.sidebar.progress(0.0, text="Ingesting...")
    for i, uf in enumerate(new_files, start=1):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = Path(tmp.name)
        try:
            n_chunks = ingest_pdf(tmp_path, filename=uf.name)
            st.session_state.ingested.add(uf.name)
            st.sidebar.write(f"+ {uf.name}: {n_chunks} chunks")
        finally:
            tmp_path.unlink(missing_ok=True)
        progress.progress(i / len(new_files), text=f"Ingesting {uf.name}...")
    progress.empty()


with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploads:
        _ingest_uploads(uploads)

    st.divider()
    st.caption(f"Vectors in store: {collection_size()}")
    if st.button("Clear knowledge base", type="secondary"):
        reset_knowledge_base()
        st.session_state.ingested.clear()
        st.session_state.messages.clear()
        st.rerun()

st.title("PDF Research Assistant")
st.caption("Ask questions about your uploaded PDFs. Answers cite filename and page.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**[{s['source']}, p.{s['page']}]**\n\n> {s['text']}"
                    )

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if collection_size() == 0:
        with st.chat_message("assistant"):
            warning = "No documents ingested yet. Upload at least one PDF in the sidebar."
            st.warning(warning)
        st.session_state.messages.append({"role": "assistant", "content": warning})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                docs = retrieve(prompt, k=5)

            history = [
                (m["content"], st.session_state.messages[i + 1]["content"])
                for i, m in enumerate(st.session_state.messages[:-1])
                if m["role"] == "user"
                and i + 1 < len(st.session_state.messages)
                and st.session_state.messages[i + 1]["role"] == "assistant"
            ]

            with st.spinner("Thinking..."):
                reply = answer(prompt, docs, history=history)

            st.markdown(reply)

            sources = [
                {
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page", "?"),
                    "text": d.page_content,
                }
                for d in docs
            ]
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**[{s['source']}, p.{s['page']}]**\n\n> {s['text']}")

        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "sources": sources}
        )
