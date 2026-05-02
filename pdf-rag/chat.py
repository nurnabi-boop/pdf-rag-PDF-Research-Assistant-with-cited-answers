"""Claude API call with strict context-only prompt and inline citations."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a research assistant that answers questions about uploaded PDFs.

Strict rules:
- Answer ONLY using information found in the CONTEXT below.
- If the answer is not contained in the context, say exactly:
  "I don't have enough information in the provided documents to answer that."
- Cite every factual claim inline using the format [filename, p.N], where filename
  and page number come from the source markers in the context.
- Multiple sources for one claim: combine like [a.pdf, p.2; b.pdf, p.7].
- Quote sparingly; prefer your own concise summary.
- Do not use outside knowledge, even if you are confident."""


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "(no documents retrieved)"
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[Source {i} | {src}, p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _get_llm() -> ChatAnthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)


def answer(
    question: str,
    context_docs: list[Document],
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Generate an answer grounded in context_docs.

    history is a list of (user, assistant) pairs from earlier turns.
    """
    llm = _get_llm()

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for user_msg, ai_msg in history or []:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=ai_msg))

    user_turn = (
        f"CONTEXT:\n{_format_context(context_docs)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the context above, with inline [filename, p.N] citations."
    )
    messages.append(HumanMessage(content=user_turn))

    response = llm.invoke(messages)
    return response.content if isinstance(response.content, str) else str(response.content)
