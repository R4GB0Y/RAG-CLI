"""Generation: synthesize an answer from the retrieved chunks — and only those.

This is the "G" in RAG. By the time we're here, the retriever has already
decided the question is answerable (its best chunk cleared the threshold), so
our job is narrow: write a natural-language answer *grounded strictly in the
provided chunks*. The single most important property — and a hard "Never" in
the SPEC — is that the model must not answer from its own training knowledge.
For medical content, a confident-sounding answer invented outside the source
documents is exactly the failure mode we must design against.

Two design choices enforce grounding:
  * A **system prompt** that explicitly restricts the model to the context and
    tells it to say it doesn't know when the context is insufficient.
  * **temperature=0** for the most deterministic, least "creative" output — we
    want faithful synthesis, not invention.
"""
from __future__ import annotations

from openai import OpenAI

from src.config import load_config

# The grounding contract, stated to the model. Kept strict and explicit: answer
# ONLY from context, admit ignorance rather than guess, don't rely on training.
_SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about a specific set of "
    "documents. Use ONLY the information in the provided context to answer. "
    "Do not use any outside or prior knowledge. If the context does not contain "
    "enough information to answer, say you don't know rather than guessing. "
    "Be concise and factual."
)


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single labelled context block.

    Each chunk is prefixed with its source filename so the model 'sees' the
    provenance too. Blank line between chunks keeps them visually separate,
    which helps the model treat them as distinct passages.
    """
    return "\n\n".join(f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in chunks)


def generate(question: str, chunks: list[dict]) -> str:
    """Answer `question` grounded strictly in `chunks`, via `CHAT_MODEL`.

    Returns the answer text. Assumes the caller (retriever) has already judged
    the question answerable — this function does not re-check the threshold; it
    trusts the chunks it's handed and synthesizes from them.
    """
    config = load_config()
    client = OpenAI(api_key=config.openai_api_key)

    context = _build_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=config.chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()
