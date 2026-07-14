"""Embeddings: turn text into vectors via the OpenAI Embeddings API.

Deliberately a *thin* wrapper — one job, one function. It exists so the rest of
the pipeline (vector store, retriever) never imports the OpenAI SDK directly or
knows which model/API shape produced a vector. If we ever swap embedding
providers, this is the only file that changes. That single-responsibility
boundary is the whole point of giving embeddings its own module.
"""
from __future__ import annotations

from openai import OpenAI

from src.config import load_config


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input in input order.

    Batching matters: we send the whole list in a single API call rather than
    one call per text. That's faster and cheaper (fewer round-trips), and it's
    why the vector store passes all its chunks here at once.

    Order guarantee: the API tags each returned embedding with the `index` of
    the input it came from. We sort by that index before stripping it, so the
    Nth vector out always corresponds to the Nth text in — the vector store
    relies on this to line vectors up with their chunk metadata.

    Empty-input guard: calling the API with an empty list is an error, so we
    short-circuit to `[]`. This also means "no chunks" flows through the
    pipeline harmlessly instead of raising deep inside an API call.
    """
    if not texts:
        return []

    config = load_config()
    client = OpenAI(api_key=config.openai_api_key)

    response = client.embeddings.create(model=config.embedding_model, input=texts)

    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]
