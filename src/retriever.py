"""Retrieval decision: turn raw similarity scores into an answer/refuse verdict.

This is the guardrail that makes the whole tool trustworthy. The vector store
will *always* return its top-k nearest chunks — even for a question about
baking cookies, it returns the three least-unrelated medical chunks. Left
unchecked, the generator would then dutifully write an answer from irrelevant
context. The retriever's job is to look at the *best* similarity score and
decide whether anything we found is actually relevant enough to answer from —
and if not, to say "I don't know" instead of hallucinating.

The decision is a single comparison against `SIMILARITY_THRESHOLD`. We keep the
`top_score` on the result even when we refuse, so callers (and the eval harness)
can see *how close* a question came to the threshold — essential for tuning it.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import load_config
from src import vector_store

# How many chunks to retrieve per question. Fixed retrieval breadth, not a
# user-facing tunable (it isn't in the .env config surface). Three gives the
# generator a little redundancy/context without drowning it in loosely-related
# text. Promote to .env later if it ever needs tuning.
_TOP_K = 3


@dataclass
class RetrievalResult:
    """Outcome of a retrieval attempt.

    is_answerable: did the best match clear the similarity threshold?
    chunks:        the retrieved {text, source, score} dicts — EMPTY on refusal,
                   so a caller can never accidentally build an answer from
                   below-threshold context.
    top_score:     the best similarity score we saw, reported either way (this
                   is what you inspect when tuning SIMILARITY_THRESHOLD).
    """

    is_answerable: bool
    chunks: list[dict]
    top_score: float


def retrieve(question: str, threshold: float | None = None) -> RetrievalResult:
    """Retrieve top-k chunks and decide whether the question is answerable.

    `threshold` defaults to the configured `SIMILARITY_THRESHOLD`; it's exposed
    as a parameter so the decision boundary can be exercised directly in tests
    (and, later, swept during threshold tuning) without editing `.env`.

    Logic (mirrors the SPEC's reference shape):
      * No matches at all -> top_score 0.0 -> not answerable.
      * Best score below the threshold -> not answerable, drop the chunks.
      * Otherwise -> answerable, hand back all retrieved chunks.
    """
    if threshold is None:
        threshold = load_config().similarity_threshold

    matches = vector_store.query(question, top_k=_TOP_K)
    top_score = matches[0]["score"] if matches else 0.0

    if top_score < threshold:
        # Below the bar: refuse, and deliberately withhold the chunks so no
        # downstream code can answer from irrelevant context.
        return RetrievalResult(is_answerable=False, chunks=[], top_score=top_score)

    return RetrievalResult(is_answerable=True, chunks=matches, top_score=top_score)
