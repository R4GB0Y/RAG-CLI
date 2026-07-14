"""Citation: derive the list of source files behind a set of retrieved chunks.

When the retriever returns several chunks, more than one can come from the same
PDF (adjacent windows of the same document), and the order chunks come back in
is *relevance* order (best match first). For the "Sources:" line we show the
user, we want each file named once, in the order it first appeared — most
relevant source first, no duplicates.

This is deliberately its own tiny module: a pure, side-effect-free transform
that's trivial to test and reuse (the CLI and the eval harness both call it).
"""
from __future__ import annotations


def get_sources(chunks: list[dict]) -> list[str]:
    """Return the unique source filenames from `chunks`, first-seen order kept.

    Why not just `set(...)`? A set would dedupe but *lose ordering*, and we want
    the most-relevant file listed first. Why not `sorted(set(...))`? That would
    impose alphabetical order, which is meaningless to the user. So we walk the
    chunks in their given (relevance) order and keep each source the first time
    we see it — dedupe *and* preserve intent.
    """
    seen: list[str] = []
    for chunk in chunks:
        source = chunk["source"]
        if source not in seen:
            seen.append(source)
    return seen
