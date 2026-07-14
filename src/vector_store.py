"""Vector store: persist chunk embeddings in ChromaDB and search them.

This is the heart of the "R" in RAG. It wraps a *persistent* Chroma collection
(stored on disk under `CHROMA_DIR`) and exposes three operations:

  * `build()`   — make sure every PDF's chunks are embedded and stored, WITHOUT
                  re-embedding files that haven't changed. This is the second,
                  outer cache in the pipeline (the first is the text-extraction
                  cache in ocr.py). Embedding is the expensive, paid step, so
                  skipping it for unchanged files is what makes iterative
                  development cheap.
  * `query()`   — embed a question and return the top-k most similar chunks,
                  each with a cosine-similarity *score* the retriever will
                  threshold against.
  * `collection_count()` — a tiny observability helper (how many chunks are
                  stored) used by tests and smoke checks.

## How the "skip unchanged files" cache works

Chroma stores, per chunk, a metadata field `file_hash` = SHA-256 of the source
PDF's bytes (the same content hash ocr.py uses). On `build()`, for each file we
ask Chroma "do any chunks with this exact `file_hash` already exist?":
  * yes -> the file is unchanged since we last embedded it -> skip it entirely.
  * no  -> the file is new OR its bytes changed -> delete any stale chunks for
           that source filename, then embed + store the fresh chunks.

Putting the hash in the chunk ID (`"<file_hash>:<n>"`) makes a changed file
produce brand-new IDs, so there's never an ID collision between old and new
versions of the same document.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import Config, load_config
from src.embeddings import embed_texts
from src.ingest import load_chunks

# One collection holds every chunk from every PDF. Cosine space matches the
# similarity metric the whole design assumes (and the 0-1 SIMILARITY_THRESHOLD).
_COLLECTION_NAME = "rag_chunks"

# Cache the Chroma client per on-disk path. A PersistentClient opens the
# underlying sqlite + index files, so rebuilding one on every query() in the
# REPL loop would be wasteful — and Chroma itself warns about multiple client
# instances for the same path. A connection handle is a resource, not business
# state, so caching it here (rather than in config) is the right home.
_clients: dict[str, chromadb.api.ClientAPI] = {}


def _file_hash(path: Path) -> str:
    """SHA-256 of a file's raw bytes — the cache key for 'has this changed?'.

    Intentionally the same scheme ocr.py uses for its text cache, so the two
    caches invalidate together when a PDF's bytes change. (Kept as a tiny local
    helper rather than shared, to keep this task's change confined to one file.)
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_collection(config: Config):
    """Return the persistent collection, reusing a cached client per path."""
    client = _clients.get(config.chroma_dir)
    if client is None:
        client = chromadb.PersistentClient(
            path=config.chroma_dir,
            # We don't want a learning tool phoning home during eval runs.
            settings=Settings(anonymized_telemetry=False),
        )
        _clients[config.chroma_dir] = client
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build() -> int:
    """Embed + persist all corpus chunks, skipping files that haven't changed.

    Returns the number of files that were actually (re)embedded this run — 0
    means every file was already up to date (the cache-hit path the spec asks
    us to prove). The CLI calls this once at startup so the store is warm before
    the first question.
    """
    config = load_config()
    collection = _get_collection(config)
    data_dir = Path(config.data_dir)

    # load_chunks() is cache-aware and cheap (no API); group its flat output by
    # source file so we can make the embed/skip decision one file at a time.
    chunks_by_source: dict[str, list[str]] = {}
    for chunk in load_chunks():
        chunks_by_source.setdefault(chunk["source"], []).append(chunk["text"])

    files_embedded = 0
    for source, texts in chunks_by_source.items():
        file_hash = _file_hash(data_dir / source)

        # Already embedded this exact content? -> nothing to do for this file.
        existing = collection.get(where={"file_hash": file_hash}, limit=1)
        if existing["ids"]:
            continue

        # New or changed file: clear any stale chunks for this source filename
        # (e.g. a previous version of the PDF), then embed and store fresh ones.
        collection.delete(where={"source": source})
        vectors = embed_texts(texts)
        collection.add(
            ids=[f"{file_hash}:{index}" for index in range(len(texts))],
            documents=texts,
            embeddings=vectors,
            metadatas=[{"source": source, "file_hash": file_hash} for _ in texts],
        )
        files_embedded += 1

    return files_embedded


def query(question: str, top_k: int) -> list[dict]:
    """Return the `top_k` chunks most similar to `question`, best first.

    Each result is `{"text", "source", "score"}` where `score` is cosine
    *similarity* in roughly [0, 1] (higher = closer). Chroma reports cosine
    *distance* (0 = identical), so we convert with `similarity = 1 - distance`.
    That conversion is important: the retriever thresholds on similarity, and a
    similarity is far more intuitive to reason about than a distance.
    """
    config = load_config()
    collection = _get_collection(config)

    question_vector = embed_texts([question])[0]
    response = collection.query(query_embeddings=[question_vector], n_results=top_k)

    # Chroma returns list-of-lists (one inner list per query); we sent one query,
    # so everything we want is at index [0].
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    results: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        results.append(
            {
                "text": document,
                "source": metadata["source"],
                "score": 1.0 - distance,
            }
        )
    return results


def collection_count() -> int:
    """Number of chunks currently stored — a small observability helper."""
    return _get_collection(load_config()).count()
