"""Ingestion: turn the PDF corpus into embeddable, source-tagged text chunks.

This is the bridge between raw extraction (`ocr.extract_text`) and the vector
store. It does two things:

  1. **Chunk** — split each document's text into fixed-size, overlapping windows
     (`_chunk_text`). Why chunk at all? Embedding models have a context limit and,
     more importantly, retrieval quality degrades if a single vector has to
     represent a whole document — a focused paragraph-sized chunk gives a much
     sharper similarity signal for the specific fact a question is asking about.
     Why *overlap*? So a sentence that straddles a window boundary isn't split in
     a way that loses its meaning in both halves; the overlap keeps the seam
     readable in at least one chunk.

  2. **Tag** — attach the source PDF filename to every chunk (`load_chunks`), so
     that once a chunk is retrieved we can always cite where it came from. This
     is a hard requirement of the spec (traceable citations) and it is cheapest
     to attach the provenance *here*, at the moment we still know which file the
     text came from, rather than trying to recover it later.
"""
from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.ocr import extract_text


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split `text` into overlapping fixed-size character windows.

    Pure function (no I/O) so the windowing math can be tested with tiny,
    hand-computed inputs. The window advances by `step = chunk_size -
    chunk_overlap` characters each iteration, so consecutive windows share
    exactly `chunk_overlap` characters.

    `config.load_config` already guarantees `0 <= chunk_overlap < chunk_size`,
    so `step` is always positive and the loop always terminates — but we assert
    it here too, because a silent zero/negative step would be an infinite loop,
    the kind of bug that's far cheaper to catch at the boundary than in prod.

    Windows that are entirely whitespace are dropped: they carry no meaning,
    would waste an embedding call, and would pollute retrieval with blank hits.
    """
    step = chunk_size - chunk_overlap
    assert step > 0, "chunk_size must exceed chunk_overlap (validated in config)"

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        window = text[start : start + chunk_size]
        if window.strip():  # skip whitespace-only windows
            chunks.append(window)
        start += step
    return chunks


def load_chunks() -> list[dict]:
    """Extract, chunk, and source-tag every PDF in `DATA_DIR`.

    Returns a flat list of `{"text": <chunk>, "source": <filename>}` dicts —
    deliberately plain data (no custom class) so it flows unchanged into the
    embeddings + vector-store stages, matching the spec's "functions take and
    return plain data" style.

    Files are processed in sorted order for deterministic, reproducible output
    (stable chunk ordering makes the vector store and eval results repeatable).
    `extract_text` is cache-aware, so re-running this over an unchanged corpus
    does no re-extraction and — for digital-native PDFs — makes no API call.
    """
    config = load_config()
    data_dir = Path(config.data_dir)

    chunks: list[dict] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        document_text = extract_text(pdf_path)
        for window in _chunk_text(document_text, config.chunk_size, config.chunk_overlap):
            # `.name` is just the filename (e.g. "urgence.pdf") — the citation
            # granularity the spec asks for (per-file, not per-path).
            chunks.append({"text": window, "source": pdf_path.name})
    return chunks
