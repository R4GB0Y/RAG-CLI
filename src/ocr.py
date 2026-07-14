"""Text extraction from PDFs — native-first, OCR-fallback, decided per page.

This module owns two responsibilities for the ingestion pipeline:

  1. **Routing** — for each page, decide whether its embedded text layer is
     good enough to use directly (fast, free, lossless) or whether the page is
     really a scanned image that must be transcribed by a vision model (slow,
     costly, and — critically for medical content — capable of silently
     paraphrasing). See `_page_needs_ocr`.

  2. **Caching** — extraction is deterministic for a given PDF, so we key the
     assembled result on a hash of the file's *bytes*. Re-running on an
     unchanged file is then a single file read instead of re-parsing every page
     (and, once Task 3 lands, re-calling the vision API). See `extract_text`.

Design note (why the OCR body is a stub here): Task 2 deliberately builds the
routing + caching *machinery* and leaves the actual vision call to Task 3. That
keeps each change small and independently verifiable — a core habit of shipping
incrementally rather than landing one giant, hard-to-review commit.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pymupdf  # PyMuPDF: one dependency for both text extraction and rendering
from openai import OpenAI

from src.config import load_config

# Rendering resolution for pages sent to the vision model. 200 DPI is a
# deliberate trade-off: high enough that small print (drug doses, mmHg values)
# stays legible, low enough to keep the image — and the token cost of sending
# it — modest. Lives here, not in .env, because it is an internal quality knob
# of the OCR path, not a user-facing tunable.
_OCR_RENDER_DPI = 200

# The instruction we hand the vision model. Worded to fight the single biggest
# risk of this whole pipeline: a model that *paraphrases* medical text instead
# of transcribing it. We demand verbatim output, forbid summarising/translating,
# and give it an explicit escape hatch ([illegible]) so it never invents text to
# fill a gap it cannot read.
_TRANSCRIBE_PROMPT = (
    "You are a precise OCR engine. Transcribe ALL text visible in this image "
    "exactly as it appears, verbatim. Preserve numbers, units, punctuation, "
    "line breaks, and table structure as faithfully as plain text allows. Do "
    "NOT summarise, translate, correct, reorder, or add anything. If part of "
    "the image is unreadable, write [illegible] in its place. Output only the "
    "transcribed text, with no preamble or commentary."
)


def _page_needs_ocr(
    native_text: str,
    force_ocr: bool,
    min_chars: int | None = None,
) -> bool:
    """Decide whether a single page must fall back to OCR.

    This is a *pure* function: no file I/O, no network, no global state beyond
    an optional config lookup for the default threshold. That purity is what
    makes the routing rule — the heart of the native-vs-OCR decision — trivial
    to test in isolation, without touching a real PDF or the OpenAI API.

    A page needs OCR when either:
      * `force_ocr` is set (operator explicitly wants every page transcribed), or
      * the page's native text layer is too thin to trust — fewer than
        `min_chars` *non-whitespace* characters. A scanned/image-only page
        typically yields an empty or near-empty text layer, which is exactly
        the signal we key on.

    We strip whitespace before counting so that a page which is "blank" apart
    from layout whitespace (newlines, spaces from an empty text frame) is
    correctly treated as having no real text.

    `min_chars` defaults to the configured `TEXT_LAYER_MIN_CHARS` when omitted,
    so callers in a hot loop pass it explicitly (one config load, not one per
    page) while ad-hoc callers and tests can rely on the default.
    """
    if force_ocr:
        return True
    if min_chars is None:
        min_chars = load_config().text_layer_min_chars
    non_whitespace_count = len("".join(native_text.split()))
    return non_whitespace_count < min_chars


def _transcribe_page_via_ocr(
    page: pymupdf.Page,
    client: OpenAI,
    ocr_model: str,
) -> str:
    """Transcribe a single scanned page via a vision model.

    Three steps:
      1. **Render** the PDF page to a raster image (PNG) with PyMuPDF — the same
         library we use for native extraction, so no extra system dependency.
         A vision model reads pixels, not a PDF, so we must rasterise first.
      2. **Encode** those PNG bytes as a base64 `data:` URL. That inlines the
         image directly in the request instead of hosting it somewhere and
         passing a URL — simplest and keeps the page bytes off any third party
         but OpenAI.
      3. **Ask** `ocr_model` (a vision-capable chat model) to transcribe it
         verbatim via `_TRANSCRIBE_PROMPT`, at `temperature=0` for the most
         deterministic, least-creative output we can get.

    The `client` and `ocr_model` are passed in (not built here) so the caller
    creates the OpenAI client once and reuses it across every OCR'd page in a
    document, rather than paying that setup cost per page.
    """
    # 1. Render page -> PNG bytes.
    pixmap = page.get_pixmap(dpi=_OCR_RENDER_DPI)
    png_bytes = pixmap.tobytes("png")

    # 2. Inline the image as a base64 data URL.
    base64_png = base64.b64encode(png_bytes).decode("ascii")
    image_data_url = f"data:image/png;base64,{base64_png}"

    # 3. Ask the vision model to transcribe it verbatim.
    response = client.chat.completions.create(
        model=ocr_model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _TRANSCRIBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    )
    # `.content` can be None if the model returns nothing usable; coerce to ""
    # so the caller always gets a string and the per-file assembly never breaks.
    return response.choices[0].message.content or ""


def extract_text(pdf_path: str | Path) -> str:
    """Return the full text of a PDF, native-first with OCR fallback, cached.

    Steps:
      1. Read the file's raw bytes and hash them (SHA-256). The hash is the
         cache key: identical bytes -> identical extraction, so we can reuse a
         prior result. Changing even one byte of the PDF changes the hash and
         invalidates the cache, which is the behavior we want.
      2. On a cache hit, return the stored text immediately — no parsing, no API.
      3. On a miss, walk every page: keep the native text where it's rich
         enough, otherwise route to OCR (`_transcribe_page_via_ocr`). Assemble
         the pages in order, write the assembled text to the cache, and return.

    The cache is per *file* even though routing is per *page* — a deliberate
    simplification (SPEC/plan): at this corpus size, re-processing a whole file
    when it changes is cheap and far simpler than tracking per-page entries.
    """
    config = load_config()
    pdf_path = Path(pdf_path)

    raw_bytes = pdf_path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    cache_dir = Path(config.text_cache_dir)
    cache_file = cache_dir / f"{content_hash}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    # Cache miss: open from the bytes we already read (avoids a second disk read)
    # and extract page by page.
    #
    # The OpenAI client is created lazily — only the first time a page actually
    # needs OCR. A fully digital-native PDF therefore never constructs a client
    # (and never needs a valid API key just to read its text layer).
    page_texts: list[str] = []
    ocr_client: OpenAI | None = None
    with pymupdf.open(stream=raw_bytes, filetype="pdf") as document:
        for page in document:
            native_text = page.get_text()
            if _page_needs_ocr(native_text, config.force_ocr, config.text_layer_min_chars):
                if ocr_client is None:
                    ocr_client = OpenAI(api_key=config.openai_api_key)
                page_texts.append(
                    _transcribe_page_via_ocr(page, ocr_client, config.ocr_model)
                )
            else:
                page_texts.append(native_text)

    # Join with a blank line so page boundaries stay visible in the assembled
    # text (helps later chunking and human spot-checks); order is preserved.
    assembled_text = "\n\n".join(page_texts)

    # Only reached if no page raised — a failed OCR page aborts before we cache,
    # so we never persist a partial/incorrect extraction.
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(assembled_text, encoding="utf-8")
    return assembled_text
