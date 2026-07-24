# RAG CLI — Q&A over a local PDF knowledge base

A command-line Retrieval-Augmented Generation (RAG) tool. You run an interactive
REPL, type a question, and it answers **only** from a small local corpus of PDF
files — with citations. Questions outside the corpus get an honest "I don't know"
refusal instead of a hallucinated answer.

This is a learning project: every stage of the RAG pipeline — text extraction,
chunking, embedding, vector storage, similarity search, thresholding, generation,
and citation — is built explicitly rather than hidden behind a framework. Clarity
beats cleverness at every decision point. See [`SPEC.md`](SPEC.md) for the full
design rationale.

The bundled corpus (`data/urgence.pdf`) is French emergency-medicine guidance, so
the assistant answers in the persona of an emergency-triage assistant.

## How it works

```
data/*.pdf
   │  ocr.py — per page: native pymupdf text extraction first;
   │  OCR fallback (OpenAI vision) only for pages below TEXT_LAYER_MIN_CHARS
   │  or when FORCE_OCR=true. Assembled per file, cached in .text_cache/ by hash.
   ▼
document text
   │  ingest.py — fixed-size sliding-window chunking (CHUNK_SIZE / CHUNK_OVERLAP),
   │  each chunk tagged with its source filename
   ▼
chunks
   │  embeddings.py — OpenAI embeddings API
   ▼
vector_store.py ──persists──▶ .chroma/   (unchanged files are not re-embedded)

question (typed in REPL)
   │  embed question ─▶ vector_store top-k query ─▶ retriever.py
   ▼
retriever.py compares the best score to SIMILARITY_THRESHOLD
   ├─ below threshold ─▶ output.render_refusal   ("I don't know", in persona)
   └─ above threshold ─▶ generator.generate (grounded strictly in retrieved chunks)
                         citation.get_sources  ─▶  output.render_answer
```

The retriever is the guardrail: the vector store always returns its nearest
chunks, even for an unrelated question, so the retriever inspects the *best*
similarity score and refuses when nothing clears the threshold — withholding the
chunks entirely so nothing downstream can answer from irrelevant context. The
generator runs at `temperature=0` with a system prompt that forbids using any
knowledge outside the provided chunks.

## Tech stack

| Concern            | Choice |
|--------------------|--------|
| Language           | Python 3.11+ |
| Package manager    | [`uv`](https://docs.astral.sh/uv/) |
| PDF text + render  | `pymupdf` (native text layer first, no poppler dependency) |
| OCR fallback       | OpenAI vision chat completions (`gpt-4o-mini`), per scanned page only |
| Embeddings         | OpenAI Embeddings API (`text-embedding-3-small`) |
| Generation         | OpenAI Chat Completions (`gpt-4o-mini`) |
| Vector store       | ChromaDB (persistent, on-disk — caches embeddings between runs) |
| Terminal output    | `rich` |
| Config             | `python-dotenv` — everything tunable lives in `.env` |

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
# 1. Install dependencies
uv sync

# 2. Configure — copy the template and add your key
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...

# 3. Add your corpus (optional — data/urgence.pdf is included)
#    Drop 3–5 .pdf files into data/
```

## Usage

```bash
# Start the interactive REPL
uv run python -m src.cli
```

On first launch the corpus is extracted, chunked, embedded, and indexed into
`.chroma/`. Subsequent runs reuse the cache — unchanged PDFs cost no API calls.
Type a question at the prompt; type `exit` (or `quit`, or Ctrl-D) to leave.

### Patient demo

```bash
uv run python scripts/patient_demo.py
```

Same pipeline, but for each turn it also prints the structured **JSON payload**
a real RAG service would return to a backend client — the answer, the grounding
chunks with scores, the retrieval decision, model metadata, and per-stage
latencies. Useful for seeing the machine-readable contract behind the pretty
terminal panel.

## Configuration

All behaviour is driven by `.env` — changing it changes behaviour without
touching code. See [`.env.example`](.env.example) for the annotated template.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | *(required)* | Used for OCR, embeddings, and generation |
| `OCR_MODEL` | `gpt-4o-mini` | Vision model for scanned pages |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | Answer-generation model |
| `SIMILARITY_THRESHOLD` | `0.3` | Min top cosine score to answer vs. refuse (0–1) |
| `CHUNK_SIZE` | `500` | Chunk length (characters) |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TEXT_LAYER_MIN_CHARS` | `100` | Below this many native chars, a page falls back to OCR |
| `FORCE_OCR` | `false` | Force every page through OCR |
| `DATA_DIR` | `data` | Where source PDFs live |
| `CHROMA_DIR` | `.chroma` | Persistent vector store |
| `TEXT_CACHE_DIR` | `.text_cache` | Cached extracted text (keyed by content hash) |

## Project structure

```
Project_RAG_CLI/
├── pyproject.toml       uv-managed project + dependencies
├── .env.example         template for required env vars (copy to .env)
├── SPEC.md              full design spec and rationale
├── data/                source .pdf files (data/urgence.pdf bundled)
├── src/
│   ├── config.py        loads + validates all .env vars in one place
│   ├── ocr.py           native extraction first, OCR fallback per page; caches result
│   ├── ingest.py        extracts PDFs, splits into overlapping chunks, tags source
│   ├── embeddings.py    thin wrapper around the OpenAI embeddings API
│   ├── vector_store.py  builds/queries the Chroma collection (re-embeds only changes)
│   ├── retriever.py     top-k query + threshold decision (answer vs. refuse)
│   ├── generator.py     builds the grounded prompt and calls the chat model
│   ├── citation.py      deduplicated source filenames from retrieved chunks
│   ├── output.py        renders the rich answer / refusal panels
│   └── cli.py           REPL entry point wiring the pipeline together
├── scripts/
│   └── patient_demo.py  REPL that also prints the backend JSON payload
└── eval/                automated eval harness (see SPEC.md — Testing Strategy)
```

## Notes

- `.env`, `.chroma/`, and `.text_cache/` are gitignored — never commit your key
  or the generated caches.
- No build, lint, or unit-test tooling is configured beyond the eval harness —
  intentionally minimal for a learning exercise.
