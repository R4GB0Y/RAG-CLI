"""Interactive patient-facing demo of the RAG service, with API-payload preview.

Run this to play the role of a patient asking the knowledge base questions. For
every turn it does two things:

  1. Renders the human-facing answer (or refusal) exactly as `src/cli.py` would.
  2. Prints the **structured JSON payload** that the RAG service would hand to a
     backend client in a real deployment — the machine-readable contract behind
     the pretty terminal panel.

Nothing about the pipeline changes here; this script only *wraps* the existing
stages (`retriever` → `generator`/refusal → `citation`) and serialises their
output into the response envelope a real API would return. Think of it as a
window into the wire format: what the CLI shows a human vs. what a service would
send a caller.

Run:
    uv run python scripts/patient_demo.py
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# This script lives in scripts/, but imports the pipeline from the project's
# top-level `src` package. Put the project root on the path so it runs the same
# whether invoked as `python scripts/patient_demo.py` or from anywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.rule import Rule

from src import citation, generator, output, retriever, vector_store
from src.config import load_config

# How much chunk text to include in the payload preview. The real service might
# ship the full chunk, but for a readable demo we truncate — the score and source
# are what matter for showing *why* the answer is grounded.
_CHUNK_PREVIEW_CHARS = 240

_EXIT_COMMANDS = {"exit", "quit"}
_PROMPT = "\n[bold cyan]Patient[/] (type your question, or 'exit'): "


def _confidence_band(score: float, threshold: float) -> str:
    """A coarse, human-readable confidence label derived from the top score.

    Purely presentational metadata for the payload — the answer/refuse decision
    itself is made by the retriever against `threshold`, not by this label.
    """
    if score < threshold:
        return "none"
    if score >= threshold + 0.25:
        return "high"
    if score >= threshold + 0.10:
        return "medium"
    return "low"


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= _CHUNK_PREVIEW_CHARS:
        return text
    return text[:_CHUNK_PREVIEW_CHARS].rstrip() + "…"


def _build_payload(
    question: str,
    result: retriever.RetrievalResult,
    answer: str | None,
    sources: list[str],
    threshold: float,
    config,
    latency_ms: dict[str, float],
) -> dict:
    """Assemble the JSON envelope a real RAG service would return to a backend.

    This is the whole point of the demo: a single, self-describing object that
    carries the answer, the grounding evidence, the retrieval decision, and the
    operational metadata a downstream client would log, display, or route on.
    """
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": {
            "text": question,
            "role": "patient",
        },
        "response": {
            "answerable": result.is_answerable,
            "answer": answer,
            # When we refuse, spell out why in a machine-readable way so a client
            # can branch on it (e.g. route to a human) rather than parse prose.
            "refusal_reason": (
                None
                if result.is_answerable
                else "no_relevant_context_above_threshold"
            ),
        },
        "retrieval": {
            "top_score": round(result.top_score, 4),
            "threshold": threshold,
            "confidence": _confidence_band(result.top_score, threshold),
            "chunks_considered": len(result.chunks),
            "chunks": [
                {
                    "rank": index + 1,
                    "source": chunk["source"],
                    "score": round(chunk["score"], 4),
                    "text_preview": _truncate(chunk["text"]),
                }
                for index, chunk in enumerate(result.chunks)
            ],
        },
        "citations": sources,
        "model": {
            "chat_model": config.chat_model,
            "embedding_model": config.embedding_model,
        },
        "latency_ms": {stage: round(value, 1) for stage, value in latency_ms.items()},
    }


def _answer_turn(question: str, console: Console, config) -> None:
    """Run one patient question: render the answer, then dump the API payload."""
    threshold = round(config.similarity_threshold, 4)
    latency_ms: dict[str, float] = {}

    started = time.perf_counter()
    result = retriever.retrieve(question)
    latency_ms["retrieval"] = (time.perf_counter() - started) * 1000

    answer: str | None = None
    sources: list[str] = []

    if result.is_answerable:
        started = time.perf_counter()
        answer = generator.generate(question, result.chunks)
        latency_ms["generation"] = (time.perf_counter() - started) * 1000
        sources = citation.get_sources(result.chunks)
        output.render_answer(answer, sources, result.top_score, console=console)
    else:
        # Out of scope: the patient sees the persona redirect, and the payload
        # carries that same message as the answer — while `answerable`/
        # `refusal_reason` below still report the true retrieval decision.
        answer = output.REFUSAL_MESSAGE
        output.render_refusal(result.top_score, console=console)

    payload = _build_payload(
        question=question,
        result=result,
        answer=answer,
        sources=sources,
        threshold=threshold,
        config=config,
        latency_ms=latency_ms,
    )

    # Render as syntax-highlighted JSON for the terminal, but it is genuine,
    # valid JSON — copy it straight out and a backend would accept it.
    console.print(
        Panel(
            JSON(json.dumps(payload)),
            title="→ JSON payload sent to backend client",
            title_align="left",
            border_style="magenta",
        )
    )


def main(console: Console | None = None) -> None:
    console = console or Console()
    config = load_config()

    console.print(
        Panel.fit(
            "[bold]RAG service — patient demo[/]\n"
            "Ask a question as if you were a patient. You'll see the answer the\n"
            "service renders, followed by the JSON it would POST to a backend.",
            border_style="cyan",
        )
    )

    console.print("[dim]Indexing corpus...[/]")
    files_embedded = vector_store.build()
    console.print(
        f"[dim]Ready — {vector_store.collection_count()} chunks indexed "
        f"({files_embedded} file(s) embedded this run). "
        f"Threshold={config.similarity_threshold} · model={config.chat_model}.[/]"
    )

    while True:
        try:
            question = console.input(_PROMPT)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            return

        question = question.strip()
        if not question:
            continue
        if question.lower() in _EXIT_COMMANDS:
            console.print("[dim]Goodbye.[/]")
            return

        console.print(Rule(style="dim"))
        _answer_turn(question, console, config)


if __name__ == "__main__":
    main()
