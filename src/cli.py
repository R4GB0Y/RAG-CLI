"""CLI: the REPL that wires every stage into one interactive loop.

This is the only module that knows the *order* of the pipeline. Every stage it
calls was built and verified on its own (Tasks 2-10); nothing new is decided
here. The controller stays deliberately thin — if you find yourself adding
retrieval, prompting, or formatting logic to this file, it belongs in the module
that owns that job instead.

The shape of a single turn:

    question ─▶ retriever.retrieve ─┬─ not answerable ─▶ output.render_refusal
                                    │
                                    └─ answerable ─────▶ generator.generate
                                                         citation.get_sources
                                                         output.render_answer

Two details worth calling out:

  * **`vector_store.build()` runs once at startup**, not per question. Building
    is cache-aware (unchanged PDFs are skipped, costing no API calls), so it's
    cheap on every run after the first — but it still has to finish before the
    first question can be answered, so it belongs in startup, not in the loop.
  * **The refusal branch never reaches the generator.** The retriever already
    withholds its chunks when the top score misses the threshold, so this is
    belt-and-braces — but it's the guardrail the whole design exists to protect,
    and keeping the branch explicit here makes it impossible to miss on a read.
"""
from __future__ import annotations

from rich.console import Console

from src import citation, generator, output, retriever, vector_store

# Typed at the prompt to leave the REPL. Compared case-insensitively.
_EXIT_COMMANDS = {"exit", "quit"}

_PROMPT = "\n[bold cyan]Question[/] (or 'exit'): "


def _answer_question(question: str, console: Console) -> None:
    """Run one question through the pipeline and render the outcome."""
    result = retriever.retrieve(question)

    if not result.is_answerable:
        output.render_refusal(result.top_score, console=console)
        return

    answer = generator.generate(question, result.chunks)
    sources = citation.get_sources(result.chunks)
    output.render_answer(answer, sources, result.top_score, console=console)


def main(console: Console | None = None) -> None:
    """Build the store, then read-eval-print until the user exits.

    `console` is injectable for the same reason `output.py` allows it: it lets
    the REPL be driven with a recording Console and scripted stdin, so the loop
    and its branches can be verified without a real terminal or a real API call.
    """
    console = console or Console()

    console.print("[dim]Indexing corpus...[/]")
    files_embedded = vector_store.build()
    console.print(
        f"[dim]Ready — {vector_store.collection_count()} chunks indexed "
        f"({files_embedded} file(s) embedded this run).[/]"
    )

    while True:
        try:
            question = console.input(_PROMPT)
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D / Ctrl-C are ordinary ways to leave a REPL, not crashes —
            # exit as cleanly as typing 'exit' does.
            console.print("\n[dim]Goodbye.[/]")
            return

        question = question.strip()
        if not question:
            continue
        if question.lower() in _EXIT_COMMANDS:
            console.print("[dim]Goodbye.[/]")
            return

        _answer_question(question, console)


if __name__ == "__main__":
    main()
