"""Output: render the two terminal states — an answer, or a refusal.

Presentation lives here and nowhere else, so the CLI stays a thin controller and
the look of the tool can change without touching pipeline logic. There are
exactly two things to show the user:

  * `render_answer`  — the grounded answer, the source file(s) it came from, and
                       a confidence (the top similarity score). Green frame.
  * `render_refusal` — the "I don't know" state, with the near-miss score so the
                       user (and we, when tuning) can see how close it came.
                       Yellow frame.

Distinct border colours + titles make the two states instantly recognisable.

Both functions accept an optional `console` — defaulting to a shared module
Console — purely so tests (and any future caller) can inject a recording or
redirected Console and capture the output instead of only printing to a real
terminal. That small seam is what lets Task 10 be verified programmatically
rather than by eye alone.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Shared default console for normal (real-terminal) use.
_console = Console()


def render_answer(
    answer: str,
    sources: list[str],
    score: float,
    console: Console | None = None,
) -> None:
    """Render the grounded answer with its sources and confidence score."""
    console = console or _console

    sources_line = ", ".join(sources) if sources else "(none)"

    body = Text()
    body.append(answer.strip() + "\n\n")
    body.append("Sources: ", style="bold")
    body.append(sources_line + "\n")
    body.append("Confidence: ", style="bold")
    body.append(f"{score:.2f}")

    console.print(
        Panel(body, title="Answer", title_align="left", border_style="green")
    )


def render_refusal(score: float, console: Console | None = None) -> None:
    """Render the 'I don't know' refusal, showing the near-miss score."""
    console = console or _console

    body = Text()
    body.append(
        "I don't know — I couldn't find anything relevant to your question "
        "in the knowledge base.\n\n"
    )
    body.append("Closest match score: ", style="bold")
    body.append(f"{score:.2f}")

    console.print(
        Panel(body, title="I don't know", title_align="left", border_style="yellow")
    )
