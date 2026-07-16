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


# Spoken to the user whenever a question falls outside the knowledge base. It
# stays in character (an emergency-triage assistant) and redirects, rather than
# guessing — the refusal is still a refusal, just phrased as the persona.
REFUSAL_MESSAGE = (
    "Je suis une IA experte en urgences, ici pour vous assister dans le cadre "
    "des urgences médicales. Je ne peux répondre qu'aux questions relevant de "
    "ce domaine — n'hésitez pas à reformuler votre question dans ce contexte."
)


def render_refusal(score: float, console: Console | None = None) -> None:
    """Render the out-of-scope refusal in the assistant's persona.

    Still shows the near-miss score beneath the message so the tool remains
    observable for threshold tuning — the persona is presentation, the score is
    the diagnostic.
    """
    console = console or _console

    body = Text()
    body.append(REFUSAL_MESSAGE + "\n\n")
    body.append("Closest match score: ", style="bold")
    body.append(f"{score:.2f}")

    console.print(
        Panel(
            body,
            title="Assistant Urgences",
            title_align="left",
            border_style="yellow",
        )
    )
