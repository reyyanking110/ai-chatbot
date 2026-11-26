"""Console presentation helpers (Rich only; no network)."""

from __future__ import annotations

import textwrap
from collections.abc import Iterable

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

THEME = Theme(
    {
        "jarvis": "bold cyan",
        "jarvis.dim": "dim cyan",
        "ok": "bold green",
        "warn": "bold yellow",
        "fail": "bold red",
        "info": "bold blue",
        "muted": "grey62",
        "accent": "magenta",
    }
)


def make_console(no_color: bool = False, quiet: bool = False) -> Console:
    return Console(
        theme=THEME,
        highlight=False,
        soft_wrap=False,
        no_color=no_color,
        quiet=quiet,
    )


def banner(console: Console, name: str, subtitle: str, extra: str = "") -> None:
    title = Text()
    title.append("◈ ", style="jarvis")
    title.append(name, style="bold white")
    body = Text(subtitle, style="muted")
    lines = [Align.center(title), Align.center(body)]
    if extra:
        lines.append(Align.center(Text(extra, style="jarvis.dim")))
    console.print(Panel(Group(*lines), border_style="jarvis", box=box.ROUNDED, padding=(0, 4)))


def section(console: Console, title: str) -> None:
    console.print()
    console.print(f"[jarvis]{title}[/]")


def kv_table(rows: list[tuple[str, str]], title: str | None = None, show_header: bool = False) -> Table:
    table = Table(
        title=title,
        box=box.SIMPLE_HEAD if show_header else box.SIMPLE,
        show_header=show_header,
        header_style="jarvis",
        padding=(0, 2, 0, 0),
        expand=False,
    )
    table.add_column("Key", style="muted", no_wrap=True)
    table.add_column("Value")
    for key, value in rows:
        table.add_row(key, value)
    return table


def check_badge(status: str) -> Text:
    styles = {"ok": ("✓", "ok"), "warn": ("!", "warn"), "fail": ("✗", "fail"), "info": ("·", "info")}
    glyph, style = styles.get(status, ("·", "muted"))
    return Text(glyph, style=style)


def checks_table(checks: Iterable) -> Table:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=True)
    table.add_column("", width=2, no_wrap=True)
    table.add_column("What", style="white", no_wrap=True)
    table.add_column("Detail", overflow="fold")
    for check in checks:
        detail = Text(check.detail, style="muted" if check.status != "fail" else "fail")
        if check.fix:
            detail.append("\n→ ", style="accent")
            detail.append(check.fix, style="accent")
        table.add_row(check_badge(check.status), check.name, detail)
    return table


def tool_step_text(name: str, detail: str, ok: bool, elapsed: float) -> Text:
    icon = Text("✓" if ok else "!", style="ok" if ok else "warn")
    label = Text()
    label.append(" ")
    label.append(name, style="accent")
    if detail:
        label.append("  ", style="muted")
        label.append(_clip(detail, 100), style="muted")
    if elapsed > 0.05:
        label.append(f"  {elapsed:.1f}s", style="jarvis.dim")
    return Text.assemble(icon, label)


def approval_panel(kind: str, prompt: str) -> Panel:
    title = Text.assemble(("Approve ", "bold white"), (kind, "accent"), ("?", "bold white"))
    return Panel(
        Text(prompt, style="white"),
        title=title,
        border_style="warn",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def sources_panel(hits) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("#", style="jarvis", width=3, no_wrap=True)
    table.add_column("Source")
    from .documents import citation_label

    for i, hit in enumerate(hits, 1):
        table.add_row(f"[{i}]", f"{citation_label(hit)}\n[grey62]{hit.path}[/]")
    return Panel(table, title="Sources", border_style="jarvis.dim", box=box.ROUNDED)


def answer_panel(text: str, title: str | None = None) -> Panel:
    return Panel(
        Markdown(text),
        title=title,
        border_style="jarvis",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def spinner(console: Console, message: str):
    return console.status(f"[jarvis]{message}[/]", spinner="dots")


def render_result(console: Console, result, title: str | None = None, show_sources: bool = False) -> None:
    console.print()
    console.print(answer_panel(result.text, title=title))
    if show_sources and result.hits:
        used = _cited_sources(result.text, result.hits)
        if used:
            console.print(sources_panel(used))
    meta = []
    if result.tool_calls:
        meta.append(f"{result.tool_calls} tool call{'s' if result.tool_calls != 1 else ''}")
    if result.model:
        meta.append(result.model)
    meta.append(f"{result.elapsed_s:.1f}s")
    console.print(Text("  ·  ".join(meta), style="jarvis.dim"))


def _cited_sources(text: str, hits) -> list:
    import re

    numbers = sorted({int(m) for m in re.findall(r"\[(\d{1,2})\]", text)})
    used = [hits[n - 1] for n in numbers if 1 <= n <= len(hits)]
    return used or hits[: min(3, len(hits))]


def wordmark(console: Console, name: str, version: str) -> None:
    art = Text()
    art.append(f"  {name.upper()}  ", style="bold cyan")
    art.append(f"v{version}", style="jarvis.dim")
    console.print(art)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width=width))
