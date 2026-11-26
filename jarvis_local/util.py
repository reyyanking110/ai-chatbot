"""Small shared helpers."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def local_now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def parse_date(value: str) -> dt.datetime | None:
    """Parse common date formats, including Google's RFC3339 timestamps."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Trim sub-second precision that fromisoformat chokes on in some versions.
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    for candidate in (text, text.replace(" ", "T")):
        try:
            parsed = dt.datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.astimezone()
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(text, fmt).astimezone()
        except ValueError:
            continue
    return None


def fmt_time(value: dt.datetime | str | None, tz: dt.tzinfo | None = None) -> str:
    parsed = value if isinstance(value, dt.datetime) else parse_date(value or "")
    if parsed is None:
        return "time TBD"
    if tz:
        parsed = parsed.astimezone(tz)
    if parsed.minute:
        return parsed.strftime("%H:%M")
    return parsed.strftime("%H:%M")


def fmt_datetime(value: dt.datetime | str | None) -> str:
    parsed = value if isinstance(value, dt.datetime) else parse_date(value or "")
    if parsed is None:
        return str(value or "")
    return parsed.strftime("%a %d %b, %H:%M")


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = str(text)
    return text if len(text) <= limit else text[: max(0, limit - len(suffix))].rstrip() + suffix


def human_bytes(n: int | float, suffix: str = "B") -> str:
    n = float(n)
    for unit in (suffix, "K" + suffix, "M" + suffix, "G" + suffix, "T" + suffix):
        if n < 1024 or unit.startswith("T"):
            return f"{n:.0f} {unit}" if unit == suffix else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} T{suffix}"


def json_dumps(obj: Any, limit: int | None = None) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    return truncate(text, limit) if limit else text


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def run_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 20,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a command without a shell and capture output."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Timed out after {timeout:g}s"
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"
    except OSError as exc:
        return 126, "", f"Could not execute: {exc}"


def iter_text_files(
    roots: list[Path],
    extensions: set[str],
    ignore_names: set[str],
    max_bytes: int,
    limit: int | None = None,
) -> Iterator[tuple[Path, int]]:
    """Walk roots, yielding (path, size) for candidate files."""
    seen: set[Path] = set()
    count = 0
    for root in roots:
        if not root.exists():
            continue
        path_iter = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in path_iter:
            if limit is not None and count >= limit:
                return
            try:
                if not path.is_file():
                    continue
                if any(part in ignore_names for part in path.parts):
                    continue
                if path.suffix.lower() not in extensions:
                    continue
                if path.stat().st_size > max_bytes or path.stat().st_size == 0:
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
            except OSError:
                continue
            count += 1
            yield path, path.stat().st_size


def render_markdown(console, text: str) -> None:
    console.print(Markdown(text))


def render_note(console, title: str, body: str, style: str = "cyan") -> None:
    console.print(Panel(Text(body), title=title, border_style=style, expand=False))


def supports_unicode() -> bool:
    try:
        "✓".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def stopwatch() -> float:
    return time.monotonic()


def elapsed(start: float) -> str:
    secs = time.monotonic() - start
    return f"{secs:.1f}s" if secs < 60 else f"{int(secs // 60)}m {int(secs % 60)}s"


def safe_slug(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-").lower()
    return slug[:limit] or "item"


def env_without_secrets() -> dict[str, str]:
    """Environment for shell/code tools: keep PATH/HOME, drop obvious tokens."""
    blocked = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|CREDENTIAL|AUTH)", re.I)
    env = {k: v for k, v in os.environ.items() if not blocked.search(k)}
    env.setdefault("JARVIS_SANDBOX", "1")
    env["PYTHONIOENCODING"] = "utf-8"
    return env
