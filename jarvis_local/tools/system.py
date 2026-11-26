"""Read-only system introspection tools."""

from __future__ import annotations

import datetime as dt
import platform
import shutil

import httpx

from ..errors import PolicyError
from ..hardware import detect_hardware
from .base import Tool, ToolResult, arg, obj


def get_datetime(ctx) -> str:
    now = dt.datetime.now().astimezone()
    return now.strftime("%A %d %B %Y, %H:%M %Z") + f"\nISO: {now.isoformat(timespec='seconds')}"


def system_info(ctx) -> str:
    profile = detect_hardware()
    disk = shutil.disk_usage(str(ctx.policy.workspace))
    return (
        f"{profile.summary()}\n"
        f"Free disk in workspace: {disk.free / 1024**3:.0f} GB of {disk.total / 1024**3:.0f} GB\n"
        f"Python: {platform.python_version()} on {platform.platform()}"
    )


def web_fetch(ctx, url: str, max_chars: int = 4000) -> ToolResult:
    """Fetch a URL directly — only when the user explicitly asks for it."""
    if not url.lower().startswith(("http://", "https://")):
        raise PolicyError("Only http(s) URLs are supported.")
    if not ctx.cfg.skills.enabled:
        raise PolicyError("Network tools are disabled (skills.enabled = false).")
    try:
        with httpx.Client(timeout=15, follow_redirects=True, trust_env=True) as client:
            resp = client.get(url, headers={"User-Agent": "jarvis-local/0.1 (user initiated)"})
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise PolicyError(
            f"Fetch failed: {exc}",
            hint="Check your connection; Jarvis never fetches URLs on its own.",
        ) from exc
    content_type = resp.headers.get("content-type", "")
    if "html" in content_type:
        import html
        import re

        raw = resp.text
        raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
        raw = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|br)>", "\n", raw)
        text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
    else:
        text = resp.text
    total = len(text)
    clipped = text[: max(500, min(int(max_chars), 12000))]
    ctx.policy.audit("web", "fetch", {"url": url, "bytes": total})
    return ToolResult(
        f"Fetched {url} ({total} chars, showing {len(clipped)}):\n\n{clipped}",
        data={"url": url, "chars": total},
    )


TOOLS = [
    Tool(
        name="get_datetime",
        description="Get the current local date, time and timezone.",
        parameters=obj({}),
        func=lambda ctx: get_datetime(ctx),
        category="system",
    ),
    Tool(
        name="system_info",
        description="Report the machine's OS, CPU, RAM, GPU and free disk space.",
        parameters=obj({}),
        func=lambda ctx: system_info(ctx),
        category="system",
        agents=("chat", "code"),
    ),
    Tool(
        name="web_fetch",
        description=(
            "Download a single URL and return readable text. Only use when the user gives or "
            "explicitly asks about a URL — never proactively."
        ),
        parameters=obj(
            {"url": arg("Full http(s) URL.", "string"), "max_chars": arg("Characters to keep.", "integer")},
            ["url"],
        ),
        func=web_fetch,
        category="web",
        agents=("research", "chat"),
    ),
]
