"""File tools — confined to the workspace for writes, tolerant for reads."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from ..errors import PolicyError
from ..util import human_bytes, iter_text_files
from .base import Tool, ToolResult, arg, obj

MAX_READ = 120_000


def _display_path(ctx, path: Path) -> str:
    try:
        return str(path.relative_to(ctx.policy.workspace))
    except ValueError:
        if path.is_relative_to(Path.home()):
            return "~/" + str(path.relative_to(Path.home()))
        return str(path)


def list_files(ctx, path: str = ".", pattern: str = "*", depth: int = 2) -> str:
    root = ctx.policy.resolve_read(ctx.policy.workspace_path(path))
    if root.is_file():
        return f"{_display_path(ctx, root)} ({human_bytes(root.stat().st_size)})"
    if not root.exists():
        raise PolicyError(f"No such file or directory: {path}")
    depth = max(0, min(int(depth), 6))
    entries: list[str] = []
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        level = len(current.parts) - base_depth
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in ctx.cfg.paths.ignore
        )
        if level > depth:
            dirnames[:] = []
            continue
        indent = "  " * level
        if level > 0:
            entries.append(f"{indent}{current.name}/")
        for name in sorted(filenames):
            if not fnmatch.fnmatch(name, pattern):
                continue
            full = current / name
            try:
                size = human_bytes(full.stat().st_size)
            except OSError:
                size = "?"
            entries.append(f"{indent}{name}  ({size})")
        if level == depth:
            dirnames[:] = []
    if not entries:
        return f"(empty) {_display_path(ctx, root)}"
    return f"{_display_path(ctx, root)}/\n" + "\n".join(entries[:400])


def read_file(ctx, path: str, start_line: int = 1, max_lines: int = 400) -> ToolResult:
    target = ctx.policy.resolve_read(ctx.policy.workspace_path(path))
    if not target.exists():
        raise PolicyError(f"No such file: {path}")
    if target.is_dir():
        raise PolicyError(f"{path} is a directory; use list_files.")
    size = target.stat().st_size
    if size > 20 * 1024 * 1024:
        raise PolicyError(f"{path} is too large to read ({human_bytes(size)}).")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PolicyError(f"Could not read {path}: {exc}") from exc
    lines = text.splitlines()
    start = max(1, int(start_line))
    end = min(len(lines), start - 1 + max(0, int(max_lines)))
    numbered = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1))
    header = f"{_display_path(ctx, target)} — lines {start}-{end} of {len(lines)}"
    body = numbered[:MAX_READ]
    if len(numbered) > MAX_READ:
        body += "\n… (truncated; read a later range with start_line)"
    return ToolResult(f"{header}\n{body}")


def write_file(ctx, path: str, content: str, overwrite: bool = True) -> ToolResult:
    target = ctx.policy.resolve_write(ctx.policy.workspace_path(path))
    if isinstance(content, (dict, list)):
        import json

        content = json.dumps(content, indent=2)
    data = str(content)
    if len(data.encode("utf-8")) > ctx.cfg.security.max_write_bytes:
        raise PolicyError(
            f"Refusing to write {human_bytes(len(data.encode()))}; "
            f"limit is {human_bytes(ctx.cfg.security.max_write_bytes)}."
        )
    if target.exists() and not overwrite:
        raise PolicyError(f"{_display_path(ctx, target)} already exists (overwrite=false).")
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(data, encoding="utf-8")
    ctx.policy.audit("file", "write", {"path": str(target), "bytes": len(data.encode())})
    verb = "Updated" if existed else "Created"
    return ToolResult(f"{verb} {_display_path(ctx, target)} ({len(data.splitlines())} lines).")


def append_file(ctx, path: str, content: str) -> ToolResult:
    target = ctx.policy.resolve_write(ctx.policy.workspace_path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(str(content))
    ctx.policy.audit("file", "append", {"path": str(target)})
    return ToolResult(f"Appended {len(str(content))} characters to {_display_path(ctx, target)}.")


def delete_file(ctx, path: str) -> ToolResult:
    target = ctx.policy.resolve_write(ctx.policy.workspace_path(path))
    if not target.exists():
        raise PolicyError(f"No such file: {path}")
    if target.is_dir():
        raise PolicyError("delete_file only removes files; directories need a shell command.")
    target.unlink()
    ctx.policy.audit("file", "delete", {"path": str(target)})
    return ToolResult(f"Deleted {_display_path(ctx, target)}.")


def move_file(ctx, source: str, destination: str) -> ToolResult:
    src = ctx.policy.resolve_write(ctx.policy.workspace_path(source))
    dst = ctx.policy.resolve_write(ctx.policy.workspace_path(destination))
    if not src.exists():
        raise PolicyError(f"No such file: {source}")
    if dst.exists():
        raise PolicyError(f"Destination exists: {destination}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    ctx.policy.audit("file", "move", {"from": str(src), "to": str(dst)})
    return ToolResult(f"Moved {_display_path(ctx, src)} → {_display_path(ctx, dst)}.")


def search_files(ctx, pattern: str, path: str = ".", glob: str = "*", max_results: int = 40) -> str:
    root = ctx.policy.resolve_read(ctx.policy.workspace_path(path or "."))
    if not root.exists():
        raise PolicyError(f"No such path: {path}")
    exts = {e.lower() for e in ctx.cfg.library.extensions}
    results: list[str] = []
    roots = [root] if root.is_file() else [root]
    for file_path, _size in iter_text_files(
        roots, exts, set(ctx.cfg.paths.ignore), ctx.cfg.library.max_file_mb * 1024 * 1024, limit=4000
    ):
        if glob and glob != "*" and not fnmatch.fnmatch(file_path.name, glob):
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.lower() in line.lower():
                snippet = line.strip()[:220]
                results.append(f"{_display_path(ctx, file_path)}:{lineno}: {snippet}")
                if len(results) >= max_results:
                    return "\n".join(results) + f"\n… stopped at {max_results} matches"
    return "\n".join(results) if results else f"No matches for {pattern!r}."


TOOLS = [
    Tool(
        name="list_files",
        description="List files and directories inside the workspace (or an allowed folder), with sizes.",
        parameters=obj(
            {
                "path": arg("Directory or file path. Relative paths resolve inside the workspace.", "string"),
                "pattern": arg("Optional glob filter for file names, e.g. '*.py'.", "string"),
                "depth": arg("How many directory levels to descend (0-6).", "integer"),
            }
        ),
        func=list_files,
        category="files",
        agents=("chat", "code"),
    ),
    Tool(
        name="read_file",
        description="Read a text file with line numbers. Supports paging with start_line/max_lines.",
        parameters=obj(
            {
                "path": arg("File to read.", "string"),
                "start_line": arg("First line number (1-based).", "integer"),
                "max_lines": arg("Maximum lines to return.", "integer"),
            },
            ["path"],
        ),
        func=read_file,
        category="files",
        agents=("chat", "code"),
    ),
    Tool(
        name="write_file",
        description="Create or overwrite a file inside the workspace. Shows old/new for review when enabled.",
        parameters=obj(
            {
                "path": arg("Destination path inside the workspace.", "string"),
                "content": arg("Full file content to write.", "string"),
                "overwrite": arg("Set false to refuse if the file exists.", "boolean"),
            },
            ["path", "content"],
        ),
        func=write_file,
        category="files",
        needs_approval=True,
        agents=("code",),
    ),
    Tool(
        name="append_file",
        description="Append text to a file inside the workspace.",
        parameters=obj(
            {"path": arg("File to append to.", "string"), "content": arg("Text to append.", "string")},
            ["path", "content"],
        ),
        func=append_file,
        category="files",
        needs_approval=True,
        agents=("code",),
    ),
    Tool(
        name="delete_file",
        description="Delete a single file inside the workspace.",
        parameters=obj({"path": arg("File to delete.", "string")}, ["path"]),
        func=delete_file,
        category="files",
        needs_approval=True,
        agents=("code",),
    ),
    Tool(
        name="move_file",
        description="Move or rename a file inside the workspace.",
        parameters=obj(
            {"source": arg("Current path.", "string"), "destination": arg("New path.", "string")},
            ["source", "destination"],
        ),
        func=move_file,
        category="files",
        needs_approval=True,
        agents=("code",),
    ),
    Tool(
        name="search_files",
        description="Case-insensitive text search across workspace files (like grep).",
        parameters=obj(
            {
                "pattern": arg("Text to find.", "string"),
                "path": arg("Folder to search, relative to the workspace.", "string"),
                "glob": arg("Only search files matching this glob, e.g. '*.md'.", "string"),
                "max_results": arg("Maximum matches to return.", "integer"),
            },
            ["pattern"],
        ),
        func=search_files,
        category="files",
        agents=("chat", "code"),
    ),
]
