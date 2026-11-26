"""Shell and Python execution with approval + policy enforcement."""

from __future__ import annotations

import sys

from ..errors import PolicyError
from ..policy import format_command_preview
from ..util import human_bytes, run_capture
from .base import Tool, ToolResult, arg, obj


def run_shell(ctx, command: str, timeout_s: int | None = None) -> ToolResult:
    verdict = ctx.policy.inspect_command(command)
    if not verdict.allowed:
        ctx.policy.audit("shell", "blocked", {"command": command, "reason": verdict.reason})
        raise PolicyError(
            verdict.reason,
            hint="Split the task into smaller steps, or ask me to do something less destructive.",
        )
    approved = True
    if verdict.needs_confirm and not verdict.is_read_only:
        approved = ctx.approve(
            "shell",
            f"Run command:\n\n  {format_command_preview(command)}\n\n"
            f"Working directory: {ctx.policy.workspace}\nReason: {verdict.reason}",
        )
    if not approved:
        ctx.policy.audit("shell", "declined", {"command": command})
        return ToolResult("The user declined to run this command.", ok=False)

    timeout = timeout_s or ctx.cfg.security.shell_timeout_s
    timeout = max(1, min(int(timeout), 600))
    code, stdout, stderr = run_capture(
        ["/bin/sh", "-c", command],
        cwd=ctx.policy.workspace,
        timeout=timeout,
        env=ctx.policy.scrubbed_env(),
    )
    ctx.policy.audit("shell", "run", {"command": command, "exit": code})
    limit = ctx.cfg.security.max_output_chars
    out = _clip(stdout, limit)
    err = _clip(stderr, limit // 2)
    parts = [f"exit code: {code}"]
    if out:
        parts.append("stdout:\n" + out)
    if err:
        parts.append("stderr:\n" + err)
    if not out and not err:
        parts.append("(no output)")
    return ToolResult("\n".join(parts), ok=code == 0)


def run_python(ctx, code: str, timeout_s: int | None = None) -> ToolResult:
    if not ctx.cfg.security.allow_shell:
        raise PolicyError("Code execution is disabled (security.allow_shell = false).")
    if len(code) > 60_000:
        raise PolicyError("Script is too long (60k character limit).")
    if ctx.cfg.security.approval != "auto":
        preview = code if len(code) <= 1200 else code[:1200] + "\n… (truncated)"
        approved = ctx.approve("python", f"Run this Python locally?\n\n{preview}")
        if not approved:
            ctx.policy.audit("python", "declined", {"script": code[:200]})
            return ToolResult("The user declined to run this script.", ok=False)

    timeout = max(1, min(int(timeout_s or ctx.cfg.security.python_timeout_s), 600))
    script = ctx.policy.workspace / ".jarvis_scratch" / f"run_{_stamp()}.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(code, encoding="utf-8")
    code_rc, stdout, stderr = run_capture(
        [sys.executable, str(script)],
        cwd=ctx.policy.workspace,
        timeout=timeout,
        env=ctx.policy.scrubbed_env(),
    )
    ctx.policy.audit("python", "run", {"exit": code_rc, "bytes": len(code)})
    limit = ctx.cfg.security.max_output_chars
    parts = [f"exit code: {code_rc}"]
    if stdout:
        parts.append("stdout:\n" + _clip(stdout, limit))
    if stderr:
        parts.append("stderr:\n" + _clip(stderr, limit // 2))
    if not stdout and not stderr:
        parts.append("(no output)")
    return ToolResult("\n".join(parts), ok=code_rc == 0)


def _stamp() -> str:
    import datetime as dt

    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.6)]
    tail = text[-int(limit * 0.35):]
    return f"{head}\n… [{human_bytes(len(text) - limit)} of output elided] …\n{tail}"


TOOLS = [
    Tool(
        name="run_shell",
        description=(
            "Run a shell command in the workspace and return exit code, stdout and stderr. "
            "Read-only commands run directly; anything that changes files asks for confirmation."
        ),
        parameters=obj(
            {
                "command": arg("The shell command to run.", "string"),
                "timeout_s": arg("Timeout in seconds (max 600).", "integer"),
            },
            ["command"],
        ),
        func=run_shell,
        category="shell",
        needs_approval=True,  # refined by policy inside the tool
        agents=("code",),
    ),
    Tool(
        name="run_python",
        description="Execute a Python script locally in a subprocess and return its output.",
        parameters=obj(
            {
                "code": arg("Complete Python source to execute.", "string"),
                "timeout_s": arg("Timeout in seconds (max 600).", "integer"),
            },
            ["code"],
        ),
        func=run_python,
        category="shell",
        needs_approval=True,
        agents=("code",),
    ),
]
