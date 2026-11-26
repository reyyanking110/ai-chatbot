"""The agent runtime: context, tool loop, approvals and streaming.

A ``RunContext`` is the single object tools touch.  It carries the config,
policy, memory, library, voice engine and a callback surface so the CLI can
render steps without the agent code knowing about Rich.

The loop is deliberately simple and auditable::

    messages -> model -> tool calls -> policy/approval -> results -> model
        -> ... until a final answer or max_steps

Every step is written to the audit log.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .backends import BaseBackend, Message, estimate_tokens, extract_json
from .config import Config
from .documents import Hit, Library
from .errors import BackendError
from .memory import MemoryStore, StateStore
from .policy import Policy
from .tools.base import Registry, ToolResult
from .voice import VoiceEngine


@dataclass
class Approver:
    """Decides whether a side-effecting action may run.

    The CLI supplies one that prompts the user; batch mode supplies one that
    either auto-approves read-only work or declines everything else.
    """

    callback: Callable[[str, str], bool]

    def __call__(self, kind: str, prompt: str) -> bool:
        return self.callback(kind, prompt)


def deny_all(kind: str, prompt: str) -> bool:
    return False


def allow_all(kind: str, prompt: str) -> bool:
    return True


@dataclass
class Step:
    index: int
    kind: str  # "tool" | "final" | "error"
    name: str = ""
    detail: str = ""
    ok: bool = True
    elapsed_s: float = 0.0


@dataclass
class AgentResult:
    text: str
    steps: list[Step] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    tool_calls: int = 0
    elapsed_s: float = 0.0
    model: str = ""

    @property
    def used_tools(self) -> bool:
        return self.tool_calls > 0


@dataclass
class RunContext:
    cfg: Config
    backend: BaseBackend
    policy: Policy
    memory: MemoryStore
    state: StateStore
    voice: VoiceEngine
    registry: Registry
    approve: Callable[[str, str], bool] = deny_all
    confirm_writes: bool = True
    library: Library | None = None
    session_id: str = ""
    on_step: Callable[[Step], None] | None = None
    on_token: Callable[[str], None] | None = None
    debug: bool = False
    skill_manager: Any = None
    last_hits: list[Hit] = field(default_factory=list)
    citation_hits: list[Hit] = field(default_factory=list)  # numbered sources for [n]

    def notify(self, step: Step) -> None:
        if self.on_step:
            try:
                self.on_step(step)
            except Exception:  # noqa: BLE001 - UI must never break the loop
                pass


def build_context(
    cfg: Config,
    backend: BaseBackend,
    *,
    approve: Callable[[str, str], bool] = deny_all,
    library: Library | None = None,
    skill_manager: Any = None,
    memory: MemoryStore | None = None,
    state: StateStore | None = None,
    voice: VoiceEngine | None = None,
    registry: Registry | None = None,
    on_step: Callable[[Step], None] | None = None,
    on_token: Callable[[str], None] | None = None,
    debug: bool = False,
) -> RunContext:
    from .tools import build_default_registry

    return RunContext(
        cfg=cfg,
        backend=backend,
        policy=Policy(cfg),
        memory=memory or MemoryStore(cfg.state_db),
        state=state or StateStore(cfg.state_db),
        voice=voice or VoiceEngine(cfg),
        registry=registry or build_default_registry(),
        approve=approve,
        confirm_writes=cfg.security.approval != "auto",
        library=library,
        skill_manager=skill_manager,
        on_step=on_step,
        on_token=on_token,
        debug=debug,
    )


class AgentRuntime:
    """Runs one agent (chat, research, code, digest) to completion."""

    def __init__(self, ctx: RunContext, agent) -> None:
        self.ctx = ctx
        self.agent = agent
        self.registry = ctx.registry.subset(agent.name)
        self.model = agent.model_for(ctx.cfg)
        self.num_ctx = int(ctx.cfg.models.num_ctx or 8192)

    def _absorb_citations(self, hits: list[Hit]) -> None:
        """Number tool-provided sources so [n] in the answer maps to real files."""
        for hit in hits:
            if all(existing.chunk_id != hit.chunk_id for existing in self.ctx.citation_hits):
                self.ctx.citation_hits.append(hit)

    # ------------------------------------------------------------------ loop
    def run(self, user_input: str, history: list[Message] | None = None) -> AgentResult:
        started = time.monotonic()
        steps: list[Step] = []
        hits: list[Hit] = []

        prep = self.agent.prepare(self.ctx, user_input)
        if self.ctx.skill_manager is not None and "# Installed skills" in prep.system:
            import re as _re

            for match in _re.finditer(r"### Skill: ([a-z0-9-]+)", prep.system):
                self.ctx.skill_manager.record_use(match.group(1))
        if prep.hits:
            hits = prep.hits
            self.ctx.last_hits = prep.hits
            self.ctx.citation_hits = list(prep.hits)
        messages: list[Message] = [{"role": "system", "content": prep.system}]
        for msg in history or []:
            if msg.get("role") in {"user", "assistant"} and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prep.user})

        tool_schemas = self.registry.schemas()
        max_steps = max(1, int(self.ctx.cfg.agent.max_steps))
        total_calls = 0
        final_text = ""
        last_model = self.model

        for step_no in range(1, max_steps + 1):
            messages = _fit_context(messages, self.num_ctx)
            step_started = time.monotonic()
            try:
                result = self.ctx.backend.chat(
                    self.model,
                    messages,
                    tool_schemas if tool_schemas else None,
                    temperature=self.ctx.cfg.models.temperature,
                    options={"num_ctx": self.num_ctx},
                )
            except BackendError as exc:
                step = Step(step_no, "error", detail=exc.message, ok=False)
                steps.append(step)
                self.ctx.notify(step)
                hint = f"\n\n_{exc.hint}_" if exc.hint else ""
                extra = ""
                if self.ctx.cfg.backend == "ollama":
                    extra = (
                        "\n\nUseful first commands:\n"
                        "- `jarvis doctor` — what is installed and what is not\n"
                        "- `jarvis models pull` — download the recommended models\n"
                        "- `jarvis demo` — the whole workflow with no model at all"
                    )
                return AgentResult(
                    text=f"I couldn't reach the local model.\n\n**{exc.message}**{hint}{extra}",
                    steps=steps,
                    hits=hits,
                    tool_calls=total_calls,
                    elapsed_s=time.monotonic() - started,
                    model=last_model,
                )
            last_model = result.model or self.model

            if not result.tool_calls:
                final_text = result.content.strip()
                steps.append(Step(step_no, "final", elapsed_s=result.elapsed_s))
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": result.tool_calls,
                }
            )
            for call in result.tool_calls:
                total_calls += 1
                name = call.get("function", {}).get("name", "")
                args = call.get("function", {}).get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                step_started = time.monotonic()
                tool_msg, step = self._execute(call.get("id", f"call_{total_calls}"), name, args)
                step.index = step_no
                step.elapsed_s = time.monotonic() - step_started
                steps.append(step)
                self.ctx.notify(step)
                messages.append(tool_msg)
                if name == "search_library":
                    self._absorb_citations(_tool_hits(tool_msg))

        if not final_text:
            if any(s.kind == "tool" for s in steps):
                final_text = (
                    "I hit the step limit while working on that. Here is what I did so far — "
                    "ask me to continue, or raise agent.max_steps (`jarvis config set agent.max_steps 12`)."
                )
            else:
                final_text = "I could not produce an answer for that. Try rephrasing, or run `jarvis doctor`."

        return AgentResult(
            text=final_text,
            steps=steps,
            hits=self.ctx.citation_hits or hits,
            tool_calls=total_calls,
            elapsed_s=time.monotonic() - started,
            model=last_model,
        )

    # -------------------------------------------------------------- tool exec
    def _execute(self, call_id: str, name: str, args: dict[str, Any]) -> tuple[Message, Step]:
        tool = self.registry.get(name)
        if tool is None:
            detail = f"Unknown tool '{name}'. Available: {', '.join(self.registry.tools) or 'none'}."
            result = ToolResult(detail, ok=False)
            step = Step(0, "tool", name=name, detail=detail, ok=False)
        else:
            step = Step(0, "tool", name=name, detail=_summarise_args(args))
            approved = True
            if tool.needs_approval and self.ctx.confirm_writes:
                approved = self.ctx.approve(name, _approval_prompt(tool, args))
            if not approved:
                result = ToolResult("The user declined this action.", ok=False)
                step.ok = False
                step.detail = "declined by user"
            else:
                result = tool.call(self.ctx, args)
                step.ok = result.ok
                step.detail = _first_line(result.content)
            self.ctx.policy.audit(
                "agent:" + self.agent.name,
                f"tool:{name}",
                {"args": _safe_audit_args(tool.name if tool else name, args), "ok": result.ok},
            )
        return result.to_message(call_id, name), step


# Keys that can contain user content or secrets: audit their shape, not values.
_CONTENT_KEYS = {"content", "text", "code", "body", "request", "script", "query", "pattern", "command"}


def _safe_audit_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Describe a tool call without copying file contents or message text."""
    safe: dict[str, Any] = {}
    for key, value in list(args.items())[:8]:
        if key in _CONTENT_KEYS:
            if isinstance(value, str):
                safe[key] = f"<{len(value)} chars>"
            else:
                safe[key] = f"<{type(value).__name__}>"
        elif key in {"path", "source", "destination", "url", "memory_id", "top_k", "timeout_s"}:
            text = str(value)
            safe[key] = text[:160] + ("…" if len(text) > 160 else "")
        else:
            safe[key] = f"<{type(value).__name__}>"
    return safe


def _tool_hits(tool_msg: Message) -> list[Hit]:
    """Recover Hit objects attached by the search_library tool, if any."""
    hits = tool_msg.get("_hits")
    return list(hits) if hits else []


def _approval_prompt(tool, args: dict[str, Any]) -> str:
    if tool.name in {"write_file", "append_file"}:
        path = args.get("path", "?")
        content = str(args.get("content", ""))
        lines = content.count("\n") + 1 if content else 0
        return f"Write {lines} lines to `{path}`?"
    if tool.name == "delete_file":
        return f"Delete `{args.get('path', '?')}`? This cannot be undone."
    if tool.name == "move_file":
        return f"Move `{args.get('source', '?')}` → `{args.get('destination', '?')}`?"
    if tool.name == "forget":
        return f"Delete memory #{args.get('memory_id', '?')}?"
    return f"Allow `{tool.name}` with {json.dumps(args, default=str)[:160]}?"


def _summarise_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    bits = []
    for key, value in list(args.items())[:3]:
        text = str(value).replace("\n", "⏎")
        bits.append(f"{key}={text[:60]}{'…' if len(text) > 60 else ''}")
    return ", ".join(bits)


def _first_line(text: str, limit: int = 160) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:limit] + ("…" if len(line) > limit else "")
    return ""


def _fit_context(messages: list[Message], num_ctx: int) -> list[Message]:
    """Trim older history if the prompt is getting close to the context window."""
    budget = max(512, int(num_ctx * 0.75))
    total = sum(estimate_tokens(str(m.get("content") or "")) for m in messages)
    if total <= budget or len(messages) <= 3:
        return messages
    system = messages[0]
    tail = messages[1:]
    # Keep complete tool-call exchanges: never orphan a tool result from its call.
    while tail and total > budget:
        removed = tail.pop(0)
        total -= estimate_tokens(str(removed.get("content") or ""))
        if removed.get("role") == "assistant" and removed.get("tool_calls"):
            while tail and tail[0].get("role") == "tool":
                orphan = tail.pop(0)
                total -= estimate_tokens(str(orphan.get("content") or ""))
    return [system, *tail]


# ---------------------------------------------------------------------------
# Shared helpers used by agents
# ---------------------------------------------------------------------------

def json_answer(text: str) -> Any | None:
    return extract_json(text)


def truncate_for_prompt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated {len(text) - limit} characters]"
