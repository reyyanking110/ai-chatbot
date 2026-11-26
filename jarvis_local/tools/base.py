"""Tool definitions and the registry.

Every tool declares a JSON-schema for its arguments so any local model with
tool-calling support can drive it.  Tools never execute side effects on their
own: the registry routes confirmations through the runtime, and the policy
module enforces filesystem/shell rules.
"""

from __future__ import annotations

import inspect
import json
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..errors import JarvisError


@dataclass
class ToolResult:
    content: str
    ok: bool = True
    data: Any = None
    hits: list[Any] = None  # type: ignore[assignment]  # citation sources, if any

    def __post_init__(self) -> None:
        if self.hits is None:
            self.hits = []

    def to_message(self, call_id: str, name: str) -> dict[str, Any]:
        message = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": self.content,
        }
        if self.hits:
            # Private channel: backends strip this before the model sees it.
            message["_hits"] = self.hits
        return message


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    category: str = "general"
    needs_approval: bool = False
    agents: tuple[str, ...] | None = None  # None → available to every agent

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, ctx: Any, arguments: dict[str, Any]) -> ToolResult:
        signature = inspect.signature(self.func)
        accepted = set(signature.parameters)
        kwargs = {k: v for k, v in (arguments or {}).items() if k in accepted}
        try:
            result = self.func(ctx, **kwargs)
        except JarvisError as exc:
            hint = f"\nHint: {exc.hint}" if exc.hint else ""
            return ToolResult(f"Error: {exc.message}{hint}", ok=False)
        except TypeError as exc:
            return ToolResult(
                f"Error: bad arguments for {self.name} ({exc}). "
                f"Expected: {', '.join(sorted(accepted - {'ctx'}))}.",
                ok=False,
            )
        except Exception as exc:  # noqa: BLE001 — surface, never crash the loop
            if getattr(ctx, "debug", False):
                return ToolResult(f"Error: {exc}\n{traceback.format_exc()[-1500:]}", ok=False)
            return ToolResult(f"Error: {exc.__class__.__name__}: {exc}", ok=False)
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, (dict, list)):
            return ToolResult(json.dumps(result, ensure_ascii=False, indent=2, default=str), data=result)
        return ToolResult(str(result))


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def subset(self, agent: str | None, include: set[str] | None = None) -> Registry:
        selected = Registry()
        for tool in self.tools.values():
            if agent and tool.agents is not None and agent not in tool.agents:
                continue
            if include is not None and tool.name not in include and tool.category != "memory":
                continue
            selected.add(tool)
        return selected

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def describe(self) -> str:
        lines = []
        for tool in self.tools.values():
            lines.append(f"- {tool.name}: {tool.description.splitlines()[0]}")
        return "\n".join(lines)

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __iter__(self):
        return iter(self.tools.values())

    def __len__(self) -> int:
        return len(self.tools)


def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def arg(desc: str, type_: str = "string", enum: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": type_, "description": desc}
    if enum:
        schema["enum"] = enum
    return schema


def build_default_registry() -> Registry:
    from . import files, knowledge, shell, system

    registry = Registry()
    for module in (files, shell, system, knowledge):
        for tool in module.TOOLS:
            registry.add(tool)
    return registry
