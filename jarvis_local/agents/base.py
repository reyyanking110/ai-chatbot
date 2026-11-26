"""Agent base class and prompt pieces shared by every agent."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import Config
from ..documents import Hit

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import RunContext


@dataclass
class Prepared:
    """Everything an agent wants to put in front of the model for one turn."""

    system: str
    user: str
    hits: list[Hit] = field(default_factory=list)


class Agent:
    name: str = "base"
    title: str = "Base agent"
    description: str = "Abstract agent"
    tools: tuple[str, ...] | None = None  # None → all tools allowed for this agent
    lean_and_local: bool = True

    def model_for(self, cfg: Config) -> str:
        return cfg.models.general

    def describe(self) -> str:
        return self.description

    def system_prompt(self, ctx: RunContext) -> str:
        return BASE_PERSONA.format(
            name=ctx.cfg.assistant_name,
            today=dt.datetime.now().strftime("%A %d %B %Y"),
            time=dt.datetime.now().strftime("%H:%M"),
        )

    def prepare(self, ctx: RunContext, user_input: str) -> Prepared:
        system = self.system_prompt(ctx)
        memories = ctx.memory.relevant(user_input, limit=6)
        if memories:
            system += "\n\n# What you remember about the user\n" + "\n".join(
                f"- {m}" for m in memories
            )
        manager = getattr(ctx, "skill_manager", None)
        if manager is not None:
            hints = manager.prompt_hints(user_input, self.name)
            if hints:
                system += (
                    "\n\n# Installed skills\n"
                    "These local skills may help. Follow their instructions when relevant.\n\n"
                    + hints
                )
        return Prepared(system=system, user=user_input)


BASE_PERSONA = """\
You are {name}, a private AI assistant running entirely on the user's own machine.
Today is {today}, and the local time is {time}.

Ground rules:
- Be concise and useful. Prefer short paragraphs and tight bullet lists.
- Never invent facts. If you do not know, say so plainly.
- You do not send data anywhere: no cloud APIs, no telemetry. Everything you do is local.
- The user's data is sensitive. Do not paste large secrets or file contents back unless asked.
- When you use a tool, summarise what you did in one line, then give the answer.
- Write in the user's language and match their tone.
"""


class AgentRegistry(dict):
    def names(self) -> list[str]:
        return sorted(self.keys())


agent_registry = AgentRegistry()


def register(agent: Agent) -> Agent:
    agent_registry[agent.name] = agent
    return agent


def get_agent(name: str) -> Agent:
    from ..errors import JarvisError

    if name not in agent_registry:
        raise JarvisError(
            f"Unknown agent '{name}'.",
            hint=f"Available: {', '.join(agent_registry.names())}",
        )
    return agent_registry[name]


def list_agents() -> list[Agent]:
    return [agent_registry[name] for name in agent_registry.names()]
