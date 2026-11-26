"""Agent definitions.

Built-in agents register themselves in :data:`jarvis_local.agents.base.agent_registry`
when this package is imported.
"""

from __future__ import annotations

from .base import Agent, AgentRegistry, Prepared, agent_registry, get_agent, list_agents, register

# Importing the modules below registers each agent (see their decorators).
from .chat import ChatAgent
from .code import CodeAgent
from .digest import DigestAgent
from .research import ResearchAgent

__all__ = [
    "Agent",
    "AgentRegistry",
    "Prepared",
    "agent_registry",
    "get_agent",
    "list_agents",
    "register",
    "ChatAgent",
    "CodeAgent",
    "DigestAgent",
    "ResearchAgent",
]
