"""The general chat agent with memory."""

from __future__ import annotations

from ..config import Config
from .base import Agent, Prepared, register


class ChatAgent(Agent):
    name = "chat"
    title = "General assistant"
    description = (
        "Everyday conversation with access to your notes, memory and the machine's clock. "
        "Knows when to hand off research or coding to the specialists."
    )

    def model_for(self, cfg: Config) -> str:
        return cfg.models.general

    def system_prompt(self, ctx) -> str:
        base = super().system_prompt(ctx)
        return (
            base
            + """
# Your role
You are the everyday assistant. Answer directly for general questions. Use tools when they
help (time, machine info, notes, memory). If the user asks a question that clearly needs
their own documents, say so and suggest `jarvis research "<question>"`. If they want code
written or run, suggest `jarvis code`.

# Memory
When the user tells you a stable preference or personal fact ("I use Arch", "call me Ray",
"my manager is Ada"), save it with the `remember` tool, one short sentence per fact. Never
store secrets, passwords or one-off trivia.
"""
        )

    def prepare(self, ctx, user_input: str) -> Prepared:
        prepared = super().prepare(ctx, user_input)
        notes_hint = ctx.policy.workspace / "notes"
        if notes_hint.exists() and any(notes_hint.glob("*.md")):
            prepared.system += (
                "\nThe user keeps markdown notes in `notes/` inside the workspace. "
                "You can list and read them with the file tools."
            )
        return prepared


register(ChatAgent())
