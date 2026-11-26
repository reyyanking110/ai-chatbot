"""The coding agent: files, shell, Python — with guardrails."""

from __future__ import annotations

from ..config import Config
from .base import Agent, Prepared, register

CODE_RULES = """
# Your role
You are a careful senior engineer working inside the user's workspace: {workspace}.
You can list and read files, write files, run shell commands and execute Python.

# How to work
1. **Look before you leap.** Read the relevant files before changing them. Never guess
   at a file's contents when a tool can show you.
2. **Make the smallest change that solves the problem.** Preserve existing style.
3. **Verify.** After writing code, run it (or its tests) with `run_shell` / `run_python`
   when a quick check is possible. Report the exit code honestly.
4. **Stay inside the workspace.** Writes outside it are blocked by policy; if a task
   needs that, explain and ask the user to do it.
5. Commands that change files or state require the user's approval. In your final
   answer, list every file you created or changed and summarize the diff.
6. Never print secrets. Never run destructive commands.

# Output
End with a short summary:
- what changed (files)
- how you verified it (command + result)
- anything the user should do next
"""


class CodeAgent(Agent):
    name = "code"
    title = "Code assistant"
    description = (
        "Works with files, runs shell commands and Python, and verifies its work — all "
        "sandboxed to your workspace with approvals for anything that writes."
    )

    def model_for(self, cfg: Config) -> str:
        return cfg.models.coding

    def system_prompt(self, ctx) -> str:
        base = super().system_prompt(ctx)
        return base + CODE_RULES.format(workspace=ctx.policy.workspace)

    def prepare(self, ctx, user_input: str) -> Prepared:
        prepared = super().prepare(ctx, user_input)
        prepared.system += (
            f"\n# Machine\n{ctx.cfg.backend} backend, model {self.model_for(ctx.cfg)}. "
            f"Shell timeout {ctx.cfg.security.shell_timeout_s}s, python timeout "
            f"{ctx.cfg.security.python_timeout_s}s."
        )
        return prepared


register(CodeAgent())
