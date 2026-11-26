"""The research agent: answers from local documents, with citations."""

from __future__ import annotations

import re

from ..config import Config
from ..documents import citation_label, format_context
from .base import Agent, Prepared, register

CITATION_RULES = """
# Answering with citations
The user's local library is the only source of truth for this question.

- Cite every factual claim with its source number in square brackets: [1], [2].
- Only cite sources that appear under "Sources". Never invent [7] if you were given three.
- If the sources do not contain the answer, say clearly: "I couldn't find that in your
  documents." You may then offer a general answer, marked "From general knowledge:".
- Prefer quoting a short phrase over long excerpts.
- Do not cite a file path that is not in the sources.
"""


class ResearchAgent(Agent):
    name = "research"
    title = "Research assistant"
    description = (
        "Searches your indexed local documents (PDF, Markdown, DOCX, text...) and answers "
        "with inline citations showing file and page/line."
    )

    def model_for(self, cfg: Config) -> str:
        return cfg.models.general

    def system_prompt(self, ctx) -> str:
        return super().system_prompt(ctx) + CITATION_RULES

    def prepare(self, ctx, user_input: str) -> Prepared:
        system = self.system_prompt(ctx)
        memories = ctx.memory.relevant(user_input, limit=4)
        if memories:
            system += "\n# What you remember about the user\n" + "\n".join(f"- {m}" for m in memories)
        manager = getattr(ctx, "skill_manager", None)
        if manager is not None:
            hints = manager.prompt_hints(user_input, self.name)
            if hints:
                system += "\n\n# Installed skills\n" + hints

        hits = []
        if ctx.library is not None:
            hits = ctx.library.search(user_input, top_k=8)
        context, used = format_context(hits, max_chars=_context_chars(ctx.cfg))
        question = getattr(ctx, "research_question", None) or user_input
        if used:
            system += (
                "\n# Task\nAnswer this question using the sources below, citing them as [n]:\n"
                f"\"{question}\"\n"
                "\n# Sources\nEach source has a number, a title with location, and a path.\n\n"
                + context
            )
            user = user_input
        else:
            has_docs = ctx.library is not None and ctx.library.has_documents()
            if has_docs:
                user = (
                    user_input
                    + "\n\n(Retrieval found no matching passages in the library. "
                    "Tell the user this specific question isn't covered by their documents, "
                    "then suggest what to index or rephrase.)"
                )
            else:
                user = (
                    user_input
                    + "\n\n(The document library is empty. Tell the user to run "
                    "`jarvis library index ~/Documents` first, then ask again.)"
                )
        # The research agent may also search the library itself for follow-ups.
        return Prepared(system=system, user=user, hits=used)

    @staticmethod
    def format_sources(hits) -> str:
        if not hits:
            return ""
        lines = []
        for i, hit in enumerate(hits, 1):
            lines.append(f"[{i}] {citation_label(hit)} — `{hit.path}`")
        return "\n".join(lines)


def _context_chars(cfg: Config) -> int:
    # ~4 characters per token; leave room for the question and the answer.
    budget_tokens = int(cfg.models.num_ctx * 0.45)
    return max(2000, min(budget_tokens * 4, 24000))


def extract_citations(text: str, hits) -> list[dict]:
    """Map [n] markers in an answer back to real sources, for the CLI footer."""
    used = sorted({int(m) for m in re.findall(r"\[(\d{1,2})\]", text)})
    out = []
    for number in used:
        if 1 <= number <= len(hits):
            hit = hits[number - 1]
            out.append(
                {
                    "number": number,
                    "title": hit.title,
                    "path": hit.path,
                    "page": hit.page,
                    "line": hit.line,
                }
            )
    return out


register(ResearchAgent())
