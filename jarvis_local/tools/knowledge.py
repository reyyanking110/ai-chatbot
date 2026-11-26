"""Knowledge tools: library search, memory, notes and speech."""

from __future__ import annotations

import datetime as dt

from ..errors import PolicyError
from ..util import safe_slug
from .base import Tool, ToolResult, arg, obj

# ---------------------------------------------------------------------------
# Library / knowledge base
# ---------------------------------------------------------------------------

def search_library(ctx, query: str, top_k: int = 6) -> ToolResult:
    if ctx.library is None:
        return ToolResult("The document library is not available in this session.", ok=False)
    hits = ctx.library.search(query, top_k=top_k)
    if not hits:
        return ToolResult(
            f"No matches in the local library for {query!r}. "
            "If you have not indexed documents yet, run `jarvis library index`.",
            ok=False,
        )
    blocks = []
    for i, hit in enumerate(hits, 1):
        from ..documents import citation_label

        blocks.append(
            f"[{i}] {citation_label(hit)} — {hit.path}\n{hit.text.strip()[:900]}"
        )
    ctx.last_hits = hits
    return ToolResult("\n\n".join(blocks), data=[h.path for h in hits], hits=hits)


def open_document(ctx, path: str, max_chars: int = 6000) -> ToolResult:
    """Read more of a cited document when the first chunk is not enough."""

    target = ctx.policy.resolve_read(path)
    if not target.exists():
        raise PolicyError(f"No such document: {path}")
    row = None
    if ctx.library is not None:
        row = ctx.library.conn.execute(
            "SELECT id FROM documents WHERE path=?", (str(target),)
        ).fetchone()
    if row and ctx.library is not None:
        from ..documents import Hit, citation_label

        chunks = ctx.library.chunks_for_doc(row["id"], limit=12)
        text = "\n\n".join(c["text"] for c in chunks)
        hits = [
            Hit(
                chunk_id=-1, doc_id=row["id"], path=str(target), title=target.stem,
                ordinal=0, text=chunks[0]["text"] if chunks else "", page=None, line=None, score=1.0,
            )
        ]
    else:

        text = target.read_text(encoding="utf-8", errors="replace")
        hits = []
    if not hits:
        return ToolResult(text[:max_chars])
    return ToolResult(f"{citation_label(hits[0])} — {target}\n---\n{text[:max_chars]}")


def take_note(ctx, text: str, title: str = "") -> ToolResult:
    notes_dir = ctx.policy.resolve_write(ctx.policy.workspace / "notes")
    notes_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now()
    name = f"{stamp:%Y-%m-%d_%H%M}-{safe_slug(title or text[:40], 40)}.md"
    path = notes_dir / name
    header = f"# {title.strip() or stamp.strftime('%d %b %Y, %H:%M')}\n\n_{stamp:%Y-%m-%d %H:%M}_\n\n"
    path.write_text(header + text.strip() + "\n", encoding="utf-8")
    ctx.policy.audit("notes", "create", {"path": str(path)})
    return ToolResult(f"Saved note → {path.relative_to(ctx.policy.workspace)}")


def list_notes(ctx, limit: int = 10) -> str:
    notes_dir = ctx.policy.workspace / "notes"
    if not notes_dir.exists():
        return "No notes yet — use `take_note` or `jarvis note \"...\"`."
    files = sorted(notes_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return "No notes yet."
    rows = []
    for path in files[: max(1, min(int(limit), 50))]:
        title = path.stem
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            title = first.lstrip("# ").strip() or title
        except OSError:
            pass
        rows.append(f"{dt.datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M}  {title}  ({path.name})")
    return "\n".join(rows)


TOOLS = [
    Tool(
        name="search_library",
        description=(
            "Search the user's indexed local documents (notes, PDFs, docs) and return "
            "the most relevant passages with file paths and page/line anchors."
        ),
        parameters=obj(
            {
                "query": arg("Natural-language question or keywords.", "string"),
                "top_k": arg("How many passages to return (1-15).", "integer"),
            },
            ["query"],
        ),
        func=search_library,
        category="memory",
        agents=("research",),
    ),
    Tool(
        name="open_document",
        description="Open an indexed document by path to read a longer excerpt.",
        parameters=obj(
            {"path": arg("Absolute path from a citation.", "string"), "max_chars": arg("Maximum characters.", "integer")},
            ["path"],
        ),
        func=open_document,
        category="memory",
        agents=("research",),
    ),
    Tool(
        name="remember",
        description=(
            "Save a durable fact about the user (preferences, names, projects) to local memory. "
            "Use short, self-contained statements."
        ),
        parameters=obj(
            {
                "text": arg("The fact to remember, e.g. 'Prefers concise bullet summaries'.", "string"),
                "kind": arg("Category of memory.", "string", ["fact", "preference", "person", "project"]),
            },
            ["text"],
        ),
        func=lambda ctx, text, kind="fact": ctx.memory.remember(text, kind),
        category="memory",
    ),
    Tool(
        name="search_memory",
        description="Look up previously remembered facts about the user.",
        parameters=obj(
            {"query": arg("Search text; leave empty for the most recent memories.", "string"),
             "limit": arg("Maximum results.", "integer")}
        ),
        func=lambda ctx, query="", limit=10: ctx.memory.search(query, limit),
        category="memory",
    ),
    Tool(
        name="forget",
        description="Delete a memory by id (get ids from search_memory).",
        parameters=obj({"memory_id": arg("Numeric id of the memory to delete.", "integer")}, ["memory_id"]),
        func=lambda ctx, memory_id: ctx.memory.forget(int(memory_id)),
        category="memory",
        needs_approval=True,
    ),
    Tool(
        name="take_note",
        description="Quickly capture a note in the user's notes folder (inside the workspace).",
        parameters=obj(
            {"text": arg("Note body.", "string"), "title": arg("Optional short title.", "string")},
            ["text"],
        ),
        func=take_note,
        category="notes",
    ),
    Tool(
        name="list_notes",
        description="List recent notes.",
        parameters=obj({"limit": arg("Maximum notes to list.", "integer")}),
        func=list_notes,
        category="notes",
    ),
    Tool(
        name="speak",
        description=(
            "Convert text to speech with the local TTS engine and play it (also saved to disk). "
            "Use for read-aloud requests or spoken digests."
        ),
        parameters=obj(
            {"text": arg("Text to speak (plain text, about 1-2 minutes max).", "string")},
            ["text"],
        ),
        func=lambda ctx, text: ctx.voice.speak_rich(text),
        category="voice",
        agents=("digest", "chat"),
    ),
]
