"""Long-term local memory.

Two layers:

* :class:`MemoryStore` — durable facts in SQLite with full-text search.  This
  is what the ``remember`` / ``search_memory`` tools use.
* :class:`StateStore` — sessions, conversation history and per-agent
  bookkeeping for ``jarvis chat`` and the digest scheduler.

Everything is a plain SQLite file in your data directory.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .tools.base import ToolResult

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'fact',
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used TEXT,
    uses INTEGER DEFAULT 0,
    UNIQUE(text)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content='memories', content_rowid='id', tokenize='porter unicode61');

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_name TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

STOP = re.compile(r"\b(the|a|an|is|are|what|who|when|where|do|does|did|my|i|about|remember)\b", re.I)


class MemoryStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def remember(self, text: str, kind: str = "fact") -> ToolResult:
        text = " ".join(str(text).split()).strip()
        if not text:
            return ToolResult("Nothing to remember (empty text).", ok=False)
        if len(text) > 1000:
            text = text[:1000]
        now = dt.datetime.now().isoformat(timespec="seconds")
        existing = self.conn.execute("SELECT id FROM memories WHERE text=?", (text,)).fetchone()
        if existing:
            return ToolResult(f"Already remembered (id {existing['id']}): {text}")
        cur = self.conn.execute(
            "INSERT INTO memories(kind, text, created_at) VALUES(?,?,?)", (kind, text, now)
        )
        self.conn.commit()
        return ToolResult(f"Remembered (id {cur.lastrowid}): {text}")

    def search(self, query: str = "", limit: int = 10) -> str:
        limit = max(1, min(int(limit or 10), 50))
        if query and query.strip():
            terms = [t for t in re.findall(r"[\w'-]+", query) if len(t) > 1 and not STOP.fullmatch(t)]
            rows: list[sqlite3.Row] = []
            if terms:
                match = " OR ".join(f'"{t}"*' for t in terms[:8])
                try:
                    rows = self.conn.execute(
                        "SELECT m.* FROM memories_fts f JOIN memories m ON m.id=f.rowid "
                        "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                        (match, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            if not rows:
                like = f"%{query.strip()}%"
                rows = self.conn.execute(
                    "SELECT * FROM memories WHERE text LIKE ? ORDER BY created_at DESC LIMIT ?",
                    (like, limit),
                ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        if not rows:
            return "No stored memories match."
        ids = [row["id"] for row in rows]
        self.conn.execute(
            f"UPDATE memories SET uses=uses+1, last_used=? WHERE id IN ({','.join('?' * len(ids))})",
            (dt.datetime.now().isoformat(timespec="seconds"), *ids),
        )
        self.conn.commit()
        return "\n".join(
            f"[{row['id']}] ({row['kind']}, {row['created_at'][:10]}) {row['text']}" for row in rows
        )

    def forget(self, memory_id: int) -> ToolResult:
        row = self.conn.execute("SELECT text FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            return ToolResult(f"No memory with id {memory_id}.", ok=False)
        self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.conn.commit()
        return ToolResult(f"Forgot: {row['text']}")

    def all(self, limit: int = 200) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def relevant(self, text: str, limit: int = 6) -> list[str]:
        """Cheap recall heuristic used to build the system prompt."""
        words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", text)}
        if not words:
            return []
        scored: list[tuple[int, str]] = []
        for row in self.conn.execute("SELECT text FROM memories ORDER BY uses DESC LIMIT 400"):
            hit = len(words & {w.lower() for w in re.findall(r"[A-Za-z]{4,}", row["text"])})
            if hit:
                scored.append((hit, row["text"]))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return [row["text"] for row in self.all(limit=3)]
        return [text for _, text in scored[:limit]]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])


class StateStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --------------------------------------------------------------- sessions
    def create_session(self, agent: str, title: str = "") -> str:
        session_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + f"{abs(hash(title)) % 10000:04d}"
        now = dt.datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT INTO sessions(id, agent, title, created_at, updated_at) VALUES(?,?,?,?,?)",
            (session_id, agent, title or None, now, now),
        )
        self.conn.commit()
        return session_id

    def touch_session(self, session_id: str, title: str | None = None) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        if title:
            self.conn.execute(
                "UPDATE sessions SET updated_at=?, title=COALESCE(title, ?) WHERE id=?",
                (now, title[:120], session_id),
            )
        else:
            self.conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
        self.conn.commit()

    def add_message(self, session_id: str, role: str, content: str, tool_name: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, tool_name, created_at) VALUES(?,?,?,?,?)",
            (session_id, role, content, tool_name, dt.datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def history(self, session_id: str, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT role, content, tool_name FROM messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def sessions(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS msgs "
            "FROM sessions s ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def latest_session(self, agent: str | None = None) -> str | None:
        if agent:
            row = self.conn.execute(
                "SELECT id FROM sessions WHERE agent=? ORDER BY updated_at DESC LIMIT 1", (agent,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return row["id"] if row else None

    def clear_sessions(self) -> int:
        count = int(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        self.conn.executescript("DELETE FROM messages; DELETE FROM sessions;")
        self.conn.commit()
        return count

    # ------------------------------------------------------------ key-value
    def set_fact(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO facts(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, default=str), dt.datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def get_fact(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]
