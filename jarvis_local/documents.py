"""Local document ingestion, chunking and search.

The library lives in one SQLite file (``<data>/library.db``):

* ``documents`` — one row per indexed file (path, title, mtime, hash, ...)
* ``chunks``    — overlapping text windows with page/line anchors
* ``chunks_fts``— FTS5 index with BM25 ranking
* ``vectors``   — optional float32 embeddings for hybrid search
* ``files``     — indexed folders so re-scans are cheap

No network access, no telemetry; delete the db to forget everything.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import math
import re
import sqlite3
import struct
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .util import iter_text_files

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    indexed_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ok',
    error TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    root TEXT NOT NULL,
    title TEXT NOT NULL,
    ext TEXT NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    sha1 TEXT NOT NULL,
    pages INTEGER DEFAULT 0,
    chars INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL,
    UNIQUE(path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    page INTEGER,
    line INTEGER,
    tokens INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

-- Trigram index: substring search for scripts without spaces or stemming
-- (Chinese, Japanese, Thai) and for Arabic-script languages like Urdu where
-- the definite article attaches to the word.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_trgm USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai_trgm AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_trgm(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad_trgm AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_trgm(chunks_trgm, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TABLE IF NOT EXISTS vectors (
    chunk_id INTEGER PRIMARY KEY,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your", "about", "into", "any", "can",
}


@dataclass
class Chunk:
    text: str
    page: int | None = None
    line: int | None = None


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    path: str
    title: str
    ordinal: int
    text: str
    page: int | None
    line: int | None
    score: float
    method: str = "keyword"


@dataclass
class IndexStats:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: int = 0
    chunks: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = None  # type: ignore[assignment]
    embedded: int = 0
    embed_error: str = ""

    def __post_init__(self) -> None:
        self.errors = self.errors or []


class Library:
    def __init__(self, cfg: Config, embedder=None) -> None:
        self.cfg = cfg
        self.db_path = cfg.library_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.last_embed_error: str | None = None
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> Library:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        cur = self.conn.execute(
            "SELECT COUNT(*) AS docs, COALESCE(SUM(LENGTH(text)),0) AS chars FROM chunks"
        )
        row = cur.fetchone()
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vectors = self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        roots = [r["path"] for r in self.conn.execute("SELECT path FROM files ORDER BY path")]
        last = self.conn.execute("SELECT MAX(updated_at) FROM documents").fetchone()[0]
        embed_model = self.conn.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
        return {
            "documents": row["docs"],
            "chars": row["chars"],
            "chunks": chunks,
            "vectors": vectors,
            "sources": len(roots),
            "last_indexed": last,
            "embed_model": embed_model[0] if embed_model else None,
            "db_size": self.db_path.stat().st_size if self.db_path.exists() else 0,
        }

    def has_documents(self) -> bool:
        return self.conn.execute("SELECT 1 FROM documents LIMIT 1").fetchone() is not None

    # ------------------------------------------------------------------ ingest
    def index(
        self,
        roots: Sequence[Path] | None = None,
        progress=None,
        rebuild: bool = False,
    ) -> IndexStats:
        import time

        started = time.monotonic()
        stats = IndexStats()
        roots = list(roots) if roots else self.cfg.library_roots()
        if rebuild:
            # Deleting from the content tables fires the AFTER DELETE triggers,
            # which remove the matching rows from both FTS indexes. Deleting
            # from the FTS tables directly would corrupt external-content tables.
            self.conn.executescript(
                "DELETE FROM vectors; DELETE FROM chunks; DELETE FROM documents; DELETE FROM files;"
            )
            self.conn.commit()

        pending_embed = bool(self.cfg.library.use_embeddings and self.embedder)
        queue_embed = pending_embed
        extensions = {e.lower() for e in self.cfg.library.extensions}
        ignore = set(self.cfg.paths.ignore)
        max_bytes = self.cfg.library.max_file_mb * 1024 * 1024
        existing_roots = [r for r in roots if r.exists()]
        seen_paths: set[str] = set()

        for root in existing_roots:
            for path, size in iter_text_files([root], extensions, ignore, max_bytes):
                stats.scanned += 1
                if progress:
                    progress(stats.scanned, path)
                resolved = str(path.resolve())
                seen_paths.add(resolved)
                try:
                    mtime = path.stat().st_mtime
                    row = self.conn.execute(
                        "SELECT mtime, size, status FROM files WHERE path=?", (resolved,)
                    ).fetchone()
                    if row and abs(row["mtime"] - mtime) < 0.5 and row["size"] == size and row["status"] == "ok":
                        if pending_embed and self._doc_needs_vectors(resolved):
                            # Text is already indexed; only the vectors are missing.
                            self._touch_file(resolved, mtime, size)
                            stats.updated += 1
                            queue_embed = True
                            continue
                        stats.unchanged += 1
                        continue
                    chunks = extract_chunks(path, self.cfg.library.chunk_chars, self.cfg.library.chunk_overlap)
                    if not chunks:
                        self._mark_file(resolved, mtime, size, "empty")
                        stats.failed += 1
                        stats.errors.append(f"{path.name}: no extractable text")
                        continue
                    is_update = row is not None
                    self._store_document(path, root, chunks, is_update)
                    self._mark_file(resolved, mtime, size, "ok")
                    stats.chunks += len(chunks)
                    queue_embed = queue_embed or pending_embed
                    if is_update:
                        stats.updated += 1
                    else:
                        stats.added += 1
                except Exception as exc:  # noqa: BLE001 — keep indexing the rest
                    stats.failed += 1
                    stats.errors.append(f"{path.name}: {exc}")
                    try:
                        self._mark_file(resolved, 0, 0, "error", str(exc)[:300])
                    except sqlite3.Error:
                        pass
            self.conn.commit()

        # Prune documents whose files disappeared from scanned roots.
        for row in self.conn.execute("SELECT path FROM documents").fetchall():
            doc_path = Path(row["path"])
            in_scanned_root = any(_is_within(doc_path, root) for root in existing_roots)
            if in_scanned_root and str(doc_path) not in seen_paths and not doc_path.exists():
                self._delete_document(row["path"])
                stats.removed += 1

        self.conn.commit()
        if queue_embed and self.embedder:
            stats.embedded = self._embed_missing(progress=None)
            if getattr(self, "last_embed_error", None):
                stats.embed_error = self.last_embed_error
        stats.elapsed_s = round(time.monotonic() - started, 1)
        return stats

    def _touch_file(self, path: str, mtime: float, size: int) -> None:
        self.conn.execute(
            "UPDATE files SET indexed_at=? WHERE path=?", (_now(), path)
        )

    def _doc_needs_vectors(self, path: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM chunks c JOIN documents d ON d.id=c.doc_id "
            "LEFT JOIN vectors v ON v.chunk_id=c.id "
            "WHERE d.path=? AND v.chunk_id IS NULL LIMIT 1",
            (path,),
        ).fetchone()
        return row is not None

    def _mark_file(self, path: str, mtime: float, size: int, status: str, error: str = "") -> None:
        self.conn.execute(
            "INSERT INTO files(path, mtime, size, indexed_at, status, error) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, "
            "indexed_at=excluded.indexed_at, status=excluded.status, error=excluded.error",
            (path, mtime, size, _now(), status, error),
        )

    def _store_document(self, path: Path, root: Path, chunks: list[Chunk], is_update: bool) -> None:
        text_size = sum(len(c.text) for c in chunks)
        pages = len({c.page for c in chunks if c.page})
        sha1 = hashlib.sha1(str(path).encode()).hexdigest()[:16]
        existing = self.conn.execute("SELECT id FROM documents WHERE path=?", (str(path),)).fetchone()
        if existing:
            doc_id = existing["id"]
            old_ids = [r["id"] for r in self.conn.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
            for chunk_id in old_ids:
                self.conn.execute("DELETE FROM vectors WHERE chunk_id=?", (chunk_id,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.conn.execute(
                "UPDATE documents SET title=?, ext=?, mtime=?, size=?, sha1=?, pages=?, chars=?, updated_at=? "
                "WHERE id=?",
                (
                    path.stem, path.suffix.lower(), path.stat().st_mtime, path.stat().st_size,
                    sha1, pages, text_size, _now(), doc_id,
                ),
            )
        else:
            cur = self.conn.execute(
                "INSERT INTO documents(path, root, title, ext, mtime, size, sha1, pages, chars, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    str(path), str(root), path.stem, path.suffix.lower(), path.stat().st_mtime,
                    path.stat().st_size, sha1, pages, text_size, _now(),
                ),
            )
            doc_id = int(cur.lastrowid)
        self.conn.executemany(
            "INSERT INTO chunks(doc_id, ordinal, text, page, line, tokens) VALUES(?,?,?,?,?,?)",
            [
                (doc_id, i, chunk.text, chunk.page, chunk.line, len(chunk.text.split()))
                for i, chunk in enumerate(chunks)
            ],
        )

    def _delete_document(self, path: str) -> None:
        row = self.conn.execute("SELECT id FROM documents WHERE path=?", (path,)).fetchone()
        if not row:
            return
        doc_id = row["id"]
        for chunk in self.conn.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,)):
            self.conn.execute("DELETE FROM vectors WHERE chunk_id=?", (chunk["id"],))
        self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.conn.execute("DELETE FROM files WHERE path=?", (path,))

    # ------------------------------------------------------------------ vectors
    def _embed_missing(self, progress=None) -> int:
        if not self.embedder:
            return 0
        model = self.embedder.model
        rows = self.conn.execute(
            "SELECT c.id, c.text FROM chunks c LEFT JOIN vectors v ON v.chunk_id=c.id "
            "WHERE v.chunk_id IS NULL LIMIT 4000"
        ).fetchall()
        if not rows:
            return 0
        done = 0
        batch_size = 8
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            try:
                vectors = self.embedder.embed([r["text"] for r in batch])
            except Exception as exc:  # noqa: BLE001
                # Keyword search still works, so never lose the indexed text:
                # stop embedding, record why, and carry on.
                self.last_embed_error = str(exc)
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('embed_error', ?)", (str(exc)[:300],),
                )
                self.conn.commit()
                break
            for row, vector in zip(batch, vectors, strict=False):
                if not vector:
                    continue
                dim = len(vector)
                self.conn.execute(
                    "INSERT OR REPLACE INTO vectors(chunk_id, dim, vector, model) VALUES(?,?,?,?)",
                    (row["id"], dim, _pack(vector), model),
                )
                done += 1
            self.conn.commit()
            if progress:
                progress(done, len(rows))
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('embed_model', ?)", (model,)
        )
        self.conn.commit()
        return done

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: int = 6, mode: str = "hybrid") -> list[Hit]:
        query = (query or "").strip()
        if not query or not self.has_documents():
            return []
        top_k = max(1, min(int(top_k), 30))
        keyword_hits = self._keyword_search(query, top_k * 4)
        vector_hits: list[Hit] = []
        if mode in ("hybrid", "semantic") and self.embedder and self._has_vectors():
            try:
                vector_hits = self._vector_search(query, top_k * 4)
            except Exception:
                vector_hits = []
        if mode == "semantic" and not vector_hits:
            # Fall back gracefully so research never silently returns nothing.
            return keyword_hits[:top_k]
        if not vector_hits:
            return keyword_hits[:top_k]
        return _fuse(keyword_hits, vector_hits, top_k)

    def _has_vectors(self) -> bool:
        return self.conn.execute("SELECT 1 FROM vectors LIMIT 1").fetchone() is not None

    def _keyword_search(self, query: str, limit: int) -> list[Hit]:
        if _has_nonlatin(query):
            return self._substring_search(query, limit)
        terms = [t for t in re.findall(r"[A-Za-z0-9_'-]+", query.lower()) if t not in STOPWORDS and len(t) > 1]
        if not terms:
            return []
        match = " OR ".join(_quote_fts(t) for t in terms[:12])
        sql = (
            "SELECT c.id AS chunk_id, c.doc_id, c.ordinal, c.text, c.page, c.line, d.path, d.title, "
            "bm25(chunks_fts) AS rank FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.rowid "
            "JOIN documents d ON d.id = c.doc_id "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?"
        )
        try:
            rows = self.conn.execute(sql, (match, limit)).fetchall()
        except sqlite3.OperationalError:
            # Very defensive fallback for odd tokenisation edge cases.
            rows = self.conn.execute(
                "SELECT c.id AS chunk_id, c.doc_id, c.ordinal, c.text, c.page, c.line, d.path, d.title, "
                "-1.0 AS rank FROM chunks c JOIN documents d ON d.id=c.doc_id "
                "WHERE c.text LIKE ? LIMIT ?",
                (f"%{terms[0]}%", limit),
            ).fetchall()
        hits = []
        for i, row in enumerate(rows):
            # bm25 returns negative numbers (more negative = better); normalise 0..1
            score = 1.0 / (1.0 + i) + (0.5 if _phrase_bonus(query, row["text"]) else 0)
            hits.append(
                Hit(
                    chunk_id=row["chunk_id"], doc_id=row["doc_id"], path=row["path"], title=row["title"],
                    ordinal=row["ordinal"], text=row["text"], page=row["page"], line=row["line"],
                    score=float(score), method="keyword",
                )
            )
        return hits

    def _substring_search(self, query: str, limit: int) -> list[Hit]:
        """Trigram match for CJK, Arabic-script and mixed-script queries."""
        phrase = " ".join(query.split())[:120]
        if not phrase:
            return []
        sql = (
            "SELECT c.id AS chunk_id, c.doc_id, c.ordinal, c.text, c.page, c.line, d.path, d.title "
            "FROM chunks_trgm JOIN chunks c ON c.id = chunks_trgm.rowid "
            "JOIN documents d ON d.id = c.doc_id "
            "WHERE chunks_trgm MATCH ? ORDER BY bm25(chunks_trgm) LIMIT ?"
        )
        # Trigram needs 3+ characters. For 1–2 character CJK words, OR together
        # the short terms and let the LIKE fallback catch anything else.
        if len(phrase) < 3:
            short_terms = [t for t in phrase.split() if t]
            if short_terms:
                match = " OR ".join(f'"{t}"' for t in short_terms)
                try:
                    rows = self.conn.execute(sql, (match, limit)).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            else:
                rows = []
        else:
            try:
                rows = self.conn.execute(sql, ('"' + phrase.replace('"', "") + '"', limit)).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            # Fallback for very short queries or a missing trigram table.
            like = f"%{phrase}%"
            rows = self.conn.execute(
                "SELECT c.id AS chunk_id, c.doc_id, c.ordinal, c.text, c.page, c.line, d.path, d.title "
                "FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.text LIKE ? LIMIT ?",
                (like, limit),
            ).fetchall()
        hits = []
        for i, row in enumerate(rows):
            hits.append(
                Hit(
                    chunk_id=row["chunk_id"], doc_id=row["doc_id"], path=row["path"], title=row["title"],
                    ordinal=row["ordinal"], text=row["text"], page=row["page"], line=row["line"],
                    score=1.0 / (1.0 + i), method="substring",
                )
            )
        return hits

    def _vector_search(self, query: str, limit: int) -> list[Hit]:
        vectors = self.embedder.embed([query]) if self.embedder else []
        if not vectors:
            return []
        qvec = vectors[0]
        scored: list[tuple[float, sqlite3.Row]] = []
        rows = self.conn.execute(
            "SELECT v.chunk_id, v.dim, v.vector, c.doc_id, c.ordinal, c.text, c.page, c.line, d.path, d.title "
            "FROM vectors v JOIN chunks c ON c.id=v.chunk_id JOIN documents d ON d.id=c.doc_id"
        ).fetchall()
        for row in rows:
            vector = _unpack(row["vector"], row["dim"])
            if len(vector) != len(qvec):
                continue
            scored.append((_cosine(qvec, vector), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = []
        for score, row in scored[:limit]:
            hits.append(
                Hit(
                    chunk_id=row["chunk_id"], doc_id=row["doc_id"], path=row["path"], title=row["title"],
                    ordinal=row["ordinal"], text=row["text"], page=row["page"], line=row["line"],
                    score=float(score), method="semantic",
                )
            )
        return hits

    def chunks_for_doc(self, doc_id: int, limit: int = 40) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunks WHERE doc_id=? ORDER BY ordinal LIMIT ?", (doc_id, limit)
        ).fetchall()

    def document_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_chunks(path: Path, chunk_chars: int, overlap: int) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_chunks(path, chunk_chars, overlap)
    if suffix == ".docx":
        return _docx_chunks(path, chunk_chars, overlap)
    if suffix == ".epub":
        return _epub_chunks(path, chunk_chars, overlap)
    if suffix in {".html", ".htm"}:
        text = _html_to_text(path.read_text(encoding="utf-8", errors="replace"))
        return chunk_text(text, chunk_chars, overlap)
    if suffix == ".json":
        raw = path.read_text(encoding="utf-8", errors="replace")
        return chunk_text(raw, chunk_chars, overlap)
    if suffix == ".csv" or suffix == ".tsv":
        return _delimited_chunks(path, chunk_chars, overlap)
    text = path.read_text(encoding="utf-8", errors="replace")
    if "\x00" in text[:1000]:
        raise ValueError("binary file")
    return chunk_text(text, chunk_chars, overlap)


_NONLATIN = re.compile(r"[^\x00-\x7F]")


def _has_nonlatin(text: str) -> bool:
    return bool(_NONLATIN.search(text))


def chunk_text(text: str, chunk_chars: int, overlap: int, page: int | None = None, base_line: int | None = None) -> list[Chunk]:
    text = _normalise_text(text)
    if not text.strip():
        return []
    chunk_chars = max(200, int(chunk_chars))
    overlap = max(0, min(int(overlap), chunk_chars // 2))
    chunks: list[Chunk] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_chars)
        if end < length:
            window = text[start:end]
            # Prefer semantic boundaries, but require the cut to be far enough in
            # that we don't emit tiny chunks (paragraph breaks may cut earlier).
            for sep, minimum in (("\n\n", 0.12), ("\n", 0.50), (". ", 0.55), (" ", 0.60)):
                cut = window.rfind(sep)
                if cut > chunk_chars * minimum:
                    end = start + cut + len(sep)
                    break
        body = text[start:end].strip()
        if body:
            line = None
            if base_line is not None:
                line = base_line + text[:start].count("\n")
            chunks.append(Chunk(body, page=page, line=line))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
        # With small chunks the overlap can reach back into a short previous
        # paragraph; snap forward to the next paragraph break if one is near.
        if text[max(0, start - 1) : start + 1] != "\n\n":
            nxt = text.find("\n\n", start, min(length, start + overlap))
            if nxt != -1:
                start = nxt + 2
    return chunks


def _normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def _pdf_chunks(path: Path, chunk_chars: int, overlap: int) -> list[Chunk]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDFs: pip install 'jarvis-local[docs]'") from exc
    # Warn about encrypted PDFs up front; pypdf can read with an empty password
    # but many bank statements and payslips are locked.
    with path.open("rb") as fh:
        if b"/Encrypt" in fh.read(4096):
            raise RuntimeError("PDF is encrypted; unlock it before indexing")
    try:
        reader = PdfReader(str(path))
        pages = list(reader.pages)
    except Exception as exc:  # noqa: BLE001 - pypdf raises various errors
        raise RuntimeError(f"unreadable PDF ({exc.__class__.__name__})") from exc
    chunks: list[Chunk] = []
    for index, page in enumerate(pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        chunks.extend(chunk_text(text, chunk_chars, overlap, page=index))
    if not chunks:
        raise RuntimeError("no text layer (scanned PDF?); OCR it first")
    return chunks


def _docx_chunks(path: Path, chunk_chars: int, overlap: int) -> list[Chunk]:
    try:
        import docx  # type: ignore
    except ImportError:
        # Minimal fallback: docx is a zip of XML; extract paragraph text.
        try:
            with zipfile.ZipFile(path) as archive:
                xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
            xml = re.sub(r"</w:p>", "\n\n", xml)
            text = html.unescape(re.sub(r"<[^>]+>", "", xml))
            return chunk_text(text, chunk_chars, overlap)
        except (KeyError, zipfile.BadZipFile) as exc:
            raise RuntimeError("Could not read .docx file") from exc
    document = docx.Document(str(path))
    text = "\n\n".join(p.text for p in document.paragraphs)
    return chunk_text(text, chunk_chars, overlap)


def _epub_chunks(path: Path, chunk_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist() if n.lower().endswith((".xhtml", ".html", ".htm")))
        for name in names:
            raw = archive.read(name).decode("utf-8", errors="replace")
            chunks.extend(chunk_text(_html_to_text(raw), chunk_chars, overlap))
    return chunks


def _delimited_chunks(path: Path, chunk_chars: int, overlap: int) -> list[Chunk]:
    import csv

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        rows = []
        for i, row in enumerate(reader):
            rows.append(" | ".join(cell.strip() for cell in row))
            if i > 5000:
                break
    if not rows:
        return []
    header = rows[0]
    return chunk_text(header + "\n" + "\n".join(rows[1:]), chunk_chars, overlap)


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", raw)
    text = re.sub(r"<[^>]+>", " ", raw)
    return html.unescape(text)


# ---------------------------------------------------------------------------
# Fusion / maths helpers
# ---------------------------------------------------------------------------

def _fuse(keyword: list[Hit], semantic: list[Hit], top_k: int) -> list[Hit]:
    scores: dict[int, float] = {}
    store: dict[int, Hit] = {}
    k = 60.0
    for rank, hit in enumerate(keyword, 1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0) + 1.0 / (k + rank)
        store.setdefault(hit.chunk_id, hit)
    for rank, hit in enumerate(semantic, 1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0) + 1.2 / (k + rank)
        store.setdefault(hit.chunk_id, hit)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    fused = []
    for chunk_id, score in ranked:
        hit = store[chunk_id]
        fused.append(
            Hit(
                chunk_id=hit.chunk_id, doc_id=hit.doc_id, path=hit.path, title=hit.title,
                ordinal=hit.ordinal, text=hit.text, page=hit.page, line=hit.line,
                score=score, method=f"hybrid:{hit.method}",
            )
        )
    return fused


def _phrase_bonus(query: str, text: str) -> bool:
    phrase = query.strip().strip('"“”').lower()
    return len(phrase) > 8 and phrase in text.lower()


def _quote_fts(term: str) -> str:
    if any(ch in term for ch in '":*()'):
        return '"' + term.replace('"', "") + '"'
    return f'"{term}"' if "-" in term else term


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes, dim: int) -> list[float]:
    try:
        return list(struct.unpack(f"<{dim}f", blob))
    except struct.error:
        return []


def embedder_matches(embedder, stored_model: str | None) -> bool:
    """Never mix vectors from different embedding models in one index."""
    if embedder is None or not stored_model:
        return embedder is None and not stored_model
    return str(getattr(embedder, "model", "")).split(":")[0] == str(stored_model).split(":")[0]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def citation_label(hit: Hit) -> str:
    """Human-friendly citation like ``notes.md §3 (p. 4)``."""
    location = ""
    if hit.page:
        location = f"p. {hit.page}"
    elif hit.line:
        location = f"line {hit.line}"
    else:
        location = f"§{hit.ordinal + 1}"
    return f"{hit.title} · {location}"


def format_context(hits: Iterable[Hit], max_chars: int = 6000) -> tuple[str, list[Hit]]:
    """Number the sources so the model can cite them as [1], [2], ..."""
    lines: list[str] = []
    used: list[Hit] = []
    total = 0
    for index, hit in enumerate(hits, 1):
        block = (
            f"[{index}] {citation_label(hit)}\n"
            f"file: {hit.path}\n"
            f"---\n{hit.text.strip()}\n"
        )
        if total + len(block) > max_chars:
            break
        lines.append(block)
        used.append(hit)
        total += len(block)
    return "\n".join(lines), used
