from __future__ import annotations

from pathlib import Path

from jarvis_local.documents import Library, chunk_text, citation_label, format_context


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_chunking_overlap_and_bounds():
    text = "alpha beta gamma. " * 400
    chunks = chunk_text(text, chunk_chars=400, overlap=80)
    assert len(chunks) > 3
    assert all(len(c.text) <= 420 for c in chunks)
    assert chunks[0].text.startswith("alpha")
    # consecutive chunks share content
    assert chunks[0].text[-40:] in text


def test_chunk_respects_paragraph_breaks():
    intro = "One short paragraph that sets the scene."
    text = intro + "\n\n" + ("Second paragraph sentence. " * 60)
    chunks = chunk_text(text, 300, 50)
    # The first chunk should end at the paragraph break, not mid-sentence.
    assert chunks[0].text.startswith(intro)
    assert chunks[0].text.strip() == intro
    assert chunks[1].text.startswith("Second paragraph")


def test_index_and_search(sandbox):
    docs = sandbox["library"]
    _write(docs / "pricing.md", "# Pricing\n\nThe Pro tier stays at $29/month for existing customers.\n")
    _write(docs / "notes.txt", "Remember to buy milk and call the dentist tomorrow morning.\n")
    cfg = sandbox["cfg"]
    lib = Library(cfg)
    stats = lib.index([docs])
    assert stats.added == 2
    assert stats.chunks >= 2
    hits = lib.search("Pro tier price", top_k=3)
    assert hits
    assert "pricing" in hits[0].path
    assert "29" in hits[0].text
    lib.close()


def test_reindex_is_incremental(sandbox):
    docs = sandbox["library"]
    _write(docs / "a.md", "first version content about gardening\n")
    cfg = sandbox["cfg"]
    lib = Library(cfg)
    lib.index([docs])
    stats = lib.index([docs])
    assert stats.unchanged == 1 and stats.added == 0
    # edit → update
    _write(docs / "a.md", "second version content about astronomy\n")
    stats = lib.index([docs])
    assert stats.updated == 1
    hits = lib.search("astronomy")
    assert hits and "astronomy" in hits[0].text
    lib.close()


def test_removed_file_pruned(sandbox):
    docs = sandbox["library"]
    target = docs / "gone.md"
    _write(target, "temporary notes about bicycles\n")
    cfg = sandbox["cfg"]
    lib = Library(cfg)
    lib.index([docs])
    assert lib.document_count() == 1
    target.unlink()
    stats = lib.index([docs])
    assert stats.removed == 1
    assert lib.document_count() == 0
    lib.close()


def test_csv_tsv_extraction(sandbox):
    docs = sandbox["library"]
    _write(docs / "budget.csv", "item,amount\nrent,42000\ngroceries,18000\n")
    cfg = sandbox["cfg"]
    lib = Library(cfg)
    stats = lib.index([docs])
    assert stats.added == 1
    hits = lib.search("groceries amount")
    assert hits and "groceries" in hits[0].text
    lib.close()


def test_citations_and_context():
    from jarvis_local.documents import Hit

    hits = [
        Hit(1, 1, "/x/a.md", "a", 0, "alpha text", None, 12, 0.5),
        Hit(2, 1, "/x/b.md", "b", 1, "beta text", 3, None, 0.4),
    ]
    assert "line 12" in citation_label(hits[0])
    assert "p. 3" in citation_label(hits[1])
    context, used = format_context(hits)
    assert "[1]" in context and "[2]" in context
    assert len(used) == 2


def test_search_empty_library(sandbox):
    lib = Library(sandbox["cfg"])
    assert lib.search("anything") == []
    assert not lib.has_documents()
    lib.close()


def test_html_and_json_are_textual(sandbox):
    docs = sandbox["library"]
    _write(docs / "page.html", "<html><body><h1>Ignore</h1><p>Persimmon harvest notes</p></body></html>")
    _write(docs / "data.json", '{"project": "apricot", "deadline": "2026-12-01"}')
    lib = Library(sandbox["cfg"])
    stats = lib.index([docs])
    assert stats.added == 2
    assert lib.search("persimmon")
    assert lib.search("apricot")
    lib.close()


def test_multilingual_substring_search(sandbox):
    docs = sandbox["library"]
    _write(docs / "memo.md", "قرار الميزانية: زيادة 10 بالمئة في الربع الرابع.\n")
    _write(docs / "japanese.txt", "東京本社の会議は十二月三日です。\n")
    _write(docs / "english.md", "The budget decision was approved in December.\n")
    cfg = sandbox["cfg"]
    lib = Library(cfg)
    lib.index([docs])
    # Arabic-script words match with or without the definite article.
    assert lib.search("ميزانية")
    assert lib.search("الميزانية")
    # Short CJK words work through the trigram path.
    assert lib.search("会議")
    assert lib.search("東京")
    assert lib.search("budget decision")
    lib.close()


def test_trigram_table_cleared_on_rebuild(sandbox):
    docs = sandbox["library"]
    _write(docs / "a.md", "東京の会議メモ\n")
    cfg = sandbox["cfg"]
    lib = Library(cfg)
    lib.index([docs])
    assert lib.search("会議")
    lib.index([docs], rebuild=True)
    assert lib.search("会議")  # rebuilt from scratch, still found
    lib.close()


def test_embedding_failure_degrades_gracefully(sandbox):
    """A dead embedding backend must not stop text indexing."""
    docs = sandbox["library"]
    _write(docs / "a.md", "The kiln fires at 1200 degrees.\n")
    cfg = sandbox["cfg"]

    class BrokenEmbedder:
        model = "nomic-embed-text"

        def embed(self, texts):
            raise RuntimeError("connection refused")

    lib = Library(cfg, embedder=BrokenEmbedder())
    stats = lib.index([docs])
    assert stats.added == 1
    assert stats.chunks >= 1
    assert stats.embed_error
    assert lib.search("kiln")
    lib.close()


def test_pending_vectors_are_filled_later(sandbox):
    """Keyword indexing first, embeddings added on the next run."""
    docs = sandbox["library"]
    _write(docs / "b.md", "Cedar planks weather well outdoors.\n")
    cfg = sandbox["cfg"]
    lib = Library(cfg, embedder=None)
    lib.index([docs])
    # no vectors yet
    assert lib.stats()["vectors"] == 0

    from jarvis_local.backends import LocalHashEmbedder

    lib2 = Library(cfg, embedder=LocalHashEmbedder(dims=768))
    stats = lib2.index([docs])
    assert stats.embedded >= 1
    assert lib2.stats()["vectors"] >= 1
    lib2.close()
    lib.close()
