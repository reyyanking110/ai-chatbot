from __future__ import annotations

from jarvis_local.memory import MemoryStore, StateStore
from jarvis_local.voice import VoiceEngine, clean_for_speech


def test_memory_crud(sandbox):
    store = MemoryStore(sandbox["cfg"].state_db)
    result = store.remember("Prefers short bullet summaries", "preference")
    assert result.ok
    duplicate = store.remember("Prefers short bullet summaries", "preference")
    assert "already" in duplicate.content.lower()
    assert store.count() == 1
    found = store.search("bullet")
    assert "summaries" in found
    row_id = int(found.split("]")[0].strip("["))
    assert store.forget(row_id).ok
    assert store.count() == 0
    store.close()


def test_memory_relevance_recall(sandbox):
    store = MemoryStore(sandbox["cfg"].state_db)
    store.remember("Uses Arch Linux on the desktop")
    store.remember("Hates merged cell spreadsheets")
    relevant = store.relevant("which distro should I install on my laptop?")
    assert any("Arch" in text for text in relevant)
    store.close()


def test_state_sessions(sandbox):
    store = StateStore(sandbox["cfg"].state_db)
    sid = store.create_session("chat", "hello")
    store.add_message(sid, "user", "hi")
    store.add_message(sid, "assistant", "hello!")
    history = store.history(sid)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert store.latest_session("chat") == sid
    store.set_fact("k", {"a": 1})
    assert store.get_fact("k") == {"a": 1}
    store.close()


def test_clean_for_speech():
    text = "## Heading\n\n**Bold** and `code` and [link](http://x) and [1]\n\n```py\nprint(1)\n```"
    cleaned = clean_for_speech(text)
    assert "##" not in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "http" not in cleaned
    assert "print(1)" not in cleaned


def test_voice_detects_none_gracefully(sandbox):
    cfg = sandbox["cfg"]
    cfg.voice.engine = "none"
    engine = VoiceEngine(cfg)
    assert engine.engine() == "none"
    assert "No TTS" in engine.describe()
