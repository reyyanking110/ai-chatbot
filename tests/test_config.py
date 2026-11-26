from __future__ import annotations

import pytest

from jarvis_local.config import Config, _serializable, dump_toml
from jarvis_local.errors import ConfigError


def test_defaults_are_sane(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_CONFIG_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "d"))
    cfg = Config.load()
    assert cfg.backend in {"ollama", "openai-compatible", "mock"}
    assert cfg.models.general
    assert cfg.workspace.name


def test_roundtrip_save_and_load(sandbox):
    cfg = sandbox["cfg"]
    cfg.assistant_name = "Friday"
    cfg.voice.rate = 200
    cfg.paths.library = ["~/Documents", "~/Notes"]
    path = cfg.save()
    assert path.exists()
    loaded = Config.load(path)
    assert loaded.assistant_name == "Friday"
    assert loaded.voice.rate == 200
    assert loaded.paths.library == ["~/Documents", "~/Notes"]


def test_set_value_coercion(sandbox):
    cfg = sandbox["cfg"]
    old, new = cfg.set_value("voice.rate", "240")
    assert (old, new) == (180, 240)
    old, new = cfg.set_value("security.allow_shell", "false")
    assert new is False
    old, new = cfg.set_value("paths.library", "~/a, ~/b")
    assert new == ["~/a", "~/b"]
    with pytest.raises(ConfigError):
        cfg.set_value("nope.nothing", "1")
    with pytest.raises(ConfigError):
        cfg.set_value("voice.rate", "fast")


def test_unknown_keys_warn_not_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "data"))
    target = tmp_path / "config.toml"
    target.write_text('backend = "mock"\n[models]\ngeneral = "x"\nbogus = 1\n[wat]\na = 1\n')
    cfg = Config.load(target)
    assert cfg.backend == "mock"
    assert cfg.models.general == "x"
    assert any("bogus" in w for w in cfg.warnings)


def test_invalid_toml_raises_helpful_error(tmp_path):
    target = tmp_path / "bad.toml"
    target.write_text("this is not toml [[[")
    with pytest.raises(ConfigError) as exc:
        Config.load(target)
    assert "jarvis init" in (exc.value.hint or "")


def test_dump_toml_roundtrip(sandbox):
    cfg = sandbox["cfg"]
    data = _serializable(cfg)
    text = dump_toml(data)
    assert "[models]" in text
    assert "general" in text
    assert "config_root" not in text


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_BACKEND", "mock")
    monkeypatch.setenv("JARVIS_OLLAMA_HOST", "http://127.0.0.1:9999")
    cfg = Config.load(tmp_path / "missing.toml")
    assert cfg.backend == "mock"
    assert cfg.ollama.host.endswith("9999")
