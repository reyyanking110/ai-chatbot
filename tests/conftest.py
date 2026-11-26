from __future__ import annotations

import pytest


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Isolated config/data/workspace for every test."""
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    library = tmp_path / "documents"
    config_dir.mkdir()
    workspace.mkdir()
    library.mkdir()
    monkeypatch.setenv("JARVIS_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("JARVIS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("JARVIS_CONFIG_FILE", str(config_dir / "config.toml"))
    monkeypatch.chdir(tmp_path)

    from jarvis_local.config import Config

    cfg = Config.load()
    cfg.paths.workspace = str(workspace)
    cfg.paths.library = [str(library)]
    cfg.backend = "mock"
    cfg.hardware.budget_gb = 8
    cfg.save()
    cfg.ensure_dirs()
    return {
        "cfg": cfg,
        "config_dir": config_dir,
        "data_dir": data_dir,
        "workspace": workspace,
        "library": library,
        "tmp": tmp_path,
    }
