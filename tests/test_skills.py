from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from jarvis_local.errors import SkillError
from jarvis_local.skills import (
    SkillManager,
    install_default_skills,
    parse_skill_markdown,
    scan_folder,
)

VALID = """---
name: weather
description: Local weather lookup
version: 1.2.0
author: ray
triggers: [weather, forecast]
run: scripts/weather.py
---

Ask for a city, then run the forecast script with it.
"""


def _make_skill_folder(base: Path, name="weather", script=True) -> Path:
    folder = base / name
    folder.mkdir(parents=True)
    text = VALID.replace("weather", name, 1)
    (folder / "SKILL.md").write_text(text, encoding="utf-8")
    if script:
        (folder / "scripts").mkdir()
        (folder / "scripts" / "weather.py").write_text("print('sunny')\n")
    return folder


def test_parse_frontmatter():
    skill = parse_skill_markdown(VALID)
    assert skill.name == "weather"
    assert skill.version == "1.2.0"
    assert skill.triggers == ["weather", "forecast"]
    assert "forecast script" in skill.body


def test_parse_rejects_missing_frontmatter():
    with pytest.raises(SkillError):
        parse_skill_markdown("# just a readme")


def test_parse_rejects_bad_name():
    bad = VALID.replace("name: weather", "name: WEATHER BAD!")
    with pytest.raises(SkillError):
        parse_skill_markdown(bad)


def test_parse_requires_description():
    bad = "---\nname: ok-name\n---\nbody"
    with pytest.raises(SkillError):
        parse_skill_markdown(bad)


def test_scan_detects_risk(tmp_path):
    folder = _make_skill_folder(tmp_path)
    (folder / "scripts" / "weather.py").write_text(
        "import requests\nAPI_KEY = 'sk-live-123456'\n"
    )
    report = scan_folder(folder)
    assert report.network and report.credentials
    assert report.risk == "high"


def test_scan_prompt_only_is_low_risk(tmp_path):
    folder = _make_skill_folder(tmp_path, script=False)
    report = scan_folder(folder)
    assert report.risk == "low"


def test_install_local_folder(sandbox):
    cfg = sandbox["cfg"]
    folder = _make_skill_folder(sandbox["tmp"])
    manager = SkillManager(cfg)
    skill = manager.install(str(folder), confirm=lambda s, r: True)
    assert skill.name == "weather"
    assert (cfg.skills_dir / "weather" / "SKILL.md").exists()
    assert manager.get("weather") is not None


def test_install_respects_refusal(sandbox):
    cfg = sandbox["cfg"]
    folder = _make_skill_folder(sandbox["tmp"])
    manager = SkillManager(cfg)
    with pytest.raises(SkillError):
        manager.install(str(folder), confirm=lambda s, r: False)
    assert manager.get("weather") is None


def test_duplicate_install_requires_force(sandbox):
    cfg = sandbox["cfg"]
    folder = _make_skill_folder(sandbox["tmp"])
    manager = SkillManager(cfg)
    manager.install(str(folder), confirm=lambda s, r: True)
    with pytest.raises(SkillError):
        manager.install(str(folder), confirm=lambda s, r: True)
    manager.install(str(folder), confirm=lambda s, r: True, force=True)


def test_enable_disable_remove(sandbox):
    cfg = sandbox["cfg"]
    folder = _make_skill_folder(sandbox["tmp"])
    manager = SkillManager(cfg)
    manager.install(str(folder), confirm=lambda s, r: True)
    manager.set_enabled("weather", False)
    assert manager.get("weather").enabled is False
    assert manager.enabled() == []
    manager.set_enabled("weather", True)
    assert len(manager.enabled()) == 1
    manager.remove("weather")
    assert manager.get("weather") is None
    with pytest.raises(SkillError):
        manager.remove("weather")


def test_prompt_hints_match_triggers(sandbox):
    cfg = sandbox["cfg"]
    folder = _make_skill_folder(sandbox["tmp"])
    manager = SkillManager(cfg)
    manager.install(str(folder), confirm=lambda s, r: True)
    assert "weather" in manager.prompt_hints("what's the weather in Karachi?").lower()
    assert manager.prompt_hints("write a poem") == ""


def test_trigger_matching_is_word_aware():
    from jarvis_local.skills import _trigger_matches

    assert _trigger_matches("time in", "what time is it in karachi?")
    assert _trigger_matches("world clock", "check the world clock")
    assert not _trigger_matches("time in", "what a lovely time")
    assert not _trigger_matches("weather", "write a poem")
    assert _trigger_matches("call time", "best call time?")


def test_prompt_hints_respect_agent_filter(sandbox):
    cfg = sandbox["cfg"]
    folder = _make_skill_folder(sandbox["tmp"])
    manager = SkillManager(cfg)
    manager.install(str(folder), confirm=lambda s, r: True)
    cfg.skills.inject_agents = ["chat"]
    assert manager.prompt_hints("weather?", agent="chat")
    assert manager.prompt_hints("weather?", agent="research") == ""


def test_skill_tool_is_generated(sandbox):
    cfg = sandbox["cfg"]
    folder = _make_skill_folder(sandbox["tmp"])
    manager = SkillManager(cfg)
    manager.install(str(folder), confirm=lambda s, r: True)
    tools = manager.skill_tools()
    assert len(tools) == 1
    assert tools[0].name == "skill_weather"


def test_project_archive_rejected(sandbox):
    cfg = sandbox["cfg"]
    manager = SkillManager(cfg)
    staging = sandbox["tmp"] / "payload.zip"
    with zipfile.ZipFile(staging, "w") as zf:
        zf.writestr("README.md", "no skill here")
    with pytest.raises(SkillError) as exc:
        manager.install(str(staging), confirm=lambda s, r: True)
    assert "SKILL.md" in exc.value.message


def test_zip_slip_blocked(sandbox):
    cfg = sandbox["cfg"]
    manager = SkillManager(cfg)
    staging = sandbox["tmp"] / "evil.zip"
    with zipfile.ZipFile(staging, "w") as zf:
        zf.writestr("../../evil.txt", "pwned")
        zf.writestr("SKILL.md", VALID)
    with pytest.raises(SkillError):
        manager.install(str(staging), confirm=lambda s, r: True)


def test_valid_zip_installs(sandbox):
    cfg = sandbox["cfg"]
    manager = SkillManager(cfg)
    staging = sandbox["tmp"] / "good.zip"
    with zipfile.ZipFile(staging, "w") as zf:
        zf.writestr("weather/SKILL.md", VALID)
        zf.writestr("weather/scripts/weather.py", "print('sunny')\n")
    skill = manager.install(str(staging), confirm=lambda s, r: True)
    assert skill.name == "weather"


def test_scaffold_and_defaults(sandbox):
    cfg = sandbox["cfg"]
    manager = SkillManager(cfg)
    folder = manager.scaffold("my-skill", "does a thing")
    assert (folder / "SKILL.md").exists()
    created = install_default_skills(cfg)
    assert "daily-journal" in created
    assert len(manager.list()) >= 4


def test_export_roundtrip(sandbox):
    from jarvis_local.skills import export_skill

    cfg = sandbox["cfg"]
    folder = _make_skill_folder(sandbox["tmp"])
    manager = SkillManager(cfg)
    skill = manager.install(str(folder), confirm=lambda s, r: True)
    archive = export_skill(skill, sandbox["tmp"] / "weather.zip")
    assert archive.exists()
    assert zipfile.ZipFile(archive).namelist()


# ---------------------------------------------------------------------------
# Collections (offline: archives built in the test)
# ---------------------------------------------------------------------------

COLLECTION = {
    "repo-main/alpha/SKILL.md": "---\nname: alpha\ndescription: First skill\ntriggers: [alpha]\n---\nDo alpha things.",
    "repo-main/beta/SKILL.md": "---\nname: beta\ndescription: Second skill\ntriggers: [beta]\n---\nDo beta things.",
    "repo-main/beta/scripts/run.py": "print('beta')\n",
    "repo-main/README.md": "# a collection",
}


def _collection_zip(tmp_path):
    path = tmp_path / "collection.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, body in COLLECTION.items():
            zf.writestr(name, body)
    return path


def test_collection_requires_selection(sandbox):
    cfg = sandbox["cfg"]
    archive = _collection_zip(sandbox["tmp"])
    manager = SkillManager(cfg)
    with pytest.raises(SkillError) as exc:
        manager.install(str(archive), confirm=lambda s, r: True)
    assert "collection" in exc.value.message.lower()
    assert "--list" in (exc.value.hint or "")


def test_collection_subpath_selection(sandbox):
    cfg = sandbox["cfg"]
    archive = _collection_zip(sandbox["tmp"])
    manager = SkillManager(cfg)
    skill = manager.install(f"{archive}#beta", confirm=lambda s, r: True)
    assert skill.name == "beta"
    assert (cfg.skills_dir / "beta" / "scripts" / "run.py").exists()


def test_collection_bad_subpath(sandbox):
    cfg = sandbox["cfg"]
    archive = _collection_zip(sandbox["tmp"])
    manager = SkillManager(cfg)
    with pytest.raises(SkillError) as exc:
        manager.install(f"{archive}#gamma", confirm=lambda s, r: True)
    assert "gamma" in exc.value.message


def test_list_remote_collection(sandbox):
    cfg = sandbox["cfg"]
    archive = _collection_zip(sandbox["tmp"])
    manager = SkillManager(cfg)
    entries = manager.list_remote(str(archive))
    names = {entry["name"] for entry in entries}
    assert names == {"alpha", "beta"}
    assert all(entry["description"] for entry in entries)


def test_skill_folder_size_guard(sandbox):
    cfg = sandbox["cfg"]
    folder = _make_skill_folder(sandbox["tmp"])
    (folder / "huge.bin").write_bytes(b"0" * (33 * 1024 * 1024))
    manager = SkillManager(cfg)
    with pytest.raises(SkillError) as exc:
        manager.install(str(folder), confirm=lambda s, r: True)
    assert "MB" in exc.value.message


def test_github_url_normalisation():
    from jarvis_local.skills import _normalise_github_url

    assert _normalise_github_url("https://github.com/o/r") == "https://codeload.github.com/o/r/tar.gz/HEAD"
    assert (
        _normalise_github_url("https://github.com/o/r/tree/dev")
        == "https://codeload.github.com/o/r/tar.gz/refs/heads/dev"
    )
    assert _normalise_github_url("https://example.com/a.zip") == "https://example.com/a.zip"


def test_usage_tracking(sandbox):
    cfg = sandbox["cfg"]
    folder = _make_skill_folder(sandbox["tmp"])
    manager = SkillManager(cfg)
    manager.install(str(folder), confirm=lambda s, r: True)
    manager.record_use("weather")
    manager.record_use("weather")
    assert manager.usage()["weather"]["uses"] == 2
    assert manager.usage()["weather"]["last_used"]


def test_suggest_lessons_from_audit(sandbox):
    cfg = sandbox["cfg"]
    manager = SkillManager(cfg)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    log = cfg.log_dir / "audit.jsonl"
    import json

    with log.open("w") as fh:
        for _ in range(4):
            fh.write(json.dumps({"actor": "agent:code", "action": "tool:write_file", "args": {"path": "x.py"}}) + "\n")
        fh.write(json.dumps({"actor": "agent:code", "action": "tool:run_shell", "args": {"command": "ls"}}) + "\n")
    suggestions = manager.suggest_lessons()
    assert any("write_file" in line for line in suggestions)
    # run_shell is deliberately excluded (too varied to wrap)
    assert not any("run_shell" in line for line in suggestions)


def test_lesson_key_ignores_paths_and_secrets():
    from jarvis_local.skills import _lesson_key

    assert _lesson_key("write_file", {"path": "/home/me/secret/plan.md", "content": "x"}) == "write_file on .md"
    assert _lesson_key("run_shell", {"command": "rm -rf /"}) == ""
    assert _lesson_key("skill_weather", {"request": "x"}) == ""
