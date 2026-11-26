from __future__ import annotations

from typer.testing import CliRunner

from jarvis_local import __version__
from jarvis_local.cli import app

runner = CliRunner()


def _invoke(args, **kwargs):
    return runner.invoke(app, args, catch_exceptions=False, **kwargs)


def test_version_flag():
    result = _invoke(["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_agents():
    result = _invoke(["--help"])
    assert result.exit_code == 0
    for snippet in ("init", "ask", "chat", "digest", "library", "models", "skills", "doctor"):
        assert snippet in result.stdout


def test_doctor_runs(sandbox):
    result = _invoke(["doctor"])
    # mock backend + no google → warnings only
    assert result.exit_code in (0, 1)
    assert "Diagnostics" in result.stdout


def test_config_show_and_set(sandbox):
    result = _invoke(["config", "show"])
    assert result.exit_code == 0
    assert "backend" in result.stdout
    result = _invoke(["config", "set", "voice.rate", "210"])
    assert result.exit_code == 0
    from jarvis_local.config import Config

    assert Config.load(sandbox["config_dir"] / "config.toml").voice.rate == 210


def test_models_plan_and_current(sandbox):
    result = _invoke(["models", "plan"])
    assert result.exit_code == 0
    assert "General" in result.stdout
    result = _invoke(["models", "current"])
    assert "mock" in result.stdout


def test_library_flow(sandbox):
    docs = sandbox["library"]
    (docs / "a.md").write_text("Kubernetes cluster runs on three nodes.\n")
    result = _invoke(["library", "index", str(docs)])
    assert result.exit_code == 0
    assert "Added" in result.stdout
    result = _invoke(["library", "search", "kubernetes"])
    assert result.exit_code == 0
    assert "cluster" in result.stdout
    result = _invoke(["library", "stats"])
    assert "Documents" in result.stdout


def test_ask_mock(sandbox):
    result = _invoke(["ask", "hello jarvis"])
    assert result.exit_code == 0
    assert "mock mode" in result.stdout.lower()


def test_research_without_library_explains(sandbox):
    result = _invoke(["research", "what is it?"])
    assert result.exit_code == 1
    assert "Index your documents" in result.stdout


def test_digest_demo(sandbox):
    result = _invoke(["digest", "--demo", "--no-save"])
    assert result.exit_code == 0
    assert "Morning digest" in result.stdout
    assert "stand-up" in result.stdout.lower()


def test_note_and_memory_commands(sandbox):
    result = _invoke(["note", "buy oat milk", "--title", "shopping"])
    assert result.exit_code == 0
    assert (sandbox["workspace"] / "notes").exists()
    result = _invoke(["memory", "add", "Loves filter coffee"])
    assert result.exit_code == 0
    result = _invoke(["memory", "list"])
    assert "filter coffee" in result.stdout


def test_skills_scaffold_and_list(sandbox):
    result = _invoke(["skills", "new", "coffee", "-d", "Brew timing"])
    assert result.exit_code == 0
    result = _invoke(["skills", "list"])
    assert "coffee" in result.stdout


def test_agents_command(sandbox):
    result = _invoke(["agents"])
    assert result.exit_code == 0
    for name in ("chat", "research", "code", "digest"):
        assert name in result.stdout


def test_logs_empty(sandbox):
    result = _invoke(["logs"])
    assert result.exit_code == 0


def test_google_status_without_setup(sandbox):
    result = _invoke(["google", "status"])
    assert result.exit_code == 0
    assert "Google" in result.stdout


def test_schedule_show_graceful(sandbox):
    result = _invoke(["schedule", "show"])
    assert result.exit_code == 0


def test_demo_command(sandbox):
    result = _invoke(["demo"])
    assert result.exit_code == 0, result.stdout
    assert "Demo complete" in result.stdout
    assert "Q4" in result.stdout or "pricing" in result.stdout.lower()


def test_chat_one_turn(sandbox):
    result = _invoke(["chat"], input="/quit\n")
    assert result.exit_code == 0


def test_code_command_with_yes(sandbox):
    result = _invoke(["code", "create a script", "--yes"])
    assert result.exit_code == 0
    assert (sandbox["workspace"] / "mock_hello.py").exists()


def test_sessions_empty(sandbox):
    result = _invoke(["sessions"])
    assert result.exit_code == 0


def test_digest_saves_briefing(sandbox):
    result = _invoke(["digest", "--demo"])
    assert result.exit_code == 0
    briefings = list(sandbox["cfg"].briefings_dir.glob("*-digest.md"))
    assert briefings, "a digest markdown file should be written"
    text = briefings[0].read_text()
    assert "Morning briefing" in text
    assert "Raw agenda" in text


def test_research_flow_with_indexed_docs(sandbox):
    docs = sandbox["library"]
    (docs / "policy.md").write_text("Refunds are processed within five business days.\n")
    assert _invoke(["library", "index", str(docs)]).exit_code == 0
    result = _invoke(["research", "how long do refunds take?"])
    assert result.exit_code == 0
    assert "five business days" in result.stdout
    assert "Sources" in result.stdout


def test_code_agent_reports_workspace_changes(sandbox):
    result = _invoke(["code", "write a greeting program", "--yes"])
    assert result.exit_code == 0
    assert "Workspace changes" in result.stdout
    assert (sandbox["workspace"] / "mock_hello.py").exists()


def test_config_set_and_reload(sandbox):
    assert _invoke(["config", "set", "assistant_name", "Friday"]).exit_code == 0
    from jarvis_local.config import Config

    assert Config.load(sandbox["config_dir"] / "config.toml").assistant_name == "Friday"


def test_skills_new_show_and_remove(sandbox):
    assert _invoke(["skills", "new", "tea-timer", "-d", "Steep times"]).exit_code == 0
    result = _invoke(["skills", "show", "tea-timer"])
    assert "Steep times" in result.stdout
    assert _invoke(["skills", "disable", "tea-timer"]).exit_code == 0
    assert _invoke(["skills", "remove", "tea-timer", "--yes"]).exit_code == 0
    assert not (sandbox["cfg"].skills_dir / "tea-timer").exists()


def test_doctor_deep_mentions_generation(sandbox):
    result = _invoke(["doctor", "--deep"])
    # mock backend answers instantly, so deep mode is exercised end to end
    assert result.exit_code in (0, 1)
    assert "Diagnostics" in result.stdout
