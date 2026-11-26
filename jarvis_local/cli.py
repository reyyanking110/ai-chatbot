"""The `jarvis` command line interface."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import TAGLINE, __version__
from .agents import get_agent, list_agents
from .backends import make_backend
from .config import Config, expand_path
from .documents import Library
from .errors import JarvisError
from .hardware import CATALOG, card_for, detect_hardware, human_gb
from .integrations.google import SETUP_HELP, GoogleSource, google_status
from .memory import MemoryStore, StateStore
from .runtime import AgentRuntime, Step, allow_all, build_context, deny_all
from .setup import (
    build_plan,
    client_secret_is_placeholder,
    ensure_workspace,
    install_google_libs,
    installed_models,
    model_installed,
    ollama_present,
    ollama_running,
    pull_model,
    run_doctor,
    start_ollama,
    switch_model_plan,
    write_default_config,
    write_sample_client_secret,
)
from .skills import SkillManager, export_skill, install_default_skills
from .ui import (
    _cited_sources,
    answer_panel,
    approval_panel,
    banner,
    checks_table,
    kv_table,
    make_console,
    render_result,
    section,
    sources_panel,
    spinner,
    tool_step_text,
)

app = typer.Typer(
    name="jarvis",
    help=f"{TAGLINE} — a local-first personal AI assistant for your terminal.",
    add_completion=True,
    no_args_is_help=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

library_app = typer.Typer(help="Index and search your local documents.", rich_markup_mode="rich")
models_app = typer.Typer(help="Inspect and switch local models.", rich_markup_mode="rich")
skills_app = typer.Typer(help="Install, write and manage skills.", rich_markup_mode="rich")
google_app = typer.Typer(help="Connect Gmail + Google Calendar (read-only).", rich_markup_mode="rich")
memory_app = typer.Typer(help="Manage what Jarvis remembers about you.", rich_markup_mode="rich")
config_app = typer.Typer(help="Read and change configuration.", rich_markup_mode="rich")
schedule_app = typer.Typer(help="Run the morning digest automatically.", rich_markup_mode="rich")
app.add_typer(library_app, name="library", help="Documents, notes and PDFs.")
app.add_typer(models_app, name="models", help="Local models.")
app.add_typer(skills_app, name="skills", help="Skills from GitHub or your own.")
app.add_typer(google_app, name="google", help="Gmail + Google Calendar.")
app.add_typer(memory_app, name="memory", help="Long-term memory.")
app.add_typer(config_app, name="config", help="Configuration.")
app.add_typer(schedule_app, name="schedule", help="Automatic digests.")

console = make_console()
NO_COLOR = bool(os.environ.get("NO_COLOR"))


def _console(no_color: bool = False):
    return make_console(no_color=no_color or NO_COLOR)


def _ask(prompt: str, default=None, password: bool = False):
    """Confirm/Prompt bound to our console; never blocks without a terminal."""
    if not sys.stdin.isatty():
        return default
    if password:
        return Prompt.ask(prompt, password=True, console=console, default=default)
    return Confirm.ask(prompt, default=default, console=console)


def _ask_text(prompt: str, default: str = "") -> str:
    if not sys.stdin.isatty():
        return default or ""
    return Prompt.ask(prompt, default=default, console=console, show_default=bool(default))


def shutdown_allowed(binary: str) -> bool:
    """True when an external scheduler binary exists and we are not sandboxed oddly."""
    return shutil.which(binary) is not None


def _load_config(path: str | None = None) -> Config:
    cfg = Config.load(path)
    cfg.ensure_dirs()
    return cfg


def _load_or_onboard(path: str | None = None) -> Config:
    """Load config; on a fresh machine, run the setup wizard first."""
    cfg = Config.load(path)
    if cfg.source_path and cfg.source_path.exists():
        cfg.ensure_dirs()
        return cfg
    console.print("[jarvis]First run[/jarvis] — let's get you set up. You can re-run this any time "
                  "with [bold]jarvis init[/bold].\n")
    interactive = sys.stdin.isatty()
    from .setup import RedirectedStdin

    real_stdin = sys.stdin
    try:
        if not interactive:
            sys.stdin = RedirectedStdin()  # type: ignore[assignment]
        init(
            yes=not interactive,
            backend="",
            budget=None,
            workspace=None,
            library_path=None,
            pull=interactive,
            force=True,
        )
    except (typer.Exit, typer.Abort, EOFError):
        pass
    finally:
        sys.stdin = real_stdin
    cfg = Config.load(path)
    cfg.ensure_dirs()
    return cfg


def _handle(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except JarvisError as exc:
        console.print()
        console.print(f"[fail]✗ {exc.message}[/fail]")
        if exc.hint:
            console.print(Panel(Text(exc.hint), border_style="jarvis.dim", title="What to do"))
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config_path: str | None = typer.Option(None, "--config", envvar="JARVIS_CONFIG_FILE", help="Path to config.toml"),
    version: bool = typer.Option(False, "--version", "-V", help="Show the version and exit."),
) -> None:
    """A private Jarvis for your terminal. Start with `jarvis init`."""
    global console
    if version:
        console.print(f"jarvis-local {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _welcome()
        raise typer.Exit()


def _welcome() -> None:
    cfg = Config.load()
    backend = _safe_backend_name(cfg)
    banner(
        console,
        cfg.assistant_name,
        TAGLINE,
        extra=f"{backend} · config {cfg.config_file}",
    )
    ready = (
        (cfg.backend == "mock")
        or cfg.source_path is not None
    )
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style="jarvis", no_wrap=True)
    table.add_column(style="white")
    if not ready:
        table.add_row("1.", "[bold]jarvis init[/bold] — detect hardware, pick models, set everything up")
    table.add_row("2." if not ready else "1.", "[bold]jarvis doctor[/bold] — verify models, TTS, Gmail, library")
    table.add_row("3." if not ready else "2.", '[bold]jarvis ask "what is on my plate today?"[/bold]')
    table.add_row("4." if not ready else "3.", "[bold]jarvis chat[/bold] — interactive session      [bold]jarvis demo[/bold] — no model needed")
    console.print(Panel(table, title="Get started", border_style="jarvis.dim", box=box.SQUARE, padding=(0, 2)))
    console.print("[muted]All commands:[/] [bold]jarvis --help[/bold]\n")


def _safe_backend_name(cfg: Config) -> str:
    if cfg.backend == "mock":
        return "mock backend (no model)"
    if cfg.backend == "ollama":
        return f"ollama · {cfg.models.general}" if ollama_running(cfg) else "ollama (not running)"
    return cfg.backend


# ---------------------------------------------------------------------------
# init / setup / doctor
# ---------------------------------------------------------------------------

@app.command()
def init(
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept detected defaults without prompting."),
    backend: str = typer.Option("", "--backend", help="ollama | openai-compatible | mock"),
    budget: float | None = typer.Option(None, "--budget", help="Override model memory budget in GB."),
    workspace: str | None = typer.Option(None, "--workspace", help="Folder Jarvis may write to."),
    library_path: str | None = typer.Option(None, "--library", help="Folder of documents to index now."),
    pull: bool = typer.Option(True, "--pull/--no-pull", help="Download the recommended models."),
    force: bool = typer.Option(False, "--force", help="Rewrite an existing config."),
) -> None:
    """Set up Jarvis: detect hardware, choose local models, write config."""
    cfg = Config.load()
    if cfg.source_path and cfg.source_path.exists() and not force:
        console.print(f"[warn]Config already exists at {cfg.source_path}.[/warn]")
        if not _ask("Re-run the wizard and overwrite it?", default=False):
            console.print("[muted]Keeping the existing setup. `jarvis doctor` checks it.[/muted]")
            raise typer.Exit()

    plan = build_plan(cfg, budget_gb=budget, backend=backend or None)
    profile = plan.profile

    console.print()
    banner(console, "Setting up " + cfg.assistant_name, "Everything runs on this machine. Nothing is uploaded.")
    section(console, "This machine")
    console.print(profile.summary())
    for note in profile.notes:
        console.print(f"[muted]· {note}[/muted]")

    section(console, "Recommended local models")
    console.print(f"[muted]{plan.plan.rationale}[/muted]")
    table = Table(box=None, padding=(0, 2), show_header=True, header_style="jarvis")
    table.add_column("Role", no_wrap=True)
    table.add_column("Model", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Why")
    for role, tag, why in (
        ("General (chat, research)", plan.plan.general, "tool calling + reasoning"),
        ("Coding", plan.plan.coding, "files, shell, code edits"),
        ("Fast (digest, routing)", plan.plan.fast, "low latency, small context"),
        ("Embeddings", plan.plan.embed, "local semantic search"),
    ):
        card = card_for(tag)
        table.add_row(role, tag, human_gb(card.size_gb if card else 0), why)
    console.print(table)
    console.print(
        f"[muted]Total download ≈ {human_gb(plan.plan.download_gb())} · "
        f"context window {plan.plan.num_ctx} tokens · budget {human_gb(plan.plan.budget_gb)}[/muted]"
    )
    for note in plan.plan.notes:
        console.print(f"[warn]! {note}[/warn]")

    section(console, "Defaults")
    if workspace:
        cfg.paths.workspace = workspace
    if library_path:
        cfg.paths.library = [library_path]
    rows = [
        ("Config", str(cfg.config_file)),
        ("Data", str(cfg.data_root)),
        ("Workspace (writes allowed)", str(expand_path(cfg.paths.workspace))),
        ("Document library", ", ".join(cfg.paths.library)),
        ("Approval mode", f"{cfg.security.approval} (destructive actions ask first)"),
        ("Preloaded skills", "daily-journal, file-organiser, meeting-notes"),
    ]
    console.print(kv_table(rows))

    set_up_google = False
    if not yes:
        console.print()
        if not _ask("Use these settings?", default=True):
            console.print("[muted]Nothing changed.[/muted]")
            raise typer.Exit()
        custom = _ask_text(
            "Model memory budget in GB (Enter keeps the recommendation)",
            default="",
        ).strip()
        if custom:
            try:
                plan = build_plan(cfg, budget_gb=float(custom), backend=backend or None)
            except ValueError:
                console.print("[warn]Not a number — keeping the detected budget.[/warn]")
        ws = _ask_text("Workspace folder", default=str(expand_path(cfg.paths.workspace))).strip()
        if ws:
            cfg.paths.workspace = ws
        lib = _ask_text(
            "Folder with your documents (comma-separated, blank to skip for now)",
            default=", ".join(cfg.paths.library),
        ).strip()
        cfg.paths.library = [p.strip() for p in lib.split(",") if p.strip()] if lib else []
        set_up_google = _ask(
            "Create the Google OAuth file now, so you can connect Gmail/Calendar later?",
            default=True,
        )

    if set_up_google:
        pass  # the placeholder is written below in every case
    cfg = plan.config
    if library_path:
        cfg.paths.library = [library_path]
    elif not cfg.paths.library:
        cfg.paths.library = []
    cfg.backend = backend or cfg.backend
    path = write_default_config(cfg)
    ensure_workspace(cfg)
    sample = write_sample_client_secret(cfg)
    created = install_default_skills(cfg)

    section(console, "Done")
    console.print(kv_table([
        ("Config", str(path)),
        ("Workspace", str(cfg.workspace)),
        ("Skills", ", ".join(created) if created else "already installed"),
    ]))
    if sample:
        console.print(
            f"[muted]· Google OAuth placeholder at {sample} — fill it in later with "
            "`jarvis google setup`.[/muted]"
        )

    if cfg.backend == "mock":
        console.print("\n[ok]Mock backend selected — everything works without a model.[/ok]")
        console.print('Try: [bold]jarvis demo[/bold]')
        return

    if cfg.backend == "ollama":
        section(console, "Ollama")
        if not ollama_present():
            console.print("[warn]Ollama is not installed.[/warn]")
            console.print(
                Panel(
                    Text.from_markup(
                        "Install it, then run `jarvis models pull`:\n\n"
                        "  macOS:   [bold]brew install ollama[/bold]  (or https://ollama.com/download)\n"
                        "  Linux:   [bold]curl -fsSL https://ollama.com/install.sh | sh[/bold]\n"
                        "  Windows: [bold]winget install Ollama.Ollama[/bold]"
                    ),
                    border_style="jarvis.dim",
                    title="Install Ollama",
                )
            )
        else:
            if not ollama_running(cfg):
                console.print("[muted]Starting the Ollama server…[/muted]")
                start_ollama(cfg)
            if not ollama_running(cfg):
                console.print("[fail]Ollama is installed but not answering.[/fail] Run `ollama serve`.")
            else:
                console.print(f"[ok]Ollama is running[/ok] at {cfg.ollama.host}")
                _maybe_pull_models(
                    cfg,
                    plan.plan,
                    pull and (yes or _ask(
                        f"Download the models now (~{human_gb(plan.plan.download_gb())})?",
                        default=True,
                    )),
                )

    backend_down = cfg.backend == "ollama" and not ollama_running(cfg)
    if backend_down:
        console.print(
            Panel(
                Text.from_markup(
                    "Ollama isn't answering, so I'll index your documents with **keyword search** "
                    "for now. After installing/starting it:\n\n"
                    "  [bold]jarvis models pull[/bold]          download the recommended models\n"
                    "  [bold]jarvis library index[/bold]        add the semantic vectors\n\n"
                    "You can also try everything without a model: [bold]jarvis demo[/bold]"
                ),
                title="Models pending",
                border_style="warn",
            )
        )

    # Offer to index the library right away.
    roots = cfg.library_roots()
    existing = [r for r in roots if r.exists()]
    if existing and (yes or _ask("Index your documents now for the research agent?", default=True)):
        _index_library(cfg, existing, quiet=False, allow_embeddings=not backend_down and cfg.backend != "mock")
    elif not existing:
        console.print(
            "[muted]No document folders found. Index later with "
            "`jarvis library index ~/Documents`.[/muted]"
        )

    console.print("\n[ok]Ready.[/ok] Try [bold]jarvis doctor[/bold], then [bold]jarvis ask \"hello\"[/bold].")


def _maybe_pull_models(cfg: Config, plan, do_pull: bool) -> None:
    if not do_pull:
        console.print("[muted]Skipped download. Later: `jarvis models pull`[/muted]")
        return
    tags = [t for t in plan.tags if not model_installed(cfg, t)]
    if not tags:
        console.print("[ok]All recommended models are already installed.[/ok]")
        return
    for tag in tags:
        if not pull_model(cfg, tag, on_line=lambda line: None):
            console.print(f"[warn]Could not pull {tag}.[/warn] Try `ollama pull {tag}` manually.")
    missing = [t for t in plan.tags if not model_installed(cfg, t)]
    if missing:
        console.print(f"[warn]Still missing: {', '.join(missing)}[/warn]")
    else:
        console.print("[ok]All models are installed.[/ok]")


@app.command("setup")
def setup_alias(
    yes: bool = typer.Option(True, "--yes/--interactive", help="Non-interactive by default."),
) -> None:
    """Alias for `jarvis init` (non-interactive)."""
    init(yes=yes, backend="", budget=None, workspace=None, library_path=None, pull=True, force=False)


@app.command()
def doctor(
    deep: bool = typer.Option(False, "--deep", help="Also run a test generation through the model."),
    no_color: bool = typer.Option(False, "--no-color"),
) -> None:
    """Check every moving part and tell you exactly how to fix problems."""
    out = _console(no_color)
    cfg = _load_config()
    out.print()
    banner(out, "Diagnostics", f"jarvis-local {__version__} · {cfg.backend}")
    with spinner(out, "Checking models, TTS, Gmail, library and policies…"):
        checks = run_doctor(cfg, deep=deep)
    out.print(checks_table(checks))
    fails = sum(1 for c in checks if c.status == "fail")
    warns = sum(1 for c in checks if c.status == "warn")
    out.print()
    if fails:
        out.print(f"[fail]{fails} problem{'s' if fails != 1 else ''}[/fail] and [warn]{warns} warning{'s' if warns != 1 else ''}[/warn].")
        raise typer.Exit(code=1)
    if warns:
        out.print(f"[ok]Usable[/ok] — [warn]{warns} warning{'s' if warns != 1 else ''}[/warn] above.")
    else:
        out.print("[ok]Everything checks out.[/ok]")


# ---------------------------------------------------------------------------
# Agents: ask / chat / research / code / digest
# ---------------------------------------------------------------------------

def _ensure_backend(cfg: Config) -> None:
    """Start the local Ollama daemon if it is installed but idle."""
    if cfg.backend != "ollama":
        return
    if not ollama_present() or ollama_running(cfg):
        return
    if not cfg.ollama.autostart:
        return
    console.print("[muted]Ollama is idle — starting the local daemon…[/muted]")
    if not start_ollama(cfg):
        console.print(
            "[warn]Could not start Ollama.[/warn] Run `ollama serve` yourself, "
            "or set ollama.autostart = false."
        )


def _make_runtime(cfg: Config, agent_name: str, *, approve, need_library: bool = True, on_step=None):
    _ensure_backend(cfg)
    backend = _handle(make_backend, cfg)
    library = None
    if need_library:
        try:
            library = _library(cfg)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[warn]Library unavailable: {exc}[/warn]")
    manager = SkillManager(cfg)
    if "research" not in cfg.skills.inject_agents:
        cfg = cfg.copy()
        cfg.skills.inject_agents = [*cfg.skills.inject_agents, "research"]
    ctx = build_context(
        cfg,
        backend,
        approve=approve,
        library=library,
        on_step=on_step,
        skill_manager=manager,
    )
    for tool in manager.skill_tools():
        ctx.registry.add(tool)
    agent = get_agent(agent_name)
    return ctx, AgentRuntime(ctx, agent)


def _cli_approve(cfg: Config):
    if cfg.security.approval == "auto":
        return allow_all
    if cfg.security.approval == "read-only":
        return deny_all

    def ask(kind: str, prompt: str) -> bool:
        console.print()
        console.print(approval_panel(kind, prompt))
        try:
            return _ask("Run it?", default=False)
        except EOFError:
            return False

    return ask


def _step_printer(quiet: bool = False):
    def on_step(step: Step) -> None:
        if quiet:
            return
        if step.kind == "tool":
            console.print(tool_step_text(step.name, step.detail, step.ok, step.elapsed_s))
        elif step.kind == "error":
            console.print(f"[fail]✗ {step.detail}[/fail]")

    return on_step


def _run_agent(
    cfg: Config,
    agent_name: str,
    prompt: str,
    *,
    quiet: bool = False,
    approve=None,
    show_sources: bool = False,
    history=None,
    speak: bool = False,
    title: str | None = None,
):
    ctx, runtime = _make_runtime(
        cfg,
        agent_name,
        approve=approve or _cli_approve(cfg),
        on_step=_step_printer(quiet),
    )
    try:
        result = runtime.run(prompt, history=history)
        render_result(console, result, title=title or agent_name.capitalize(), show_sources=show_sources)
        if speak:
            _speak_text(ctx, result.text)
        return result
    finally:
        if ctx.library:
            ctx.library.close()
        ctx.memory.close()
        ctx.state.close()


def _workspace_snapshot(cfg: Config) -> dict[Path, float]:
    files: dict[Path, float] = {}
    root = cfg.workspace
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            try:
                files[path] = path.stat().st_mtime
            except OSError:
                continue
    return files


def _workspace_changes(cfg: Config, before: dict[Path, float], result) -> None:
    if result is None:
        return
    after = _workspace_snapshot(cfg)
    created = sorted(p for p in after if p not in before)
    changed = sorted(p for p in after if p in before and after[p] > before[p] + 0.001)
    if not created and not changed:
        return
    table = Table(box=None, header_style="jarvis", padding=(0, 2), title="Workspace changes")
    table.add_column("", width=2)
    table.add_column("File", overflow="fold")
    row_count = 0
    for path in created:
        table.add_row("[ok]+[/ok]", str(path.relative_to(cfg.workspace)))
        row_count += 1
    for path in changed:
        table.add_row("[warn]~[/warn]", str(path.relative_to(cfg.workspace)))
        row_count += 1
    if row_count:
        console.print(table)


def _speak_text(ctx, text: str) -> None:
    body = text
    if len(body) > 1800:
        body = body[:1800] + "…"
    try:
        path = ctx.voice.speak(body)
        if path:
            spoken = ctx.voice.last_transcript()
            if spoken:
                console.print(Panel(spoken[:900] + ("…" if len(spoken) > 900 else ""), title="Read aloud",
                                    border_style="jarvis.dim"))
            console.print(f"[muted]♪ spoken · saved to {path}[/muted]")
        else:
            console.print("[warn]No TTS engine available — text only.[/warn]")
    except JarvisError as exc:
        console.print(f"[warn]Could not speak: {exc.message}[/warn]")


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Your question or instruction."),
    agent: str = typer.Option("chat", "--agent", "-a", help="chat | research | code | digest"),
    speak: bool = typer.Option(False, "--speak", "-s", help="Read the answer aloud."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve side effects (careful)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Hide tool steps."),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show citations for research."),
    no_color: bool = typer.Option(False, "--no-color"),
) -> None:
    """One-shot question, then exit. Great for scripts and aliases."""
    cfg = _load_or_onboard()
    approve = allow_all if yes else _cli_approve(cfg)
    _run_agent(
        cfg,
        agent,
        prompt,
        quiet=quiet,
        approve=approve,
        show_sources=sources and agent == "research",
        speak=speak,
    )


@app.command()
def research(
    question: str = typer.Argument(..., help="What do you want to know from your documents?"),
    top_k: int = typer.Option(8, "--top-k", help="Passages to retrieve."),
    speak: bool = typer.Option(False, "--speak", "-s"),
    sources: bool = typer.Option(True, "--sources/--no-sources"),
) -> None:
    """Answer from your indexed local documents, with citations."""
    cfg = _load_or_onboard()
    _require_library(cfg)
    _run_agent(cfg, "research", question, approve=deny_all, show_sources=sources, speak=speak)


@app.command()
def code(
    task: str = typer.Argument(..., help="What should the coding agent do?"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve commands and writes."),
    speak: bool = typer.Option(False, "--speak", "-s"),
) -> None:
    """Work with files, run shell commands and Python in your workspace."""
    cfg = _load_or_onboard()
    ensure_workspace(cfg)
    console.print(f"[muted]workspace: {cfg.workspace}[/muted]")
    approve = allow_all if yes else _cli_approve(cfg)
    before = _workspace_snapshot(cfg)
    result = _run_agent(cfg, "code", task, approve=approve, speak=speak)
    _workspace_changes(cfg, before, result)


@app.command()
def digest(
    speak: bool = typer.Option(False, "--speak", "-s", help="Read the briefing aloud."),
    demo: bool = typer.Option(False, "--demo", help="Use sample Gmail/Calendar data."),
    save: bool = typer.Option(True, "--save/--no-save", help="Save the briefing to disk."),
    audio: bool = typer.Option(False, "--audio", help="Always render audio, even without --speak."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Your morning briefing: calendar, email and what needs attention."""
    cfg = _load_or_onboard()
    from .agents.digest import AgendaBundle, DigestAgent

    agent = DigestAgent()
    ctx, runtime = _make_runtime(cfg, "digest", approve=_cli_approve(cfg), on_step=_step_printer(quiet))
    ctx.digest_force_demo = demo  # type: ignore[attr-defined]
    bundle: AgendaBundle = agent.fetch_agenda(ctx, force_demo=demo)
    if bundle.agenda.error and not demo:
        console.print(f"[warn]Partial data: {bundle.agenda.error}[/warn]")
    console.print(
        f"[muted]{bundle.source} · {len(bundle.events)} events · "
        f"{sum(1 for m in bundle.mail if m.important)} flagged emails[/muted]"
    )
    _print_agenda_preview(bundle)
    result = runtime.run("Give me my morning briefing.", history=[])
    console.print()
    console.print(answer_panel(result.text, title="Morning digest"))
    meta = f"{result.model} · {result.elapsed_s:.1f}s"
    if save:
        path = _save_briefing(cfg, result.text, bundle)
        meta += f" · saved {path}"
    console.print(Text(meta, style="jarvis.dim"))
    if speak or audio:
        _speak_text(ctx, result.text)
    if ctx.library:
        ctx.library.close()
    ctx.memory.close()
    ctx.state.close()


def _print_agenda_preview(bundle) -> None:
    if not bundle.events and not bundle.mail:
        return
    table = Table(box=None, show_header=True, header_style="jarvis", padding=(0, 2))
    table.add_column("Time", style="jarvis", no_wrap=True)
    table.add_column("What")
    for event in bundle.events[:6]:
        when = "all day" if event.all_day else (event.start.strftime("%H:%M") if event.start else "—")
        extra = f" · {event.location}" if event.location else ""
        table.add_row(when, event.title + extra)
    if bundle.mail:
        table.add_row("", "")
        for item in bundle.mail[:4]:
            table.add_row("✉", f"{item.sender_name}: {item.subject}")
    console.print(Panel(table, title="Today at a glance", border_style="jarvis.dim"))


def _save_briefing(cfg: Config, text: str, bundle) -> Path:
    cfg.briefings_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now()
    path = cfg.briefings_dir / f"{stamp:%Y-%m-%d_%H%M}-digest.md"
    header = (
        f"# Morning briefing — {stamp:%A %d %B %Y}\n\n"
        f"_{stamp:%H:%M} · source: {bundle.source} · {len(bundle.events)} events · "
        f"{len(bundle.mail)} emails_\n\n"
    )
    path.write_text(header + text.strip() + "\n\n---\n\n## Raw agenda\n\n```\n" + bundle.text + "\n```\n",
                    encoding="utf-8")
    return path


@app.command()
def chat(
    agent: str = typer.Option("chat", "--agent", "-a", help="chat | research | code | digest"),
    speak: bool = typer.Option(False, "--speak", "-s", help="Read answers aloud."),
    session: str | None = typer.Option(None, "--session", help="Resume a session id."),
    new: bool = typer.Option(False, "--new", help="Start a fresh session."),
    no_color: bool = typer.Option(False, "--no-color"),
) -> None:
    """Interactive session. Type /help for in-chat commands."""
    out = _console(no_color)
    cfg = _load_or_onboard()
    agent_obj = get_agent(agent)
    state = StateStore(cfg.state_db)
    if session:
        session_id = session
    elif new or not state.latest_session(agent):
        session_id = state.create_session(agent)
    else:
        session_id = state.latest_session(agent) or state.create_session(agent)

    ctx, runtime = _make_runtime(cfg, agent, approve=_cli_approve(cfg), on_step=_step_printer(False))
    history = state.history(session_id, limit=cfg.agent.history_messages) if session or not new else []

    banner(
        out,
        cfg.assistant_name,
        f"{agent_obj.title} · session {session_id}",
        extra=f"{cfg.backend}:{runtime.model} · /help for commands",
    )
    _setup_readline()
    speak_on = speak
    while True:
        try:
            raw = Prompt.ask("[jarvis]›[/jarvis]", console=out)
        except (EOFError, KeyboardInterrupt):
            out.print("\n[muted]Bye.[/muted]")
            break
        text = raw.strip()
        if not text:
            continue
        if text.startswith("/"):
            action = _chat_command(out, cfg, ctx, text, speak_on)
            if action == "quit":
                break
            if action == "clear":
                history = []
                session_id = state.create_session(agent)
                out.print(f"[muted]New session {session_id}[/muted]")
            if action == "speak":
                speak_on = not speak_on
                out.print(f"[muted]speak {'on' if speak_on else 'off'}[/muted]")
            if action == "sources":
                if ctx.last_hits:
                    out.print(sources_panel(ctx.last_hits))
                else:
                    out.print("[muted]No sources in this session yet.[/muted]")
            continue

        state.add_message(session_id, "user", text)
        result = runtime.run(text, history=history)
        out.print()
        out.print(answer_panel(result.text))
        if agent == "research" and result.hits:
            used = _cited_sources(result.text, result.hits)
            if used:
                out.print(sources_panel(used))
        if result.tool_calls:
            out.print(Text(f"  {result.tool_calls} tool calls · {result.elapsed_s:.1f}s", style="jarvis.dim"))
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": result.text})
        history = history[-cfg.agent.history_messages :]
        state.add_message(session_id, "assistant", result.text)
        state.touch_session(session_id, title=text[:80])
        if speak_on:
            _speak_text(ctx, result.text)

    if ctx.library:
        ctx.library.close()
    ctx.memory.close()
    ctx.state.close()


def _chat_command(out, cfg: Config, ctx, text: str, speak_on: bool) -> str:
    command, _, rest = text.partition(" ")
    command = command.lower()
    if command in {"/quit", "/exit", "/q"}:
        return "quit"
    if command in {"/help", "/h", "/?"}:
        out.print(
            Panel(
                Text.from_markup(
                    "[bold]/help[/bold]      this list\n"
                    "[bold]/clear[/bold]     start a new session\n"
                    "[bold]/speak[/bold]     toggle read-aloud       [bold]/sources[/bold]  show retrieved passages\n"
                    "[bold]/memory[/bold]    what Jarvis remembers   [bold]/remember[/bold] save a fact\n"
                    "[bold]/model[/bold]     show the active model   [bold]/doctor[/bold]   quick checks\n"
                    "[bold]/note[/bold]      save a note            [bold]/quit[/bold]     exit"
                ),
                title="In-chat commands",
                border_style="jarvis.dim",
            )
        )
        return "none"
    if command == "/clear":
        return "clear"
    if command == "/speak":
        return "speak"
    if command == "/sources":
        return "sources"
    if command == "/memory":
        rows = ctx.memory.all(limit=20)
        if not rows:
            out.print("[muted]Nothing remembered yet. Say “remember …” or use /remember.[/muted]")
        else:
            out.print(kv_table([(f"#{r['id']}", r["text"]) for r in rows], title="Memory"))
        return "none"
    if command == "/remember":
        if rest.strip():
            ctx.memory.remember(rest.strip())
            out.print(f"[ok]Saved:[/ok] {rest.strip()}")
        else:
            out.print("[muted]Usage: /remember prefers short bullet summaries[/muted]")
        return "none"
    if command == "/note":
        if rest.strip():
            from .tools.knowledge import take_note

            res = take_note(ctx, rest.strip())
            out.print(res.content)
        else:
            out.print("[muted]Usage: /note idea for the weekend project[/muted]")
        return "none"
    if command == "/model":
        out.print(f"[muted]backend[/muted] {cfg.backend}  [muted]general[/muted] {cfg.models.general}  "
                  f"[muted]coding[/muted] {cfg.models.coding}")
        return "none"
    if command == "/doctor":
        checks = run_doctor(cfg)
        out.print(checks_table(checks))
        return "none"
    out.print(f"[warn]Unknown command {command}[/warn] — try /help")
    return "none"


def _setup_readline() -> None:
    try:
        import readline  # noqa: F401

        history = Path.home() / ".jarvis_history"
        try:
            readline.read_history_file(str(history))
        except OSError:
            pass
        import atexit

        atexit.register(lambda: _save_history(readline, history))
    except ImportError:
        pass


def _save_history(readline, path: Path) -> None:
    try:
        readline.write_history_file(str(path))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

def _library(cfg: Config) -> Library:

    embedder = None
    if cfg.library.use_embeddings and cfg.backend in {"ollama", "openai-compatible"}:
        backend = make_backend(cfg)

        class _Embedder:
            model = cfg.models.embed

            def embed(self, texts):
                return backend.embed(self.model, texts)

        embedder = _Embedder()
    return Library(cfg, embedder=embedder)


def _require_library(cfg: Config) -> None:
    with _library(cfg) as lib:
        if not lib.has_documents():
            console.print(
                Panel(
                    Text.from_markup(
                        "The research agent answers from **your** documents, so it needs an index.\n\n"
                        f"  [bold]jarvis library index ~/Documents[/bold]\n\n"
                        f"Currently watching: {', '.join(cfg.paths.library)}"
                    ),
                    title="Index your documents first",
                    border_style="warn",
                )
            )
            raise typer.Exit(code=1)


def _index_library(cfg: Config, roots, rebuild: bool = False, quiet: bool = False, allow_embeddings: bool = True):
    lib = _library(cfg)
    if not allow_embeddings:
        lib.embedder = None
    try:
        total = {"n": 0}

        def progress(n, path):
            total["n"] = n
            if not quiet and console.is_terminal:
                console.print(f"[muted]  {n} · {path.name}[/muted]", end="\r")

        stats = lib.index(roots, progress=None if quiet else progress, rebuild=rebuild)
        if not quiet:
            console.print(f"[muted]  scanned {stats.scanned} files[/muted]")
        return stats
    finally:
        lib.close()


@library_app.command("index")
def library_index(
    paths: list[str] = typer.Argument(None, help="Folders or files to index (defaults to configured library)."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Wipe the index and start over."),
    add: bool = typer.Option(False, "--add", help="Remember these folders for future scans."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Scan folders and update the search index (incremental)."""
    cfg = _load_config()
    if add and paths:
        cfg.paths.library = sorted({*cfg.paths.library, *paths})
        cfg.save()
        console.print(f"[muted]Library roots now: {', '.join(cfg.paths.library)}[/muted]")
    roots = [expand_path(p) for p in paths] if paths else cfg.library_roots()
    existing = [r for r in roots if r.exists()]
    if not existing:
        console.print("[warn]None of these folders exist:[/warn] " + ", ".join(str(r) for r in roots))
        console.print("[muted]Pass a path: jarvis library index ~/Documents[/muted]")
        raise typer.Exit(code=1)
    console.print(f"[muted]Indexing {', '.join(str(r) for r in existing)}[/muted]")
    stats = _index_library(cfg, existing, rebuild=rebuild, quiet=quiet)
    console.print()
    console.print(
        kv_table(
            [
                ("Scanned", str(stats.scanned)),
                ("Added", f"[ok]{stats.added}[/ok]" if stats.added else "0"),
                ("Updated", str(stats.updated)),
                ("Unchanged", str(stats.unchanged)),
                ("Removed", str(stats.removed)),
                ("Chunks", str(stats.chunks)),
                ("Embedded", str(stats.embedded)),
                ("Failed", f"[warn]{stats.failed}[/warn]" if stats.failed else "0"),
                ("Time", f"{stats.elapsed_s:.1f}s"),
            ],
            title="Index updated",
        )
    )
    if stats.errors and not quiet:
        for err in stats.errors[:8]:
            console.print(f"[warn]  ! {err}[/warn]")
        if len(stats.errors) > 8:
            console.print(f"[muted]  … and {len(stats.errors) - 8} more[/muted]")
    if stats.failed and stats.added == 0:
        console.print(
            "[muted]Tips: PDFs need pypdf (`pip install 'jarvis-local[docs]'`); scanned PDFs "
            "need OCR first; encrypted files stay unread.[/muted]"
        )
    if stats.embed_error:
        console.print(
            "[warn]Embeddings skipped: the local embedding model is unavailable.[/warn]\n"
            f"[muted]  {stats.embed_error[:160]}\n"
            f"  Keyword search still works. After starting Ollama: ollama pull {cfg.models.embed} "
            "&& jarvis library index[/muted]"
        )


@library_app.command("search")
def library_search(
    query: str = typer.Argument(..., help="Search terms."),
    top_k: int = typer.Option(6, "--top-k", "-k"),
    mode: str = typer.Option("hybrid", "--mode", help="hybrid | keyword | semantic"),
    full: bool = typer.Option(False, "--full", help="Show full passage text."),
) -> None:
    """Search the local index without invoking a model."""
    cfg = _load_config()
    with _library(cfg) as lib:
        hits = lib.search(query, top_k=top_k, mode=mode)
    if not hits:
        console.print("[muted]No matches. Try broader terms or index more folders.[/muted]")
        raise typer.Exit()
    from .documents import citation_label

    for i, hit in enumerate(hits, 1):
        console.print()
        console.print(f"[jarvis][{i}][/jarvis] [bold]{citation_label(hit)}[/bold] [muted]{hit.path}[/muted]")
        body = hit.text if full else (hit.text[:420] + ("…" if len(hit.text) > 420 else ""))
        console.print(Panel(body, border_style="jarvis.dim", box=box.SQUARE, padding=(0, 1)))


@library_app.command("stats")
def library_stats() -> None:
    """How big is the index?"""
    cfg = _load_config()
    with _library(cfg) as lib:
        stats = lib.stats()
    if not stats["documents"]:
        console.print("[muted]The library is empty. Run `jarvis library index ~/Documents`.[/muted]")
        return
    console.print(
        kv_table(
            [
                ("Documents", str(stats["documents"])),
                ("Chunks", str(stats["chunks"])),
                ("Embeddings", f"{stats['vectors']} ({stats['embed_model'] or 'none'})"),
                ("Text", f"{stats['chars'] / 1000:.1f}k characters" if stats["chars"] >= 1000 else f"{stats['chars']} characters"),
                ("Last indexed", stats["last_indexed"] or "—"),
                ("Database", f"{stats['db_size'] / 1024 / 1024:.1f} MB at {cfg.library_db}"),
            ]
        )
    )


@library_app.command("list")
def library_list(limit: int = typer.Option(30, "--limit")) -> None:
    """List indexed documents."""
    cfg = _load_config()
    with _library(cfg) as lib:
        rows = lib.conn.execute(
            "SELECT title, path, pages, chars FROM documents ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    if not rows:
        console.print("[muted]Nothing indexed yet.[/muted]")
        return
    table = Table(box=None, header_style="jarvis", padding=(0, 2))
    table.add_column("Document")
    table.add_column("Pages", justify="right")
    table.add_column("Size", justify="right")
    for row in rows:
        chars = row["chars"] or 0
        size = f"{chars / 1000:.1f}k chars" if chars >= 1000 else f"{chars} chars"
        table.add_row(row["path"], str(row["pages"] or "—"), size)
    console.print(table)


@library_app.command("reset")
def library_reset(yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Delete the local index (your files are untouched)."""
    cfg = _load_config()
    if not yes and not _ask("Delete the search index? Original files stay on disk.", default=False):
        raise typer.Exit()
    if cfg.library_db.exists():
        cfg.library_db.unlink()
    console.print("[ok]Index deleted.[/ok] Rebuild with `jarvis library index`.")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@models_app.command("list")
def models_list(limit: int = typer.Option(20, "--limit")) -> None:
    """Show installed Ollama models and the catalogue."""
    cfg = _load_config()
    names = installed_models(cfg)
    if names:
        table = Table(title="Installed", box=None, header_style="jarvis", padding=(0, 2))
        table.add_column("Model")
        table.add_column("Role", style="muted")
        roles = {}
        for role in ("general", "coding", "fast", "embed"):
            roles[getattr(cfg.models, role)] = role
        for name in names:
            table.add_row(name, roles.get(name, ""))
        console.print(table)
    else:
        console.print("[muted]No models installed (or Ollama is not running).[/muted]")
    table = Table(title=Text("Catalogue (models that fit your budget are green)"), box=None, padding=(0, 2))
    table.add_column("Tag", style="bold")
    table.add_column("Params")
    table.add_column("Size", justify="right")
    table.add_column("Roles")
    table.add_column("Notes", style="muted")
    budget = cfg.hardware.budget_gb or detect_hardware().budget_gb()
    for card in CATALOG[:limit]:
        style = "ok" if card.size_gb <= budget else "muted"
        table.add_row(
            Text(card.tag, style=style),
            card.params,
            human_gb(card.size_gb),
            ", ".join(card.roles),
            card.note,
        )
    console.print(table)
    console.print(f"[muted]Your budget: {human_gb(budget)} (override with --budget on init)[/muted]")


@models_app.command("plan")
def models_plan(budget: float | None = typer.Option(None, "--budget")) -> None:
    """Explain which models Jarvis would choose for this machine."""
    profile = detect_hardware()
    chosen = build_plan(Config.load(), budget_gb=budget)
    console.print()
    banner(console, "Model plan", profile.compute_label)
    console.print(kv_table([
        ("Usable memory budget", human_gb(chosen.plan.budget_gb)),
        ("General", chosen.plan.general),
        ("Coding", chosen.plan.coding),
        ("Fast", chosen.plan.fast),
        ("Embeddings", chosen.plan.embed),
        ("Context", f"{chosen.plan.num_ctx} tokens"),
        ("Download", human_gb(chosen.plan.download_gb())),
    ]))
    console.print(f"\n[muted]{chosen.plan.rationale}[/muted]")
    for note in chosen.plan.notes:
        console.print(f"[warn]! {note}[/warn]")


@models_app.command("pull")
def models_pull(
    tags: list[str] = typer.Argument(None, help="Specific tags (default: your configured set)."),
) -> None:
    """Download the models you need."""
    cfg = _load_config()
    if not ollama_present():
        console.print("[fail]Ollama is not installed.[/fail] See https://ollama.com/download")
        raise typer.Exit(code=1)
    if not ollama_running(cfg):
        console.print("[muted]Starting Ollama…[/muted]")
        start_ollama(cfg)
    targets = tags or [cfg.models.general, cfg.models.coding, cfg.models.fast, cfg.models.embed]
    for tag in dict.fromkeys(targets):
        if model_installed(cfg, tag):
            console.print(f"[ok]✓[/ok] {tag} already installed")
            continue
        console.print(f"[jarvis]↓[/jarvis] Pulling {tag} …")
        ok = pull_model(cfg, tag, on_line=lambda line: None)
        console.print(f"[{'ok' if ok else 'fail'}]{'✓' if ok else '✗'}[/] {tag}")


@models_app.command("switch")
def models_switch(
    role: str = typer.Argument(..., help="general | coding | fast | embed"),
    tag: str | None = typer.Argument(None, help="Model tag to use (omit for auto again)."),
) -> None:
    """Pin a model for one role, or return to the recommendation."""
    cfg = _load_config()
    try:
        old, new = switch_model_plan(cfg, role, tag)
    except ValueError as exc:
        console.print(f"[fail]{exc}[/fail]")
        raise typer.Exit(code=1) from None
    cfg.save()
    console.print(f"[ok]{role}[/ok]: {old} → [bold]{new}[/bold]")
    if tag and cfg.backend == "ollama" and not model_installed(cfg, new):
        console.print(f"[muted]Not installed yet: `ollama pull {new}`[/muted]")


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def _skill_confirm(skill, report) -> bool:
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]{skill.name}[/bold] {skill.version}\n{skill.description}\n\n"
                f"Source: {skill.source or 'local folder'}"
            ),
            title="Install skill?",
            border_style="jarvis",
        )
    )
    for line in report.lines():
        style = {"HIGH": "fail", "MEDIUM": "warn"}.get("HIGH" if "HIGH" in line else ("MEDIUM" if "MEDIUM" in line else ""), "muted")
        console.print(f"  [{style}]{line}[/{style}]")
    if skill.run:
        console.print(f"  [muted]Runs script on request: {skill.run}[/muted]")
    return _ask("Install it?", default=False)


@skills_app.command("review")
def skills_review() -> None:
    """Look for repeated work in the audit log and suggest skills."""
    cfg = _load_config()
    manager = SkillManager(cfg)
    suggestions = manager.suggest_lessons()
    usage = manager.usage()
    used = [(name, info) for name, info in usage.items() if info.get("uses")]
    if used:
        table = Table(box=None, header_style="jarvis", padding=(0, 2), title="Skill usage")
        table.add_column("Skill", style="bold")
        table.add_column("Uses", justify="right")
        table.add_column("Last used")
        for name, info in sorted(used, key=lambda item: item[1].get("uses", 0), reverse=True):
            table.add_row(name, str(info.get("uses", 0)), str(info.get("last_used", ""))[:16])
        console.print(table)
    if not suggestions:
        console.print("[muted]No clear patterns yet — keep using Jarvis; this looks at the audit log "
                      "and needs a few days of history.[/muted]")
        return
    console.print("[jarvis]Ideas for new skills[/jarvis]")
    for line in suggestions:
        console.print(f"  · {line}")
    console.print("[muted]Create one with `jarvis skills new <name>`; Jarvis will load it next time.[/muted]")


@skills_app.command("list")
def skills_list(all: bool = typer.Option(False, "--all", help="Include disabled skills.")) -> None:
    """Show installed skills."""
    cfg = _load_config()
    manager = SkillManager(cfg)
    skills = manager.list(include_disabled=all)
    if not skills:
        console.print("[muted]No skills yet. Try `jarvis skills new my-skill`.[/muted]")
        return
    table = Table(box=None, header_style="jarvis", padding=(0, 2))
    table.add_column("Skill", style="bold")
    table.add_column("State")
    table.add_column("Risk")
    table.add_column("Description", overflow="fold")
    usage = manager.usage()
    for skill in skills:
        state = "[ok]on[/ok]" if skill.enabled else "[muted]off[/muted]"
        risk_style = {"low": "ok", "medium": "warn", "high": "fail"}[skill.security.risk]
        uses = usage.get(skill.name, {}).get("uses", 0)
        desc = skill.description + (f"  [muted]({uses}×)[/muted]" if uses else "")
        table.add_row(skill.name, state, f"[{risk_style}]{skill.security.risk}[/{risk_style}]", desc)
    console.print(table)
    console.print(f"[muted]{cfg.skills_dir}[/muted]")


@skills_app.command("install")
def skills_install(
    source: str = typer.Argument(..., help="GitHub URL, archive URL, or local folder. Use repo#skill for collections."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    force: bool = typer.Option(False, "--force", help="Replace an existing skill."),
    sha256: str = typer.Option("", "--sha256", help="Verify the download against this checksum."),
    list_only: bool = typer.Option(False, "--list", help="List the skills in the repo without installing."),
) -> None:
    """Install a skill from a repository or folder (scanned, then confirmed)."""
    cfg = _load_config()
    manager = SkillManager(cfg)
    if list_only:
        console.print(f"[muted]Inspecting {source} …[/muted]")
        try:
            entries = manager.list_remote(source)
        except JarvisError as exc:
            console.print(f"[fail]✗ {exc.message}[/fail]")
            if exc.hint:
                console.print(f"[muted]{exc.hint}[/muted]")
            raise typer.Exit(code=1) from None
        table = Table(box=None, header_style="jarvis", padding=(0, 2))
        table.add_column("Skill", style="bold")
        table.add_column("Description", overflow="fold")
        for entry in entries[:60]:
            table.add_row(entry["name"], entry["description"] or "—")
        console.print(table)
        if len(entries) > 60:
            console.print(f"[muted]… {len(entries) - 60} more[/muted]")
        console.print(
            f"[muted]Install one with: jarvis skills install {source}#{entries[0]['name']}[/muted]"
        )
        return
    console.print(f"[muted]Fetching {source} …[/muted]")
    try:
        skill = manager.install(
            source,
            confirm=None if yes else _skill_confirm,
            force=force,
            expected_sha256=sha256,
        )
    except JarvisError as exc:
        console.print(f"[fail]✗ {exc.message}[/fail]")
        if exc.hint:
            console.print(f"[muted]{exc.hint}[/muted]")
        raise typer.Exit(code=1) from None
    console.print(f"[ok]✓ Installed[/ok] {skill.name} → {cfg.skills_dir / skill.name}")
    console.print(
        "[muted]It activates when a message matches its triggers. "
        f"Inspect: `jarvis skills show {skill.name}`[/muted]"
    )


@skills_app.command("remove")
def skills_remove(name: str, yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Uninstall a skill."""
    cfg = _load_config()
    if not yes and not _ask(f"Delete skill '{name}'?", default=False):
        raise typer.Exit()
    try:
        SkillManager(cfg).remove(name)
    except JarvisError as exc:
        console.print(f"[fail]{exc.message}[/fail]")
        raise typer.Exit(code=1) from None
    console.print(f"[ok]Removed {name}.[/ok]")


@skills_app.command("enable")
def skills_enable(name: str) -> None:
    """Turn a skill on."""
    _toggle_skill(name, True)


@skills_app.command("disable")
def skills_disable(name: str) -> None:
    """Turn a skill off without deleting it."""
    _toggle_skill(name, False)


def _toggle_skill(name: str, enabled: bool) -> None:
    cfg = _load_config()
    try:
        SkillManager(cfg).set_enabled(name, enabled)
    except JarvisError as exc:
        console.print(f"[fail]{exc.message}[/fail]")
        raise typer.Exit(code=1) from None
    console.print(f"[ok]{name} {'enabled' if enabled else 'disabled'}.[/ok]")


@skills_app.command("show")
def skills_show(name: str) -> None:
    """Read a skill's instructions and security report."""
    cfg = _load_config()
    skill = SkillManager(cfg).get(name)
    if not skill:
        console.print(f"[fail]Skill '{name}' not found.[/fail]")
        raise typer.Exit(code=1)
    console.print(Panel(Text(skill.description), title=f"{skill.name} {skill.version}", border_style="jarvis"))
    if skill.triggers:
        console.print(f"[muted]Triggers: {', '.join(skill.triggers)}[/muted]")
    for line in skill.security.lines():
        console.print(f"  [muted]{line}[/muted]")
    console.print()
    console.print(Panel(Markdown(skill.body), title="Instructions", border_style="jarvis.dim"))


@skills_app.command("scan")
def skills_scan(name: str | None = typer.Argument(None, help="One skill (default: all).")) -> None:
    """Re-run the security scan on installed skills."""
    cfg = _load_config()
    manager = SkillManager(cfg)
    skills = [manager.get(name)] if name else manager.list()
    if name and not skills[0]:
        console.print(f"[fail]Skill '{name}' not found.[/fail]")
        raise typer.Exit(code=1)
    for skill in skills:
        if not skill:
            continue
        style = {"low": "ok", "medium": "warn", "high": "fail"}[skill.security.risk]
        console.print(f"[{style}]{skill.name}: {skill.security.risk} risk[/{style}]")
        for line in skill.security.lines()[1:]:
            console.print(f"  [muted]{line}[/muted]")


@skills_app.command("new")
def skills_new(
    name: str,
    description: str = typer.Option("", "--description", "-d"),
) -> None:
    """Create a skill template you can edit."""
    cfg = _load_config()
    try:
        folder = SkillManager(cfg).scaffold(name, description)
    except JarvisError as exc:
        console.print(f"[fail]{exc.message}[/fail]")
        raise typer.Exit(code=1) from None
    console.print(f"[ok]Created[/ok] {folder}")
    console.print("[muted]Edit SKILL.md, then try `jarvis skills show "
                  f"{name}` and ask something that matches its triggers.[/muted]")


@skills_app.command("export")
def skills_export(
    name: str,
    output: str | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Zip a skill so you can share or re-install it."""
    cfg = _load_config()
    skill = SkillManager(cfg).get(name)
    if not skill or not skill.path:
        console.print(f"[fail]Skill '{name}' not found.[/fail]")
        raise typer.Exit(code=1)
    dest = Path(output) if output else Path.cwd() / f"{name}.zip"
    path = export_skill(skill, dest)
    console.print(f"[ok]Exported[/ok] {path}")


@skills_app.command("run")
def skills_run(
    name: str,
    request: str = typer.Argument("", help="Request passed to the skill script."),
) -> None:
    """Run a skill's script directly (still policy-checked)."""
    cfg = _load_config()
    manager = SkillManager(cfg)
    skill = manager.get(name)
    if not skill:
        console.print(f"[fail]Skill '{name}' not found.[/fail]")
        raise typer.Exit(code=1)
    if not skill.run:
        console.print(f"[warn]'{name}' is prompt-only (no run: script in SKILL.md).[/warn]")
        raise typer.Exit()
    ctx = build_context(cfg, make_backend(cfg), approve=_cli_approve(cfg))
    from .skills import run_skill_script

    result = run_skill_script(ctx, skill, request)
    console.print(result.content)


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

@google_app.command("status")
def google_status_cmd() -> None:
    """Is Gmail/Calendar connected?"""
    cfg = _load_config()
    status = google_status(cfg)
    rows = [
        ("Libraries", "[ok]installed[/ok]" if status["libraries"] else "[fail]missing[/fail] (pip install 'jarvis-local[google]')"),
        ("Client secret", f"[ok]{status['client_secret_path']}[/ok]" if status["client_secret"] else "[warn]not found[/warn]"),
        ("Authorised token", f"[ok]{status['token_path']}[/ok]" if status["token"] else "[warn]not yet[/warn]"),
        ("Access", "read-only Gmail + Calendar" if status["ready"] else "—"),
    ]
    console.print(kv_table(rows, title="Google"))
    if status["client_secret"] and client_secret_is_placeholder(cfg):
        console.print("[warn]The client secret file is still the placeholder Jarvis created.[/warn]")
    console.print(f"\n[muted]{GoogleSource(cfg).describe()}[/muted]")


@google_app.command("setup")
def google_setup_cmd(
    install_deps: bool = typer.Option(False, "--install-deps", help="Install the Google client libraries first."),
) -> None:
    """Walk through creating your OAuth client (three minutes)."""
    cfg = _load_config()
    status = google_status(cfg)
    if install_deps or (not status["libraries"] and _ask("Install the Google client libraries now?", default=True)):
        with spinner(console, "Installing google-api-python-client and google-auth-oauthlib…"):
            ok, message = install_google_libs(cfg)
        console.print(f"[{'ok' if ok else 'warn'}]{message}[/]")
    path = write_sample_client_secret(cfg)
    console.print()
    console.print(Panel(Text(SETUP_HELP.format(secret_path=cfg.google_client_secret)), title="Google setup", border_style="jarvis"))
    if path:
        console.print(f"[muted]A placeholder file is already at {path} — replace its client_id/client_secret.[/muted]")
    if client_secret_is_placeholder(cfg):
        console.print("[muted]Once you have the real JSON, run `jarvis google auth`.[/muted]")


@google_app.command("auth")
def google_auth_cmd() -> None:
    """Authorise read-only Gmail + Calendar access (browser or paste a code)."""
    cfg = _load_config()
    status = google_status(cfg)
    if not status["libraries"]:
        console.print("[fail]Google libraries missing.[/fail] pip install 'jarvis-local[google]'")
        raise typer.Exit(code=1)
    if not status["client_secret"]:
        console.print("[fail]No client secret file.[/fail] Run `jarvis google setup` first.")
        raise typer.Exit(code=1)
    if client_secret_is_placeholder(cfg):
        console.print("[fail]The client secret file is still the placeholder.[/fail]")
        console.print(f"[muted]Edit {cfg.google_client_secret} with your real client_id and client_secret.[/muted]")
        raise typer.Exit(code=1)
    console.print("[muted]A browser window will open. If this is a remote terminal, use the printed URL.[/muted]")
    try:
        GoogleSource(cfg).credentials(interactive=True)
    except JarvisError as exc:
        console.print(f"[fail]✗ {exc.message}[/fail]")
        if exc.hint:
            console.print(Panel(Text(exc.hint), border_style="jarvis.dim"))
        raise typer.Exit(code=1) from None
    console.print("[ok]✓ Connected.[/ok] Try `jarvis digest --speak`.")


@google_app.command("test")
def google_test_cmd() -> None:
    """Fetch today's calendar and inbox to verify access."""
    cfg = _load_config()
    source = GoogleSource(cfg)
    if not source.is_configured():
        console.print("[warn]Google is not configured.[/warn] Run `jarvis google setup`.")
        raise typer.Exit(code=1)
    with spinner(console, "Fetching from Google (read-only)…"):
        agenda = source.fetch(horizon_hours=cfg.google.hours_back, max_mails=cfg.google.max_mails)
    if agenda.error:
        console.print(f"[warn]{agenda.error}[/warn]")
        raise typer.Exit(code=1)
    console.print(f"[ok]{len(agenda.events)} events · {len(agenda.mail)} emails[/ok]")
    from .agents.digest import render_agenda_text

    console.print(Panel(render_agenda_text(agenda), title="Raw data", border_style="jarvis.dim"))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@memory_app.command("list")
def memory_list(limit: int = typer.Option(30, "--limit")) -> None:
    """Show stored memories."""
    cfg = _load_config()
    store = MemoryStore(cfg.state_db)
    rows = store.all(limit)
    store.close()
    if not rows:
        console.print("[muted]Nothing remembered yet. Try `jarvis memory add \"...\"`[/muted]")
        return
    console.print(kv_table([(f"#{r['id']} · {r['kind']}", r["text"]) for r in rows], title="Memory"))


@memory_app.command("add")
def memory_add(text: str = typer.Argument(...), kind: str = typer.Option("fact", "--kind")) -> None:
    """Save a durable fact."""
    cfg = _load_config()
    store = MemoryStore(cfg.state_db)
    result = store.remember(text, kind)
    store.close()
    console.print(f"[ok]{result.content}[/ok]" if result.ok else f"[warn]{result.content}[/warn]")


@memory_app.command("forget")
def memory_forget(memory_id: int) -> None:
    """Delete one memory."""
    cfg = _load_config()
    store = MemoryStore(cfg.state_db)
    result = store.forget(memory_id)
    store.close()
    console.print(result.content)


@memory_app.command("search")
def memory_search(query: str) -> None:
    """Search memories."""
    cfg = _load_config()
    store = MemoryStore(cfg.state_db)
    console.print(store.search(query))
    store.close()


@memory_app.command("clear")
def memory_clear(yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Delete all memories."""
    cfg = _load_config()
    if not yes and not _ask("Forget everything Jarvis remembers?", default=False):
        raise typer.Exit()
    store = MemoryStore(cfg.state_db)
    store.conn.execute("DELETE FROM memories")
    store.conn.commit()
    store.close()
    console.print("[ok]Memory cleared.[/ok]")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@config_app.command("show")
def config_show(no_color: bool = typer.Option(False, "--no-color")) -> None:
    """Print the active configuration."""
    cfg = _load_config()
    from .config import _serializable, dump_toml

    rendered = "# " + str(cfg.config_file) + "\n\n" + dump_toml(_serializable(cfg))
    console.print(Syntax(rendered, "toml", theme="monokai", word_wrap=True))


@config_app.command("path")
def config_path() -> None:
    """Where config and data live."""
    cfg = _load_config()
    console.print(kv_table([
        ("Config file", str(cfg.config_file)),
        ("Config dir", str(cfg.config_root)),
        ("Data dir", str(cfg.data_root)),
        ("Workspace", str(cfg.workspace)),
        ("Library db", str(cfg.library_db)),
        ("Skills", str(cfg.skills_dir)),
        ("Briefings", str(cfg.briefings_dir)),
        ("Logs / audit", str(cfg.log_dir)),
    ]))


@config_app.command("set")
def config_set(key: str = typer.Argument(..., help="Dotted key, e.g. models.general"), value: str = typer.Argument(...)) -> None:
    """Change one value and save."""
    cfg = _load_config()
    try:
        old, new = cfg.set_value(key, value)
    except JarvisError as exc:
        console.print(f"[fail]{exc.message}[/fail]")
        if exc.hint:
            console.print(f"[muted]{exc.hint}[/muted]")
        raise typer.Exit(code=1) from None
    cfg.save()
    console.print(f"[ok]{key}[/ok]: {old!r} → {new!r}")


@config_app.command("edit")
def config_edit() -> None:
    """Open config.toml in $EDITOR."""
    cfg = _load_config()
    if not cfg.config_file.exists():
        cfg.save()
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or ("open" if sys.platform == "darwin" else "nano")
    try:
        subprocess.call([editor, str(cfg.config_file)])
    except OSError as exc:
        console.print(f"[fail]Could not launch {editor}: {exc}[/fail]")
        console.print(f"[muted]Open it yourself: {cfg.config_file}[/muted]")


@config_app.command("reset")
def config_reset(yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Delete config.toml (data and models are kept)."""
    cfg = _load_config()
    target = cfg.config_file
    if not target.exists():
        console.print("[muted]No config file to reset.[/muted]")
        return
    if not yes and not _ask(f"Delete {target}? Data stays; run `jarvis init` afterwards.", default=False):
        raise typer.Exit()
    target.unlink()
    console.print("[ok]Config removed.[/ok]")


# ---------------------------------------------------------------------------
# Schedule, demo, notes, sessions
# ---------------------------------------------------------------------------

@schedule_app.command("install")
def schedule_install(
    time: str = typer.Option("07:30", "--time", "-t", help="HH:MM local time."),
    speak: bool = typer.Option(True, "--speak/--silent"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Add a cron entry that prepares your digest each morning."""
    cfg = _load_config()
    try:
        hour, minute = [int(x) for x in time.split(":")]
        assert 0 <= hour < 24 and 0 <= minute < 60
    except (ValueError, AssertionError):
        console.print("[fail]Time must be HH:MM, e.g. 07:30[/fail]")
        raise typer.Exit(code=1) from None
    if not shutdown_allowed("crontab"):
        console.print("[warn]`crontab` is not available on this machine.[/warn]")
        console.print("[muted]Use the OS scheduler instead — `jarvis schedule show` prints the command.[/muted]")
        raise typer.Exit(code=1)
    exe = shutil.which("jarvis") or "jarvis"
    command = f"{exe} digest --quiet{' --speak' if speak else ''} >> {cfg.log_dir}/digest.log 2>&1"
    line = f"{minute} {hour} * * * {command}"
    console.print(Panel(Text(line), title="Cron entry", border_style="jarvis"))
    if not yes and not _ask("Add this to your crontab?", default=True):
        raise typer.Exit()
    existing = _read_crontab()
    if line in existing:
        console.print("[muted]Already scheduled.[/muted]")
        return
    new_crontab = existing.rstrip("\n") + ("\n" if existing.strip() else "") + line + "\n"
    proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True, capture_output=True)
    if proc.returncode != 0:
        console.print(f"[fail]crontab failed: {proc.stderr.strip()}[/fail]")
        console.print("[muted]Copy the entry above into `crontab -e` manually.[/muted]")
        raise typer.Exit(code=1)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[ok]Scheduled daily at {time}.[/ok] Output → {cfg.log_dir}/digest.log")


def _read_crontab() -> str:
    if not shutdown_allowed("crontab"):
        return ""
    try:
        return subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


@schedule_app.command("show")
def schedule_show() -> None:
    """Show the current digest cron entry, if any."""
    lines = [line for line in _read_crontab().splitlines() if "jarvis digest" in line]
    if not lines:
        console.print("[muted]No digest scheduled. `jarvis schedule install`[/muted]")
        return
    for line in lines:
        console.print(line)


@schedule_app.command("remove")
def schedule_remove() -> None:
    """Remove the digest cron entry."""
    existing = _read_crontab()
    if not existing.strip():
        console.print("[muted]No crontab found.[/muted]")
        return
    kept = [line for line in existing.splitlines() if "jarvis digest" not in line]
    subprocess.run(["crontab", "-"], input="\n".join(kept) + "\n", text=True, capture_output=True, timeout=10)
    console.print("[ok]Digest schedule removed.[/ok]")


@app.command()
def note(text: str = typer.Argument(...), title: str = typer.Option("", "--title", "-t")) -> None:
    """Save a quick note to the workspace."""
    cfg = _load_config()
    from .tools.knowledge import take_note

    ctx = build_context(cfg, make_backend(cfg), approve=deny_all)
    result = take_note(ctx, text, title)
    console.print(result.content)


@app.command()
def memory_search_alias(query: str = typer.Argument(...)) -> None:
    """Shortcut for `jarvis memory search`."""
    memory_search(query)


@models_app.command("current")
def models_current() -> None:
    """Which model is used for each job."""
    cfg = _load_config()
    console.print(kv_table([
        ("Backend", cfg.backend),
        ("General", cfg.models.general),
        ("Coding", cfg.models.coding),
        ("Fast", cfg.models.fast),
        ("Embeddings", cfg.models.embed),
        ("Context", str(cfg.models.num_ctx)),
    ], title="Models"))


@app.command()
def demo(
    speak: bool = typer.Option(False, "--speak", "-s"),
) -> None:
    """Guided tour that works with no model, no account and no network."""
    cfg = _load_config()
    if not (cfg.source_path and cfg.source_path.exists()):
        console.print("[muted]No config yet — using safe defaults.\n[/muted]")
    original_backend = cfg.backend
    cfg.backend = "mock"
    banner(console, "Demo mode", "mock model · sample documents · demo calendar")
    ensure_workspace(cfg)

    # Sample documents so research is real (retrieval runs for real).
    demo_dir = cfg.data_root / "demo-documents"
    demo_dir.mkdir(parents=True, exist_ok=True)
    _write_demo_docs(demo_dir)

    section(console, "1 · Index sample documents")
    stats = _index_library(cfg, [demo_dir], quiet=False)
    console.print(f"[ok]{stats.added} documents, {stats.chunks} chunks[/ok]")

    section(console, "2 · Ask a question (real retrieval)")
    with _library(cfg) as lib:
        hits = lib.search("What did the Q4 planning offsite decide about pricing?", top_k=3)
    for i, hit in enumerate(hits, 1):
        console.print(f"[jarvis][{i}][/jarvis] {hit.title} — {hit.text[:140]}…")

    section(console, "3 · Morning digest (demo calendar)")
    from .agents.digest import DigestAgent

    agent = DigestAgent()
    ctx = build_context(cfg, make_backend(cfg), approve=deny_all)
    bundle = agent.fetch_agenda(ctx, force_demo=True)
    _print_agenda_preview(bundle)
    if speak:
        _speak_text(ctx, "Here is your digest. " + bundle.text[:600])

    section(console, "4 · Morning digest, agent style")
    digest_result = AgentRuntime(ctx, agent).run("Give me my morning briefing.")
    console.print(answer_panel(digest_result.text, title="Morning digest"))

    section(console, "5 · Skills")
    install_default_skills(cfg)
    manager = SkillManager(cfg)
    for skill in manager.enabled():
        console.print(f"  [bold]{skill.name}[/bold] — {skill.description}")
    console.print("[muted]  Ask chat “summarise my meeting notes” and the trigger fires.[/muted]")
    ctx.memory.close()
    ctx.state.close()

    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold]That was a full run with no model downloaded.[/bold]\n\n"
            "Next:\n"
            "  1. Install Ollama — https://ollama.com/download\n"
            "  2. [bold]jarvis init[/bold]  (detects your hardware, pulls fitting models)\n"
            "  3. [bold]jarvis ask \"summarise my notes\"[/bold]  or  [bold]jarvis digest --speak[/bold]\n\n"
            f"[muted]Demo documents: {demo_dir}[/muted]"
        ),
        title="Demo complete",
        border_style="jarvis",
    ))
    cfg.backend = original_backend


def _write_demo_docs(folder: Path) -> None:
    docs = {
        "q4-planning-offsite.md": """# Q4 planning offsite — decisions

**Attendees:** Ada, Sam, Priya, Ray
**Date:** 12 October

## Pricing
We agreed to keep the Pro tier at $29/month for existing customers through Q1.
New customers move to usage-based pricing. Ada owns the pricing page rewrite; deadline 1 November.
The team rejected the annual-only discount idea (two votes against, one abstention).

## Hiring
Two backend roles open. Sam will draft the job descriptions by Friday.
No new interns before January.

## Roadmap
- November: offline mode beta
- December: audit log export
- January: plugin marketplace (needs a security review first)

## Risks
The billing migration is the critical path. If it slips a week, the January launch moves too.
""",
        "team-handbook.md": """# Team handbook (excerpt)

## Working hours
Core hours are 10:00–15:00 local time. Outside that, work when you are at your best.
Meetings default to 25 minutes and need an agenda in the invite.

## Communication
- Decisions go in writing, in the issue tracker.
- Urgent = phone call. Everything else can wait for a reply within one working day.
- No status meetings on Fridays; use the async weekly update instead.

## Time off
Request time off at least two weeks ahead. The team calendar must show it.
Public holidays follow the Karachi office calendar.

## Expenses
Keep receipts for anything over $20. Submit within 30 days.
""",
        "reading-notes-rag.md": """# Reading notes — retrieval augmented generation

Hybrid search (BM25 + embeddings) beats either method alone on mixed queries.
For a personal corpus under ~100k chunks, SQLite FTS5 plus vector blobs is
plenty; no vector database server is needed.

Chunk size: 800–1200 characters with 15% overlap works well for prose.
Always store page/line anchors so answers can cite exact locations.

Evaluating RAG: measure whether the correct source appears in the top 5,
and separately whether the answer cites it. The second number is usually worse.

Privacy note: embeddings computed locally never expose document text.
""",
        "trip-islamabad.md": """# Islamabad trip — 3 to 6 December

Flight PK-368 departs Karachi 07:10, arrives Islamabad 09:05.
Hotel: Margalla View, room booked for three nights.
Meetings: Ministry briefing (day 1, 11:00), partner dinner (day 2, 19:30).
Bring: laptop, HDMI adapter, printed NDA copies.
Return flight PK-371 on the 6th at 18:40.
""",
    }
    for name, body in docs.items():
        (folder / name).write_text(body, encoding="utf-8")


@app.command()
def sessions(limit: int = typer.Option(10, "--limit")) -> None:
    """Recent chat sessions."""
    cfg = _load_config()
    store = StateStore(cfg.state_db)
    rows = store.sessions(limit)
    store.close()
    if not rows:
        console.print("[muted]No sessions yet.[/muted]")
        return
    table = Table(box=None, header_style="jarvis", padding=(0, 2))
    table.add_column("Session", style="bold")
    table.add_column("Agent")
    table.add_column("Messages", justify="right")
    table.add_column("Updated")
    table.add_column("Title", overflow="fold")
    for row in rows:
        table.add_row(row["id"], row["agent"], str(row["msgs"]), row["updated_at"][:16], row["title"] or "")
    console.print(table)
    console.print("[muted]Resume with: jarvis chat --session <id>[/muted]")


@app.command()
def agents() -> None:
    """List the agents and their jobs."""
    cfg = _load_config()
    table = Table(box=None, header_style="jarvis", padding=(0, 2))
    table.add_column("Agent", style="bold")
    table.add_column("Model")
    table.add_column("Description", overflow="fold")
    for agent in list_agents():
        table.add_row(agent.name, agent.model_for(cfg), agent.description)
    console.print(table)
    console.print(
        "[muted]chat[/muted]    jarvis chat\n"
        "[muted]research[/muted] jarvis research \"question\"\n"
        "[muted]code[/muted]  jarvis code \"task\"\n"
        "[muted]digest[/muted] jarvis digest --speak"
    )


@app.command()
def logs(lines: int = typer.Option(20, "--lines", "-n")) -> None:
    """Show the audit log (what Jarvis did on your machine)."""
    cfg = _load_config()
    path = cfg.log_dir / "audit.jsonl"
    if not path.exists():
        console.print("[muted]No actions recorded yet.[/muted]")
        return
    entries = path.read_text(encoding="utf-8").splitlines()[-lines:]
    for entry in entries:
        try:
            data = json.loads(entry)
        except json.JSONDecodeError:
            continue
        actor = data.get("actor", "?")
        action = data.get("action", "?")
        ts = data.get("ts", "")[5:16].replace("T", " ")
        detail = {k: v for k, v in data.items() if k not in {"ts", "actor", "action"}}
        console.print(f"[jarvis]{ts}[/jarvis] [bold]{actor}[/bold] {action} [muted]{json.dumps(detail, default=str)[:140]}[/muted]")


def run() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    run()
