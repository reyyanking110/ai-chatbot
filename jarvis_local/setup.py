"""Setup, onboarding and diagnostics.

``jarvis init``   — detect hardware, choose models, write config, pull models.
``jarvis doctor`` — check every moving part and print concrete fixes.

This module also generates the sample Google OAuth client secret so users
can start the OAuth flow immediately during onboarding (the file contains no
real credentials; it points at Google's installed-app redirect).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .backends import make_backend
from .config import Config
from .hardware import ModelPlan, SystemProfile, detect_hardware, recommend_models
from .integrations.google import google_status
from .util import human_bytes, which

SAMPLE_CLIENT_SECRET = {
    "installed": {
        "client_id": "PASTE-FROM-GOOGLE-CLOUD.apps.googleusercontent.com",
        "client_secret": "PASTE-FROM-GOOGLE-CLOUD",
        "project_id": "jarvis-local",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "redirect_uris": ["http://localhost"],
    }
}

OLLAMA_INSTALL = {
    "Darwin": "brew install ollama   (or download from https://ollama.com/download)",
    "Linux": "curl -fsSL https://ollama.com/install.sh | sh",
    "Windows": "winget install Ollama.Ollama   (or https://ollama.com/download)",
}


@dataclass
class SetupPlan:
    profile: SystemProfile
    plan: ModelPlan
    config: Config
    notes: list[str] = field(default_factory=list)


def build_plan(
    cfg: Config | None = None,
    *,
    budget_gb: float | None = None,
    backend: str | None = None,
) -> SetupPlan:
    cfg = cfg or Config()
    profile = detect_hardware()
    budget = profile.budget_gb(budget_gb if budget_gb else cfg.hardware.budget_gb)
    plan = recommend_models(budget, ram_gb=profile.ram_gb)

    cfg.hardware.device = (
        "unified" if profile.unified_memory else ("gpu" if profile.accelerator in {"cuda", "rocm", "vulkan"} else "cpu")
    )
    cfg.hardware.budget_gb = round(budget, 1)
    cfg.models.general = plan.general
    cfg.models.coding = plan.coding
    cfg.models.fast = plan.fast
    cfg.models.embed = plan.embed
    cfg.models.num_ctx = plan.num_ctx
    if backend:
        cfg.backend = backend
    return SetupPlan(profile=profile, plan=plan, config=cfg, notes=list(plan.notes))


def write_default_config(cfg: Config, path: Path | None = None) -> Path:
    cfg.ensure_dirs()
    return cfg.save(path)


class RedirectedStdin:
    """A tiny stand-in for stdin that answers prompts with newlines.

    Used when Jarvis is not attached to a terminal (piped input, CI, a
    `jarvis ask ...` inside a script) so setup never blocks on a prompt.
    """

    def isatty(self) -> bool:  # pragma: no cover - trivial
        return False

    def read(self, *args, **kwargs) -> str:
        return ""

    def readline(self, *args, **kwargs) -> str:
        return "\n"

    def __iter__(self):
        return iter(())


def write_sample_client_secret(cfg: Config, overwrite: bool = False) -> Path | None:
    """Create a placeholder OAuth file so the flow can be completed later."""
    path = cfg.google_client_secret
    if path.exists() and not overwrite:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SAMPLE_CLIENT_SECRET, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


GOOGLE_LIBS_HINT = "pip install 'jarvis-local[google]'  (or reinstall with `pip install -e .[google]`)"


def google_deps_installable() -> bool:
    """True when we run from a project checkout we can pip-install extras into."""
    import sys

    root = Path(__file__).resolve().parent.parent
    in_project = (root / "pyproject.toml").exists() and Path(sys.prefix).exists()
    return in_project


def install_google_libs(cfg: Config) -> tuple[bool, str]:
    """Install the optional Google client libraries into the running interpreter."""
    import sys

    if not google_deps_installable():
        return False, GOOGLE_LIBS_HINT
    cmd = [sys.executable, "-m", "pip", "install", "-q", "google-api-python-client", "google-auth-oauthlib"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{exc}. Try manually: {GOOGLE_LIBS_HINT}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return False, "\n".join(tail)
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: F401
        from googleapiclient.discovery import build  # noqa: F401
    except ImportError:
        return False, GOOGLE_LIBS_HINT
    return True, "Google libraries installed."


def client_secret_is_placeholder(cfg: Config) -> bool:
    path = cfg.google_client_secret
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    block = data.get("installed") or data.get("web") or {}
    client_id = str(block.get("client_id", ""))
    return "PASTE-FROM-GOOGLE-CLOUD" in client_id or not client_id


# ---------------------------------------------------------------------------
# Ollama lifecycle
# ---------------------------------------------------------------------------

def ollama_present() -> bool:
    return bool(which("ollama"))


def ollama_running(cfg: Config) -> bool:
    try:
        with httpx.Client(base_url=cfg.ollama.host.rstrip("/"), timeout=2.5, trust_env=False) as client:
            return client.get("/api/tags").status_code == 200
    except httpx.HTTPError:
        return False


def start_ollama(cfg: Config, wait_s: float = 12.0) -> bool:
    """Start `ollama serve` in the background if it is installed but not running."""
    if ollama_running(cfg):
        return True
    if not ollama_present():
        return False
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if ollama_running(cfg):
            return True
        time.sleep(0.4)
    return ollama_running(cfg)


def installed_models(cfg: Config) -> list[str]:
    try:
        with httpx.Client(base_url=cfg.ollama.host.rstrip("/"), timeout=4, trust_env=False) as client:
            resp = client.get("/api/tags")
            resp.raise_for_status()
            return sorted(m.get("name", "") for m in resp.json().get("models", []))
    except httpx.HTTPError:
        return []


def model_installed(cfg: Config, tag: str) -> bool:
    names = installed_models(cfg)
    base = tag.split(":")[0]
    return any(name == tag or name.split(":")[0] == base and (":" not in tag or name == tag) for name in names)


def pull_model(cfg: Config, tag: str, on_line=None) -> bool:
    """Pull a model using the Ollama CLI (shows progress, respects the daemon)."""
    if not ollama_present():
        return False
    env = dict(os.environ)
    env.setdefault("OLLAMA_HOST", cfg.ollama.host.replace("http://", "").replace("https://", ""))
    try:
        proc = subprocess.Popen(
            ["ollama", "pull", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
        )
    except OSError:
        return False
    assert proc.stdout is not None
    for line in proc.stdout:
        if on_line:
            on_line(line.rstrip())
    return proc.wait() == 0


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | info
    detail: str
    fix: str = ""


def run_doctor(cfg: Config, *, deep: bool = False) -> list[Check]:
    checks: list[Check] = []

    # Config file
    if cfg.source_path and cfg.source_path.exists():
        checks.append(Check("Config file", "ok", str(cfg.source_path)))
    else:
        checks.append(
            Check(
                "Config file",
                "warn",
                f"No config at {cfg.config_file} — running on defaults.",
                fix="Run `jarvis init` to detect hardware and write a config.",
            )
        )
    for warning in cfg.warnings:
        checks.append(Check("Config warning", "warn", warning, fix="Fix the key in config.toml."))

    # Python version (3.11+ is enforced by packaging, checked for clarity)
    checks.append(Check("Python", "ok", platform_python()))

    # Hardware
    profile = detect_hardware()
    checks.append(Check("Hardware", "info", profile.summary().replace("\n", " · ")))
    for note in profile.notes:
        if "Ollama is not installed" in note:
            continue
        checks.append(Check("Hardware note", "info", note))

    # Disk
    try:
        free = shutil.disk_usage(str(cfg.data_root)).free
        if free > 5 * 1024**3:
            checks.append(Check("Disk", "ok", f"{human_bytes(free)} free in {cfg.data_root}"))
        else:
            checks.append(
                Check("Disk", "warn", f"Only {human_bytes(free)} free", fix="Models need 2–20 GB each.")
            )
    except OSError:
        pass

    # Ollama
    install_hint = OLLAMA_INSTALL.get(sys.platform if sys.platform != "linux" else "Linux", "")
    if cfg.backend == "ollama":
        if not ollama_present():
            checks.append(
                Check(
                    "Ollama binary",
                    "fail",
                    "`ollama` is not on PATH.",
                    fix=f"Install it: {install_hint}",
                )
            )
        elif ollama_running(cfg):
            version = ""
            try:
                with httpx.Client(base_url=cfg.ollama.host.rstrip("/"), timeout=3, trust_env=False) as client:
                    version = client.get("/api/version").json().get("version", "")
            except httpx.HTTPError:
                pass
            checks.append(Check("Ollama server", "ok", f"{cfg.ollama.host} responding {version}".strip()))
            for label, tag in (
                ("general model", cfg.models.general),
                ("coding model", cfg.models.coding),
                ("fast model", cfg.models.fast),
                ("embedding model", cfg.models.embed),
            ):
                if model_installed(cfg, tag):
                    checks.append(Check(label.capitalize(), "ok", tag))
                else:
                    checks.append(
                        Check(
                            label.capitalize(),
                            "warn",
                            f"{tag} is not pulled.",
                            fix=f"`ollama pull {tag}`",
                        )
                    )
            if deep and model_installed(cfg, cfg.models.general):
                checks.append(_test_generation(cfg))
        else:
            checks.append(
                Check(
                    "Ollama server",
                    "fail",
                    f"Installed but not answering at {cfg.ollama.host}.",
                    fix="`ollama serve` (or set ollama.autostart = true and rerun `jarvis init`).",
                )
            )
    elif cfg.backend == "openai-compatible":
        backend = make_backend(cfg)
        health = backend.health()
        checks.append(
            Check(
                "Local model server",
                "ok" if health.ok else "fail",
                health.detail,
                fix="" if health.ok else f"Start your server (llama.cpp / LM Studio) at {cfg.openai.base_url}.",
            )
        )
    else:
        checks.append(Check("Backend", "info", f"{cfg.backend} (no model server needed)"))

    # Library
    try:
        from .documents import Library

        lib = Library(cfg)
        stats = lib.stats()
        lib.close()
        if stats["documents"]:
            checks.append(
                Check(
                    "Document library",
                    "ok",
                    f"{stats['documents']} documents · {stats['chunks']} chunks · "
                    f"{human_bytes(stats['db_size'])}",
                )
            )
            if cfg.library.use_embeddings and stats["vectors"] == 0:
                checks.append(
                    Check(
                        "Embeddings",
                        "warn",
                        "Semantic index is empty; search is keyword-only.",
                        fix="`jarvis library index` after the embedding model is pulled.",
                    )
                )
        else:
            checks.append(
                Check(
                    "Document library",
                    "warn",
                    "Nothing indexed yet.",
                    fix="`jarvis library index ~/Documents`  (or wherever your notes live)",
                )
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Document library", "fail", str(exc), fix="Run `jarvis library reset` to rebuild."))

    # Voice
    from .voice import VoiceEngine, available_engines

    engines = available_engines()
    if engines:
        checks.append(Check("Text-to-speech", "ok", ", ".join(engines) + f" ({VoiceEngine(cfg).describe()})"))
    else:
        checks.append(
            Check(
                "Text-to-speech",
                "warn",
                "No TTS engine found; digests will be text only.",
                fix="macOS has `say`. Linux: `sudo apt install espeak-ng` or install piper.",
            )
        )

    # Google
    gstatus = google_status(cfg)
    if not gstatus["client_secret"]:
        checks.append(
            Check(
                "Gmail + Calendar",
                "info",
                "Not configured (optional).",
                fix="`jarvis google setup` — software install, OAuth walkthrough, then `jarvis google auth`.",
            )
        )
    elif client_secret_is_placeholder(cfg):
        checks.append(
            Check(
                "Gmail + Calendar",
                "warn",
                f"Placeholder OAuth file at {gstatus['client_secret_path']}.",
                fix="Replace client_id/client_secret with your Desktop app credentials, then `jarvis google auth`.",
            )
        )
    elif not gstatus["token"]:
        checks.append(
            Check(
                "Gmail + Calendar",
                "warn",
                "Client secret OK, not authorised yet.",
                fix="`jarvis google auth`",
            )
        )
    elif not gstatus["libraries"]:
        checks.append(
            Check(
                "Gmail + Calendar",
                "fail",
                "Token present but the Google libraries are missing.",
                fix="`jarvis google setup` can install them, or: pip install 'jarvis-local[google]'",
            )
        )
    else:
        checks.append(Check("Gmail + Calendar", "ok", "Authorised (read-only scopes)"))

    # Skills
    from .skills import SkillManager

    manager = SkillManager(cfg)
    skills = manager.list()
    enabled = [s for s in skills if s.enabled]
    checks.append(
        Check(
            "Skills",
            "ok" if skills else "info",
            f"{len(enabled)} enabled of {len(skills)} installed.",
            fix="" if skills else "`jarvis skills new my-skill` or `jarvis skills install <github-url>`",
        )
    )
    for skill in skills:
        if skill.enabled and skill.security.risk == "high":
            checks.append(
                Check(
                    f"Skill '{skill.name}'",
                    "warn",
                    "High-risk content (executables or credential strings).",
                    fix="Review the folder; disable with `jarvis skills disable " + skill.name + "`.",
                )
            )

    # Policies
    checks.append(
        Check(
            "Safety",
            "ok",
            f"approval={cfg.security.approval}, shell={'on' if cfg.security.allow_shell else 'off'}, "
            f"writes limited to {cfg.workspace}",
        )
    )

    # Autonomy: knows when to hand off instead of guessing.
    if cfg.models.general and cfg.models.coding:
        fits = _plan_fits(profile, cfg)
        if not fits:
            checks.append(
                Check(
                    "Model fit",
                    "warn",
                    f"{cfg.models.general} may not fit in {profile.compute_label}.",
                    fix="`jarvis models plan` for recommendations, or `jarvis init --force` to re-detect.",
                )
            )
        else:
            checks.append(Check("Model fit", "ok", f"Configured models fit {profile.compute_label}"))
    if cfg.models.num_ctx > 16384 and not profile.unified_memory and profile.ram_gb < 24:
        checks.append(
            Check(
                "Context size",
                "warn",
                f"{cfg.models.num_ctx} tokens on {profile.ram_gb:.0f} GB RAM will be slow.",
                fix="`jarvis config set models.num_ctx 8192`",
            )
        )

    # Scheduler (digest automation)
    from .cli import _read_crontab  # local import: CLI helper, no side effects

    cron = _read_crontab()
    scheduled = [line for line in cron.splitlines() if "jarvis digest" in line]
    if scheduled:
        checks.append(Check("Digest schedule", "ok", scheduled[0].split("jarvis")[0].strip() + " (cron)"))
    else:
        checks.append(
            Check(
                "Digest schedule",
                "info",
                "Morning digest is not scheduled.",
                fix="`jarvis schedule install -t 07:30`",
            )
        )
    return checks


def _plan_fits(profile: SystemProfile, cfg: Config) -> bool:
    from .hardware import card_for

    budget = cfg.hardware.budget_gb or profile.budget_gb()
    for tag in (cfg.models.general, cfg.models.coding):
        card = card_for(tag)
        if card and card.size_gb > budget * 1.05:
            return False
    return True


def _test_generation(cfg: Config) -> Check:
    from .backends import make_backend

    try:
        backend = make_backend(cfg)
        started = time.monotonic()
        result = backend.chat(
            cfg.models.general,
            [{"role": "user", "content": "Reply with the single word: ready"}],
            options={"num_ctx": min(cfg.models.num_ctx, 2048), "num_predict": 12},
        )
        elapsed = time.monotonic() - started
        return Check(
            "Generation test",
            "ok",
            f"{cfg.models.general} answered in {elapsed:.1f}s: {result.content.strip()[:60]!r}",
        )
    except Exception as exc:  # noqa: BLE001
        return Check(
            "Generation test",
            "fail",
            str(exc)[:200],
            fix="Check `jarvis doctor` output above; try a smaller model with `jarvis models switch`.",
        )


def platform_python() -> str:
    return f"{sys.version.split()[0]} ({sys.executable})"


def switch_model_plan(cfg: Config, role: str, tag: str | None = None) -> tuple[str, str]:
    """Change one or all model roles, returning (old, new)."""
    role_map = {
        "general": "general",
        "coding": "coding",
        "code": "coding",
        "fast": "fast",
        "embed": "embed",
        "embedding": "embed",
    }
    key = role_map.get(role.lower())
    if key is None:
        raise ValueError(f"Unknown model role '{role}'. Use general, coding, fast or embed.")
    old = getattr(cfg.models, key)
    if tag:
        new = tag
    else:
        profile = detect_hardware()
        budget = cfg.hardware.budget_gb or profile.budget_gb()
        plan = recommend_models(budget, profile.ram_gb)
        new = getattr(plan, key)
        if not new:
            raise ValueError(f"No {key} model fits your current budget.")
    setattr(cfg.models, key, new)
    return old, new


def ensure_workspace(cfg: Config) -> Path:
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    readme = cfg.workspace / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Jarvis workspace\n\n"
            "This folder is where Jarvis writes files when you ask it to.\n"
            "It is also the only place the code assistant may modify.\n\n"
            "Put project folders here, or point `paths.workspace` somewhere else.\n",
            encoding="utf-8",
        )
    return cfg.workspace
