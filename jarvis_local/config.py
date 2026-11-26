"""Configuration loading, validation and TOML persistence.

The config lives in an XDG-style directory (override with ``JARVIS_CONFIG_DIR``)
and runtime data (databases, skills, briefings) lives in a separate data
directory (``JARVIS_DATA_DIR``).  Both are safe to point at a synced or
encrypted folder if you prefer.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir, user_data_dir

from .errors import ConfigError

CONFIG_FILENAME = "config.toml"
APP_SLUG = "jarvis"

RUNTIME_FIELDS = {"config_root", "data_root", "source_path", "warnings"}


def expand_path(value: str | Path) -> Path:
    """Expand ``~`` and environment variables without requiring the path to exist."""
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def default_config_dir() -> Path:
    env = os.environ.get("JARVIS_CONFIG_DIR")
    if env:
        return expand_path(env)
    return Path(user_config_dir(APP_SLUG))


def default_data_dir() -> Path:
    env = os.environ.get("JARVIS_DATA_DIR")
    if env:
        return expand_path(env)
    return Path(user_data_dir(APP_SLUG))


def default_config_file() -> Path:
    env = os.environ.get("JARVIS_CONFIG_FILE")
    if env:
        return expand_path(env)
    return default_config_dir() / CONFIG_FILENAME


@dataclass
class OllamaSettings:
    host: str = "http://127.0.0.1:11434"
    autostart: bool = True
    timeout_s: int = 180


@dataclass
class OpenAISettings:
    """Any OpenAI-compatible local server: llama.cpp, LM Studio, vLLM, ..."""

    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = ""
    model: str = "local-model"


@dataclass
class ModelSettings:
    general: str = "qwen3:8b"
    coding: str = "qwen2.5-coder:7b"
    fast: str = "qwen3:1.7b"
    embed: str = "nomic-embed-text"
    num_ctx: int = 8192
    temperature: float = 0.3


@dataclass
class HardwareSettings:
    device: str = "auto"  # auto | gpu | cpu | unified
    budget_gb: float | None = None


@dataclass
class PathSettings:
    workspace: str = "~/jarvis-workspace"
    library: list[str] = field(default_factory=lambda: ["~/Documents"])
    extra_roots: list[str] = field(default_factory=list)
    ignore: list[str] = field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".cache",
            "Library",
            "AppData",
            ".Trash",
            "dist",
            "build",
        ]
    )


@dataclass
class VoiceSettings:
    engine: str = "auto"  # auto | say | espeak | piper | none
    voice: str = ""
    rate: int = 180
    piper_model: str = ""
    play: bool = True
    keep_transcript: bool = True


@dataclass
class SecuritySettings:
    approval: str = "confirm"  # confirm | auto | read-only
    allow_shell: bool = True
    max_output_chars: int = 12000
    shell_timeout_s: int = 60
    python_timeout_s: int = 30
    max_write_bytes: int = 5_000_000


@dataclass
class LibrarySettings:
    chunk_chars: int = 1200
    chunk_overlap: int = 200
    max_file_mb: int = 20
    use_embeddings: bool = True
    extensions: list[str] = field(
        default_factory=lambda: [
            ".txt", ".md", ".markdown", ".rst", ".org", ".log",
            ".pdf", ".html", ".htm", ".csv", ".tsv", ".json",
            ".docx", ".odt", ".epub",
        ]
    )


@dataclass
class GoogleSettings:
    client_secret: str = ""
    token_file: str = ""
    calendar_id: str = "primary"
    mail_query: str = "is:unread newer_than:1d -category:promotions -category:social"
    max_mails: int = 12
    hours_back: int = 24


@dataclass
class SkillSettings:
    enabled: bool = True
    require_confirm: bool = True
    inject_agents: list[str] = field(default_factory=lambda: ["chat"])


@dataclass
class AgentSettings:
    max_steps: int = 8
    history_messages: int = 16


@dataclass
class Config:
    version: int = 1
    assistant_name: str = "Jarvis"
    backend: str = "ollama"  # ollama | openai-compatible | mock
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    hardware: HardwareSettings = field(default_factory=HardwareSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    library: LibrarySettings = field(default_factory=LibrarySettings)
    google: GoogleSettings = field(default_factory=GoogleSettings)
    skills: SkillSettings = field(default_factory=SkillSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)

    # Runtime-only (never written back to TOML)
    config_root: Path = field(default_factory=default_config_dir)
    data_root: Path = field(default_factory=default_data_dir)
    source_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    # ---- derived paths -------------------------------------------------
    @property
    def config_file(self) -> Path:
        return self.source_path or (self.config_root / CONFIG_FILENAME)

    @property
    def workspace(self) -> Path:
        return expand_path(self.paths.workspace)

    @property
    def library_db(self) -> Path:
        return self.data_root / "library.db"

    @property
    def state_db(self) -> Path:
        return self.data_root / "state.db"

    @property
    def skills_dir(self) -> Path:
        return self.data_root / "skills"

    @property
    def skills_index(self) -> Path:
        return self.data_root / "skills.json"

    @property
    def briefings_dir(self) -> Path:
        return self.data_root / "briefings"

    @property
    def log_dir(self) -> Path:
        return self.data_root / "logs"

    @property
    def google_client_secret(self) -> Path:
        return expand_path(self.google.client_secret) if self.google.client_secret else (
            self.config_root / "google_client_secret.json"
        )

    @property
    def google_token_file(self) -> Path:
        return expand_path(self.google.token_file) if self.google.token_file else (
            self.data_root / "google_token.json"
        )

    def library_roots(self) -> list[Path]:
        return [expand_path(p) for p in self.paths.library]

    def extra_roots(self) -> list[Path]:
        return [expand_path(p) for p in self.paths.extra_roots]

    # ---- lifecycle -----------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        target = expand_path(path) if path else default_config_file()
        cfg = cls(
            config_root=target.parent,
            data_root=default_data_dir(),
        )
        if target.exists():
            try:
                data = tomllib.loads(target.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError as exc:
                raise ConfigError(
                    f"Could not parse {target}: {exc}",
                    hint="Fix the TOML syntax, or move the file away and run `jarvis init`.",
                ) from exc
            if not isinstance(data, dict):
                raise ConfigError(f"{target} does not contain a TOML table.")
            _merge(cfg, data, prefix="", warnings=cfg.warnings)
            cfg.source_path = target
        _apply_env(cfg)
        return cfg

    def save(self, path: str | Path | None = None) -> Path:
        target = expand_path(path) if path else (
            self.source_path or (self.config_root / CONFIG_FILENAME)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _serializable(self)
        text = (
            "# Jarvis Local configuration\n"
            "# Edit with `jarvis config set <key> <value>` or in any editor.\n"
            "# Secrets (Google client secret, tokens) live beside this file or in the data dir.\n\n"
            + dump_toml(data)
        )
        target.write_text(text, encoding="utf-8")
        try:
            target.chmod(0o600)
        except OSError:
            pass
        self.source_path = target
        self.config_root = target.parent
        return target

    def ensure_dirs(self, create_workspace: bool = True) -> None:
        for directory in (self.config_root, self.data_root, self.skills_dir, self.briefings_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if create_workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)

    def set_value(self, dotted: str, raw: str) -> tuple[Any, Any]:
        """Set a config value from a dotted key and string value."""
        parts = dotted.split(".")
        if not parts:
            raise ConfigError("Empty configuration key.")
        obj: Any = self
        for part in parts[:-1]:
            if not hasattr(obj, part) or not is_dataclass(getattr(obj, part)):
                raise ConfigError(f"Unknown configuration section in key: {dotted}")
            obj = getattr(obj, part)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise ConfigError(
                f"Unknown configuration key: {dotted}",
                hint="See `jarvis config show` for valid keys.",
            )
        old = getattr(obj, leaf)
        new = _coerce(raw, old, dotted)
        if new != old:
            setattr(obj, leaf, new)
        return old, new

    def copy(self) -> Config:
        return copy.deepcopy(self)


def _apply_env(cfg: Config) -> None:
    if backend := os.environ.get("JARVIS_BACKEND"):
        cfg.backend = backend
    if host := os.environ.get("JARVIS_OLLAMA_HOST"):
        cfg.ollama.host = host
    if ws := os.environ.get("JARVIS_WORKSPACE"):
        cfg.paths.workspace = ws


def _merge(obj: Any, data: dict[str, Any], prefix: str, warnings: list[str]) -> None:
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        label = f"{prefix}{key}"
        if key not in known:
            warnings.append(f"Unknown config key ignored: {label}")
            continue
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                warnings.append(f"Expected a table for '{label}', got {type(value).__name__}.")
                continue
            _merge(current, value, prefix=label + ".", warnings=warnings)
            continue
        try:
            setattr(obj, key, _coerce_typed(value, current, label))
        except ConfigError as exc:
            warnings.append(exc.message)


def _coerce_typed(value: Any, current: Any, label: str) -> Any:
    if current is None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if value in ("", "none", "auto"):
            return None
        raise ConfigError(f"'{label}' must be a number or null.")
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"'{label}' must be true or false.")
        return value
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{label}' must be an integer.")
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{label}' must be a number.")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ConfigError(f"'{label}' must be a string.")
        return value
    if isinstance(current, list):
        if not isinstance(value, list):
            raise ConfigError(f"'{label}' must be a list.")
        return list(value)
    return value


def _coerce(raw: str, current: Any, label: str) -> Any:
    if current is None:
        if raw.strip().lower() in {"", "none", "auto"}:
            return None
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"'{label}' expects a number or 'none'.") from exc
    if isinstance(current, bool):
        if raw.strip().lower() in {"1", "true", "yes", "on"}:
            return True
        if raw.strip().lower() in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"'{label}' expects true/false.")
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"'{label}' expects an integer.") from exc
    if isinstance(current, float):
        if raw.strip().lower() == "auto":
            return None
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"'{label}' expects a number.") from exc
    if isinstance(current, str):
        return raw
    if isinstance(current, list):
        text = raw.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"'{label}' expects a comma-separated list or JSON array.") from exc
            if not isinstance(value, list):
                raise ConfigError(f"'{label}' expects a list.")
            return [str(v) for v in value]
        return [item.strip() for item in text.split(",") if item.strip()] if text else []
    return raw


def _serializable(cfg: Config) -> dict[str, Any]:
    data = _to_plain(cfg)
    for key in RUNTIME_FIELDS:
        data.pop(key, None)
    return data


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _fmt_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # TOML basic strings use the same escapes as JSON for our purposes.
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_toml_value(v) for v in value) + "]"
    raise TypeError(f"Cannot serialise {type(value).__name__} to TOML")


def dump_toml(data: dict[str, Any], prefix: str = "") -> str:
    lines: list[str] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            tables.append((prefix + key, value))
        else:
            lines.append(f"{key} = {_fmt_toml_value(value)}")
    for name, table in tables:
        if lines:
            lines.append("")
        lines.append(f"[{name}]")
        for key, value in table.items():
            if value is None:
                continue
            if isinstance(value, dict):
                # Our schema is at most two levels deep; flatten deeper tables below.
                continue
            lines.append(f"{key} = {_fmt_toml_value(value)}")
    return "\n".join(lines).rstrip() + "\n"


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "***"
    return value[:3] + "…" + value[-2:]


def redact(text: str) -> str:
    """Best-effort redaction of likely secrets in logs and approval prompts."""
    patterns = [
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}",
        r"(?i)(api[_-]?key|token|secret|password|passwd)(\s*[=:]\s*)(['\"]?)([^\s'\"]{4,})\3",
        r"\b(sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|xox[abp]-[A-Za-z0-9-]{10,})\b",
    ]
    replacements = [
        "-----BEGIN PRIVATE KEY----- [redacted] -----END PRIVATE KEY-----",
        r"\1[redacted]",
        r"\1\2\3[redacted]\3",
        "[redacted-secret]",
    ]
    out = text
    for pattern, replacement in zip(patterns, replacements, strict=True):
        out = re.sub(pattern, replacement, out, flags=re.S)
    return out
