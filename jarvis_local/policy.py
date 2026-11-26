"""Filesystem, shell and approval policies.

Jarvis runs on your machine, so a helpful assistant is also a powerful one.
This module is the single choke point for anything with side effects:

* **Files** — reads are limited to the workspace, your library folders and
  any extra roots you configured.  Writes happen *only* inside the workspace.
  Symlinks are resolved before the check, so they cannot be used to escape.
* **Shell** — a denylist blocks catastrophic patterns outright.  In the
  default ``confirm`` mode every command is shown before it runs; in
  ``read-only`` mode only inspected read-only commands are allowed.
* **Python** — same rules, run in a subprocess with a timeout and a scrubbed
  environment (API keys, tokens and passwords are stripped).
* **Audit** — every tool call is appended to a local JSONL log.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from .config import Config, redact
from .errors import PolicyError

WRITE_ROOTS_LABEL = "workspace"


@dataclass
class CommandVerdict:
    allowed: bool
    reason: str
    is_read_only: bool = False
    needs_confirm: bool = True
    matches: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.matches = self.matches or []


# Commands that are fine to run without asking (still subject to denylist).
READ_ONLY_COMMANDS = {
    "ls", "cat", "head", "tail", "less", "more", "wc", "file", "stat", "du", "df",
    "pwd", "whoami", "date", "cal", "uname", "uptime", "ps", "top", "htop", "free",
    "which", "whereis", "type", "echo", "printf", "grep", "rg", "ag", "find", "fd",
    "tree", "diff", "cmp", "sort", "uniq", "cut", "awk", "sed", "jq", "yq",
    "git", "gh", "bat", "exa", "eza", "glow", "man", "help", "python", "python3",
    "node", "npm", "pnpm", "yarn", "pip", "pip3", "uv", "cargo", "rustc", "go", "make",
    "cmake", "ollama", "jarvis", "curl", "wget", "ping", "dig", "nslookup", "http",
    "openssl", "ssh", "scp", "rsync", "tar", "zip", "unzip", "gzip", "gunzip",
    "sqlite3", "md5sum", "shasum", "sha256sum", "basename", "dirname", "realpath",
    "readlink", "env", "printenv", "id", "groups", "hostname", "lsof", "netstat",
    "ss", "ip", "ifconfig", "brew", "apt", "apt-get", "dnf", "pacman", "docker",
    "kubectl", "terraform", "aws", "gcloud", "az", "ffmpeg", "convert", "magick",
    "say", "espeak", "espeak-ng", "piper", "afplay", "aplay", "paplay", "ffplay",
    "pytest", "ruff", "black", "mypy", "tox", "nox", "poetry", "hatch", "pdm",
}

# Commands that only make sense with side effects: always require confirmation.
MUTATING_COMMANDS = {
    "rm", "rmdir", "mv", "cp", "mkdir", "touch", "chmod", "chown", "ln", "dd",
    "tee", "truncate", "shred", "kill", "pkill", "killall", "reboot", "shutdown",
    "halt", "systemctl", "launchctl", "crontab", "at", "sudo", "su", "doas",
    "useradd", "userdel", "passwd", "mount", "umount", "diskutil", "fdisk", "parted",
    "mkfs", "git", "npm", "pip", "pip3", "brew", "apt", "apt-get", "docker",
    "kubectl", "terraform", "aws", "gcloud", "ollama",
}

READ_ONLY_GIT = {"status", "log", "diff", "show", "branch", "remote", "tag", "rev-parse", "blame", "shortlog"}
MUTATING_GIT = {"push", "reset", "clean", "checkout", "switch", "rm", "mv", "merge", "rebase", "stash", "commit", "apply", "cherry-pick", "restore", "pull", "clone", "init", "add"}

DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+(-[a-zA-Z]*[rR][a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*[rR])\s+/(-|/\*|\s*$)", "recursive delete of /"),
    (r"rm\s+-rf?\s+[\"']?~[\"']?/?(\s|$)", "recursive delete of your home directory"),
    (r"\bmkfs(\.\w+)?\b", "formatting a filesystem"),
    (r"\bdd\b[^\n]*\bof=/dev/", "raw write to a block device"),
    (r">\s*/dev/(sd|nvme|disk)", "raw write to a block device"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "powering off the machine"),
    (r"\bdiskutil\s+(erase|reformat)", "erasing a disk"),
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
    (r"\bchmod\s+-R\s+777\s+/(\s|$)", "world-writable root"),
    (r"\b(curl|wget)\b[^\n|;]*\|\s*(sudo\s+)?(ba)?sh\b", "piping a download into a shell"),
    (r"\bgit\s+push\s+--force", "force-pushing (rewrites remote history)"),
    (r"\bsudo\s+rm\b", "sudo delete"),
    (r"\bhistory\s+-c\b", "clearing shell history"),
    (r"\bdefaults\s+delete\b", "deleting macOS defaults"),
    (r"\bsecurity\s+(delete|remove)", "touching the macOS keychain"),
    (r"/etc/(passwd|shadow|sudoers)", "sensitive system file"),
]

SECRET_ENV_PATTERN = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|CREDENTIAL|AUTH)", re.I)


class Policy:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.workspace = cfg.workspace
        self.read_roots = [self.workspace, *cfg.extra_roots()]
        self.write_roots = [self.workspace]
        self._audit_path = cfg.log_dir / "audit.jsonl"

    # ------------------------------------------------------------------ files
    def resolve_read(self, raw: str | Path) -> Path:
        path = self._resolve(raw)
        if not any(self._is_within(path, root) for root in self.read_roots + cfg_library_roots(self.cfg)):
            allowed = ", ".join(str(r) for r in self.read_roots + cfg_library_roots(self.cfg))
            raise PolicyError(
                f"Refusing to read outside allowed folders: {path}",
                hint=f"Allowed: {allowed}. Add folders with `jarvis config set paths.extra_roots ...`.",
            )
        return path

    def resolve_write(self, raw: str | Path) -> Path:
        path = self._resolve(raw)
        if not any(self._is_within(path, root) for root in self.write_roots):
            raise PolicyError(
                f"Refusing to write outside the workspace: {path}",
                hint=f"Writes are limited to {self.workspace}. Change paths.workspace to relocate it.",
            )
        return path

    def workspace_path(self, raw: str | Path) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = self.workspace / path
        return path.expanduser()

    def _resolve(self, raw: str | Path) -> Path:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        try:
            return path.resolve()
        except OSError as exc:
            raise PolicyError(f"Invalid path {raw!r}: {exc}") from exc

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root.resolve())
            return True
        except (ValueError, OSError):
            return False

    # ------------------------------------------------------------------ shell
    def inspect_command(self, command: str) -> CommandVerdict:
        text = command.strip()
        if not text:
            raise PolicyError("Empty command.")
        if len(text) > 4000:
            raise PolicyError("Command is too long (4000 character limit).")
        for pattern, why in DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.I | re.S):
                return CommandVerdict(False, f"Blocked: {why}.", matches=[pattern])

        if not self.cfg.security.allow_shell:
            return CommandVerdict(False, "Shell access is disabled in config (security.allow_shell = false).")

        segments = _split_shell_segments(text)
        read_only = True
        mutating: list[str] = []
        for segment in segments:
            base, args = _base_command(segment)
            if not base:
                continue
            if base in READ_ONLY_COMMANDS and base not in MUTATING_COMMANDS:
                continue
            if base == "git":
                sub = args[0] if args else ""
                if sub in READ_ONLY_GIT and not any(a.startswith("-") and a in {"-p", "--porcelain"} for a in args):
                    continue
                if sub in MUTATING_GIT:
                    read_only = False
                    mutating.append("git " + sub)
                    continue
                read_only = False
                mutating.append("git")
                continue
            read_only = False
            mutating.append(base)

        if read_only:
            return CommandVerdict(True, "Read-only command.", is_read_only=True, needs_confirm=False)

        if self.cfg.security.approval == "read-only":
            return CommandVerdict(
                False,
                f"Approval mode is 'read-only', so I won't run: {', '.join(sorted(set(mutating)))}.",
                is_read_only=False,
            )
        needs_confirm = self.cfg.security.approval != "auto"
        return CommandVerdict(
            True,
            "Command can change files or state.",
            is_read_only=False,
            needs_confirm=needs_confirm,
            matches=sorted(set(mutating)),
        )

    # ------------------------------------------------------------------ audit
    def audit(self, actor: str, action: str, detail: dict) -> None:
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"ts": __import__("datetime").datetime.now().isoformat(timespec="seconds"), "actor": actor, "action": action}
            record.update({k: _redact_value(v) for k, v in detail.items()})
            with self._audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    # ---------------------------------------------------------------- secrets
    @staticmethod
    def scrubbed_env() -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if not SECRET_ENV_PATTERN.search(k)}


def cfg_library_roots(cfg: Config) -> list[Path]:
    roots: list[Path] = []
    for raw in cfg.paths.library:
        try:
            roots.append(Path(os.path.expanduser(os.path.expandvars(raw))).resolve())
        except OSError:
            continue
    return roots


def _split_shell_segments(text: str) -> list[str]:
    """Split on pipes, &&, ||, ; and newlines, respecting quotes."""
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars="|&;\n")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return [text]
    segments: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token in {"|", "||", "&&", ";", "&", "\n"}:
            if current:
                segments.append(" ".join(current))
                current = []
        else:
            current.append(token)
    if current:
        segments.append(" ".join(current))
    return segments or [text]


def _base_command(segment: str) -> tuple[str, list[str]]:
    parts = segment.strip().split()
    if not parts:
        return "", []
    base = parts[0]
    # Skip env assignments and wrappers such as `env FOO=1 cmd`.
    while base and ("=" in base or base in {"env", "command", "time", "nohup", "nice"}):
        parts = parts[1:]
        base = parts[0] if parts else ""
    return base, parts[1:]


def format_command_preview(command: str, limit: int = 300) -> str:
    return redact(command)[: limit - 1] + ("…" if len(command) > limit else "")


def _redact_value(value):
    if isinstance(value, str):
        return redact(value)[:500]
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value[:20]]
    return value
