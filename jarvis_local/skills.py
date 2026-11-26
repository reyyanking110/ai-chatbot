"""Skills: small, local extensions Jarvis can install and learn over time.

A skill is a folder with a ``SKILL.md`` file:

```
---
name: weather
description: Local weather for a city using Open-Meteo (no API key)
version: 1.0.0
author: you
run: scripts/weather.py     # optional
triggers: [weather, forecast]
---

Instructions for the assistant. When the user asks about the weather, ...
```

The body is injected into the chat system prompt when the skill is relevant,
and ``run`` (if present) is a script/command the assistant can invoke through
an auto-generated ``skill_<name>`` tool.

Safety model — because "install from the internet" is a big deal:

* Installing from a URL requires confirmation. GitHub URLs work out of the box.
* Before enabling, Jarvis prints a **security scan**: scripts found, network
  imports, credential-looking strings, suspicious shell calls.
* Nothing is auto-run. Scripts execute only when the model calls the tool, and
  every such call goes through the same approval path as ``run_shell``.
* Skills live in your data directory, not on ``sys.path``; they cannot import
  Jarvis internals or monkeypatch the app.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import Config
from .errors import SkillError
from .tools.base import Tool, ToolResult, arg, obj
from .util import run_capture

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)
NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,40}$")
NETWORK_HINTS = re.compile(r"\b(requests|httpx|urllib|urllib3|socket|aiohttp|http\.client)\b")
CRED_HINTS = re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|BEGIN [A-Z ]*PRIVATE KEY)\b")
SHELL_HINTS = re.compile(r"(?i)\b(os\.system|subprocess|popen|exec\s*\(|eval\s*\(|shutil\.rmtree|rm\s+-rf)\b")
SCRIPTS = {".py", ".sh", ".bash", ".zsh", ".js", ".mjs", ".rb", ".pl"}

MAX_SKILL_FILES = 200            # files in one installed skill folder
MAX_ARCHIVE_FILES = 6000         # files in a repo tarball (collections are big)
MAX_SKILL_BYTES = 32 * 1024 * 1024
MAX_UNPACKED_BYTES = 160 * 1024 * 1024


@dataclass
class SecurityReport:
    scripts: list[str] = field(default_factory=list)
    network: list[str] = field(default_factory=list)
    credentials: list[str] = field(default_factory=list)
    shell: list[str] = field(default_factory=list)
    executables: list[str] = field(default_factory=list)

    @property
    def risk(self) -> str:
        if self.executables or self.credentials:
            return "high"
        if self.shell or self.network:
            return "medium"
        return "low"

    def lines(self) -> list[str]:
        out = [f"Overall risk: {self.risk.upper()}"]
        if self.executables:
            out.append("Executable binaries: " + ", ".join(self.executables))
        if self.scripts:
            out.append("Scripts: " + ", ".join(self.scripts))
        if self.network:
            out.append("Network access found in: " + ", ".join(self.network))
        if self.credentials:
            out.append("Credential-like strings in: " + ", ".join(self.credentials))
        if self.shell:
            out.append("Shell/subprocess calls in: " + ", ".join(self.shell))
        if self.risk == "low":
            out.append("No executable content detected — this skill is prompt-only instructions.")
        return out


@dataclass
class Skill:
    name: str
    description: str
    body: str
    version: str = "0.1.0"
    author: str = ""
    run: str = ""
    triggers: list[str] = field(default_factory=list)
    enabled: bool = True
    source: str = ""
    installed_at: str = ""
    path: Path | None = None
    security: SecurityReport = field(default_factory=SecurityReport)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "run": self.run,
            "triggers": self.triggers,
            "enabled": self.enabled,
            "source": self.source,
            "installed_at": self.installed_at or dt.datetime.now().isoformat(timespec="seconds"),
            "security": {
                "risk": self.security.risk,
                "scripts": self.security.scripts,
                "network": self.security.network,
                "credentials": self.security.credentials,
                "shell": self.security.shell,
            },
        }

    def prompt_block(self) -> str:
        trigger = f" (triggers: {', '.join(self.triggers)})" if self.triggers else ""
        return f"### Skill: {self.name}{trigger}\n{self.description}\n{self.body.strip()[:4000]}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_skill_markdown(text: str, fallback_name: str = "") -> Skill:
    match = FRONTMATTER.match(text)
    if not match:
        raise SkillError(
            "SKILL.md needs YAML front matter between --- lines.",
            hint="Start the file with:\n---\nname: my-skill\ndescription: what it does\n---",
        )
    raw_meta, body = match.group(1), match.group(2)
    meta = _parse_simple_yaml(raw_meta)
    name = str(meta.get("name") or fallback_name or "").strip().lower()
    if not name:
        raise SkillError("SKILL.md is missing a 'name'.")
    if not NAME_RE.match(name):
        raise SkillError(
            f"Invalid skill name '{name}'. Use lowercase letters, digits and dashes (e.g. 'weather')."
        )
    description = str(meta.get("description") or "").strip()
    if not description:
        raise SkillError(f"Skill '{name}' is missing a 'description'.")
    triggers = meta.get("triggers") or []
    if isinstance(triggers, str):
        triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        version=str(meta.get("version") or "0.1.0"),
        author=str(meta.get("author") or ""),
        run=str(meta.get("run") or ""),
        triggers=[str(t) for t in triggers],
    )


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by SKILL.md without a PyYAML dependency."""
    data: dict[str, Any] = {}
    current_key = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^\s*-\s+", line) and current_key:
            data.setdefault(current_key, [])
            if isinstance(data[current_key], list):
                data[current_key].append(_unquote(line.split("-", 1)[1].strip()))
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            current_key = key
            if not value:
                data[key] = []
            elif value.startswith("[") and value.endswith("]"):
                data[key] = [_unquote(v.strip()) for v in value[1:-1].split(",") if v.strip()]
            else:
                data[key] = _unquote(value)
    return data


def _unquote(value: str) -> str:
    return value.strip().strip('"').strip("'")


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_folder(folder: Path) -> SecurityReport:
    report = SecurityReport()
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(folder))
        if path.suffix.lower() in SCRIPTS:
            report.scripts.append(rel)
        if path.suffix in {".exe", ".bin", ".so", ".dylib", ".dll", ".wasm"} or (path.stat().st_mode & 0o111 and path.suffix == ""):
            report.executables.append(rel)
        if path.suffix.lower() in SCRIPTS | {".md", ".txt", ".json", ".yaml", ".yml"}:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:200_000]
            except OSError:
                continue
            if NETWORK_HINTS.search(text):
                report.network.append(rel)
            if CRED_HINTS.search(text):
                report.credentials.append(rel)
            if SHELL_HINTS.search(text):
                report.shell.append(rel)
    report.scripts = sorted(set(report.scripts))
    report.network = sorted(set(report.network))
    report.credentials = sorted(set(report.credentials))
    report.shell = sorted(set(report.shell))
    report.executables = sorted(set(report.executables))
    return report


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SkillManager:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.dir = cfg.skills_dir
        self.index_path = cfg.skills_index
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- inventory
    def list(self, include_disabled: bool = True) -> list[Skill]:
        skills: list[Skill] = []
        if not self.dir.exists():
            return skills
        meta = self._load_index()
        for folder in sorted(self.dir.iterdir()):
            if not folder.is_dir():
                continue
            skill_file = folder / "SKILL.md"
            if not skill_file.exists():
                continue
            try:
                skill = parse_skill_markdown(skill_file.read_text(encoding="utf-8"), folder.name)
            except (SkillError, OSError):
                continue
            skill.path = folder
            skill.source = meta.get(skill.name, {}).get("source", "")
            skill.installed_at = meta.get(skill.name, {}).get("installed_at", "")
            skill.enabled = meta.get(skill.name, {}).get("enabled", True)
            skill.security = scan_folder(folder)
            if include_disabled or skill.enabled:
                skills.append(skill)
        return skills

    def get(self, name: str) -> Skill | None:
        for skill in self.list():
            if skill.name == name:
                return skill
        return None

    def enabled(self) -> list[Skill]:
        return [s for s in self.list() if s.enabled]

    # -------------------------------------------------------------- install
    def install(
        self,
        source: str,
        *,
        confirm=None,
        force: bool = False,
        expected_sha256: str = "",
    ) -> Skill:
        """Install from a local folder or an http(s) URL (folder / zip / tar)."""
        original_source = source
        base_source, selector = (source.split("#", 1) + [""])[:2] if "#" in source else (source, "")
        if selector:
            source = base_source  # keep the local/remote logic on the real path
        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            payload = self._download(source, expected_sha256=expected_sha256)
            skill, report = self._extract(payload, original_source)
        else:
            folder = Path(source).expanduser()
            if not folder.exists():
                raise SkillError(
                    f"No such skill folder: {source}",
                    hint="Pass a local folder with SKILL.md, or a GitHub URL.",
                )
            if folder.is_file():
                payload = {"filename": folder.name, "data": folder.read_bytes()}
                skill, report = self._extract(payload, original_source)
            else:
                if selector:
                    raise SkillError(
                        "A #skill selector only applies to archives and collections.",
                        hint="Pass the skill folder directly, e.g. `jarvis skills install ./my-skill`.",
                    )
                _check_skill_size(folder)
                skill = parse_skill_markdown((folder / "SKILL.md").read_text(encoding="utf-8"), folder.name)
                report = scan_folder(folder)
                skill.security = report
                skill.path = folder

        if confirm is not None:
            approved = confirm(skill, report)
            if not approved:
                raise SkillError(f"Install of '{skill.name}' cancelled.")

        target = self.dir / skill.name
        if target.exists():
            if not force:
                raise SkillError(
                    f"Skill '{skill.name}' is already installed.",
                    hint="Use --force to replace it, or `jarvis skills remove` first.",
                )
            shutil.rmtree(target)
        if skill.path and skill.path.exists():
            shutil.copytree(skill.path, target)
        else:  # pragma: no cover - payload extraction path is handled below
            raise SkillError("Internal error: skill payload missing.")
        # Re-scan the installed copy (source folder may have changed).
        installed = self.get(skill.name)
        if installed is None:
            installed = parse_skill_markdown((target / "SKILL.md").read_text(encoding="utf-8"), skill.name)
            installed.path = target
        installed.security = scan_folder(target)
        meta = self._load_index()
        record = installed.to_dict()
        record["source"] = source
        record["installed_at"] = dt.datetime.now().isoformat(timespec="seconds")
        meta[skill.name] = record
        self._save_index(meta)
        return installed

    def _download(self, url: str, expected_sha256: str = "") -> dict[str, Any]:
        import hashlib

        url = _normalise_github_url(url)
        try:
            resp = _http_get(url)
        except httpx.HTTPError as exc:
            raise SkillError(
                f"Could not download skill: {exc.__class__.__name__}: {exc}",
                hint="Check the URL and your connection. Skills can also be installed from a local folder. "
                     "On a corporate machine with a TLS inspecting proxy, set SSL_CERT_FILE to your CA bundle.",
            ) from exc
        if resp.status_code >= 400:
            raise SkillError(
                f"Download failed (HTTP {resp.status_code}): {url}",
                hint="For GitHub, use the repo URL — Jarvis rewrites it to a tarball.",
            )
        data = resp.content
        if len(data) > MAX_SKILL_BYTES:
            raise SkillError(
                f"Archive is {len(data) / 1024 / 1024:.1f} MB; the limit is 32 MB.",
                hint="Install a single skill folder (`repo#skill-name`) rather than a huge collection.",
            )
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 and digest != expected_sha256.strip().lower():
            raise SkillError(
                "Checksum mismatch — the downloaded file is not what you expected.",
                hint=f"got {digest}",
            )
        filename = Path(urlparse(str(resp.url)).path).name or "skill.tar.gz"
        return {"filename": filename, "data": data, "sha256": digest, "url": url}

    def _extract(self, payload: dict[str, Any], source: str) -> tuple[Skill, SecurityReport]:
        staging = self.cfg.data_root / ".skill-staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            self._unpack(payload, staging)
            roots = _find_skill_roots(staging)
            root = _select_skill_root(roots, source, staging)
            _check_skill_size(root)
            skill = parse_skill_markdown((root / "SKILL.md").read_text(encoding="utf-8"), root.name)
            report = scan_folder(root)
            skill.security = report
            skill.path = root
            skill.source = source
            return skill, report
        except (zipfile.BadZipFile, tarfile.TarError) as exc:
            raise SkillError(f"Could not read the archive: {exc}") from exc

    def _unpack(self, payload: dict[str, Any], staging: Path) -> None:
        data = payload["data"]
        filename = payload.get("filename", "")
        if filename.endswith(".zip") or data[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = [m for m in archive.infolist() if not m.is_dir()]
                if len(members) > MAX_ARCHIVE_FILES:
                    raise SkillError(f"Archive has too many files ({len(members)}).")
                if sum(m.file_size for m in members) > MAX_UNPACKED_BYTES:
                    raise SkillError("Archive expands beyond the 160 MB safety limit.")
                for member in members:
                    _safe_extract_member(staging, member.filename)
                for member in members:
                    target = staging / member.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        elif filename.endswith((".tar.gz", ".tgz", ".tar")) or data[:2] == b"\x1f\x8b":
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
                members = [m for m in archive.getmembers() if m.isfile()]
                if len(members) > MAX_ARCHIVE_FILES:
                    raise SkillError(f"Archive has too many files ({len(members)}).")
                if sum(m.size for m in members) > MAX_UNPACKED_BYTES:
                    raise SkillError("Archive expands beyond the 160 MB safety limit.")
                for member in members:
                    _safe_extract_member(staging, member.name)
                _extract_tar_files(archive, members, staging)
        else:
            text = data.decode("utf-8", errors="replace")
            (staging / "SKILL.md").write_text(text, encoding="utf-8")

    def list_remote(self, source: str) -> list[dict[str, str]]:
        """List the skills inside a repo/archive without installing anything."""
        parsed = urlparse(source)
        if parsed.scheme not in ("http", "https"):
            folder = Path(source).expanduser()
            if not folder.exists():
                raise SkillError(f"No such skill folder: {source}")
            if folder.is_dir():
                found = sorted(
                    {p.parent for p in folder.rglob("SKILL.md")},
                    key=lambda p: (len(p.parts), str(p)),
                )
                if (folder / "SKILL.md").exists() and folder.resolve() in {f.resolve() for f in found}:
                    roots = [folder]
                elif found:
                    roots = found
                else:
                    raise SkillError(
                        f"No SKILL.md in {folder}.",
                        hint="A skill folder needs SKILL.md at its root (see `jarvis skills new`).",
                    )
            else:
                # A local archive: inspect it like a downloaded one.
                payload = {"filename": folder.name, "data": folder.read_bytes()}
                staging = self.cfg.data_root / ".skill-staging-list"
                if staging.exists():
                    shutil.rmtree(staging)
                staging.mkdir(parents=True, exist_ok=True)
                try:
                    self._unpack(payload, staging)
                    found = _find_skill_roots(staging)
                    name = folder.name
                    return [
                        {**_remote_entry(root, staging), "install_as": f"{name}#{root.name}", "name": root.name}
                        for root in found
                    ]
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
        payload = self._download(source)
        staging = self.cfg.data_root / ".skill-staging-list"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            self._unpack(payload, staging)
            roots = _find_skill_roots(staging)
            return [_remote_entry(root, staging, source) for root in roots]
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # --------------------------------------------------------------- mutate
    def set_enabled(self, name: str, enabled: bool) -> None:
        meta = self._load_index()
        if name not in meta and self.get(name) is None:
            raise SkillError(f"Skill '{name}' is not installed.")
        record = meta.get(name, {})
        record["enabled"] = enabled
        meta[name] = record
        self._save_index(meta)

    def remove(self, name: str) -> None:
        target = self.dir / name
        if not target.exists():
            raise SkillError(f"Skill '{name}' is not installed.")
        shutil.rmtree(target)
        meta = self._load_index()
        meta.pop(name, None)
        self._save_index(meta)

    def scaffold(self, name: str, description: str = "") -> Path:
        """Create a ready-to-edit skill template for the user."""
        if not NAME_RE.match(name):
            raise SkillError("Skill names must be lowercase letters, digits and dashes.")
        folder = self.dir / name
        if folder.exists():
            raise SkillError(f"Skill '{name}' already exists at {folder}.")
        (folder / "scripts").mkdir(parents=True)
        skill_md = f"""---
name: {name}
description: {description or "Describe what this skill helps with."}
version: 0.1.0
author: you
triggers: [{name}]
# run: scripts/run.py        # optional: exposes a skill_{name} tool
---

Write instructions for the assistant here. Explain what to do when the user
asks something related to {name}: which tools to use, in what order, and what
the output should look like. Keep it short and specific.
"""
        (folder / "SKILL.md").write_text(skill_md, encoding="utf-8")
        (folder / "scripts" / "run.py").write_text(
            '"""Optional entry point.\n\n'
            "Read the request from argv or stdin and print the result to stdout.\n"
            '"""\n\nimport sys\n\nrequest = " ".join(sys.argv[1:]) or sys.stdin.read()\n'
            f'print(f"{name} got: {{request}}")\n',
            encoding="utf-8",
        )
        meta = self._load_index()
        meta[name] = {"enabled": True, "source": "local", "installed_at": dt.datetime.now().isoformat(timespec="seconds")}
        self._save_index(meta)
        return folder

    # ---------------------------------------------------------------- prompt
    def prompt_hints(self, user_input: str, agent: str = "chat") -> str:
        if not self.cfg.skills.enabled:
            return ""
        allowed = self.cfg.skills.inject_agents
        if allowed and agent not in allowed:
            return ""
        text = user_input.lower()
        hints = []
        for skill in self.enabled():
            triggers = [t.lower() for t in skill.triggers] + [skill.name.lower()]
            if not skill.triggers:
                # No triggers declared: always available as reference material.
                hints.append(skill.prompt_block())
            elif any(_trigger_matches(trigger, text) for trigger in triggers if trigger):
                hints.append(skill.prompt_block())
        return "\n\n".join(hints[:4])

    # ----------------------------------------------------------------- tools
    def skill_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        for skill in self.enabled():
            if not skill.run or not skill.path:
                continue
            tools.append(self._make_tool(skill))
        return tools

    def _make_tool(self, skill: Skill) -> Tool:
        name = "skill_" + skill.name.replace("-", "_")

        def runner(ctx, request: str = "") -> ToolResult:
            return run_skill_script(ctx, skill, request)

        return Tool(
            name=name,
            description=(
                f"[skill:{skill.name}] {skill.description} Run the skill's local script with a "
                "plain-language request."
            ),
            parameters=obj(
                {"request": arg("What the user wants; passed to the script.", "string")},
            ),
            func=runner,
            category="skills",
            needs_approval=True,
        )

    # ----------------------------------------------------------------- usage
    def record_use(self, name: str) -> None:
        meta = self._load_index()
        record = meta.setdefault(name, {"enabled": True})
        record["uses"] = int(record.get("uses", 0)) + 1
        record["last_used"] = dt.datetime.now().isoformat(timespec="seconds")
        meta[name] = record
        self._save_index(meta)

    def usage(self) -> dict[str, dict[str, Any]]:
        return self._load_index()

    def suggest_lessons(self, limit: int = 40) -> list[str]:
        """Look for repeated friction in the audit log and suggest skills.

        This is deliberately simple and local: if the same tool + first
        argument keep appearing, a reusable skill probably pays off. Only
        counts are used; no message text leaves the audit file.
        """
        path = self.cfg.log_dir / "audit.jsonl"
        if not path.exists():
            return []
        counts: dict[str, int] = {}
        errors: dict[str, int] = {}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-2000:]
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = str(entry.get("action", ""))
            args = entry.get("args") or {}
            if action.startswith("tool:"):
                key = _lesson_key(action[5:], args)
                if key:
                    counts[key] = counts.get(key, 0) + 1
                if entry.get("ok") is False:
                    errors[key or action] = errors.get(key or action, 0) + 1
        suggestions = []
        for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]:
            if count >= 3:
                draft = _skill_draft_name(key)
                suggestions.append(
                    f"{key} ran {count} times — consider a skill that wraps it: "
                    f"`jarvis skills new {draft}`"
                )
        for key, count in sorted(errors.items(), key=lambda item: item[1], reverse=True)[:5]:
            if count >= 3 and "ran" not in "".join(suggestions):
                suggestions.append(f"{key} failed {count} times — a skill could pre-check the common failure.")
        return suggestions

    # ------------------------------------------------------------------ index
    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_index(self, data: dict[str, Any]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _skill_draft_name(key: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
    words = [w for w in slug.split("-") if w and w not in {"on", "ran", "times", "files"}]
    name = "-".join(words[:3]) or "helper"
    if not NAME_RE.match(name):
        name = "helper-" + name[:24].strip("-")
    return name


def _lesson_key(tool: str, args: dict[str, Any]) -> str:
    """A stable, non-sensitive description of a repeated action."""
    if tool in {"run_shell", "run_python"}:
        return ""
    if tool in {"write_file", "read_file", "append_file", "delete_file"}:
        path = str(args.get("path", ""))
        return f"{tool} on {Path(path).suffix or 'files'}" if path else tool
    if tool == "search_library":
        return "library searches"
    if tool.startswith("skill_"):
        return ""
    return tool if tool else ""


def _trigger_matches(trigger: str, text: str) -> bool:
    """A trigger matches as a substring, or when all its words appear (any order)."""
    trigger = trigger.lower().strip()
    if not trigger:
        return False
    if trigger in text:
        return True
    words = [w for w in re.split(r"[^a-z0-9]+", trigger) if w]
    if not words:
        return False
    # Very short words like "in" or "my" match too loosely; require a word
    # boundary for them instead of a bare substring test.
    for word in words:
        if len(word) <= 3:
            if not re.search(rf"\b{re.escape(word)}\b", text):
                return False
        elif word not in text:
            return False
    return True


def run_skill_script(ctx, skill: Skill, request: str) -> ToolResult:
    if not skill.path or not skill.run:
        return ToolResult("This skill has no runnable script.", ok=False)
    script = (skill.path / skill.run).resolve()
    if not str(script).startswith(str(skill.path.resolve())):
        return ToolResult("Skill script path escapes the skill folder — refusing.", ok=False)
    if not script.exists():
        return ToolResult(f"Skill script not found: {skill.run}", ok=False)
    if ctx.cfg.security.approval != "auto":
        approved = ctx.approve(
            f"skill:{skill.name}",
            f"Run skill '{skill.name}' script `{skill.run}`\nwith request: {request[:300]!r}?",
        )
        if not approved:
            return ToolResult("The user declined to run this skill.", ok=False)
    if script.suffix == ".py":
        cmd = ["python3", str(script), request]
    elif script.suffix in {".sh", ".bash", ".zsh"}:
        cmd = ["/bin/sh", str(script), request]
    elif script.suffix in {".js", ".mjs"}:
        cmd = ["node", str(script), request]
    else:
        cmd = [str(script), request]
    code, stdout, stderr = run_capture(
        cmd,
        cwd=skill.path,
        timeout=min(ctx.cfg.security.shell_timeout_s * 2, 300),
        env=ctx.policy.scrubbed_env(),
    )
    ctx.policy.audit(f"skill:{skill.name}", "run", {"request": request[:300], "exit": code})
    manager = getattr(ctx, "skill_manager", None)
    if manager is not None:
        try:
            manager.record_use(skill.name)
        except (OSError, ValueError):
            pass
    limit = ctx.cfg.security.max_output_chars
    parts = [f"skill {skill.name} exit code: {code}"]
    if stdout:
        parts.append(stdout[:limit])
    if stderr:
        parts.append("stderr:\n" + stderr[: limit // 2])
    return ToolResult("\n".join(parts), ok=code == 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http_get(url: str) -> httpx.Response:
    """Download a URL, falling back to the system CA bundle if Python's is thin."""
    headers = {"User-Agent": "jarvis-local/0.1"}
    try:
        with httpx.Client(timeout=30, follow_redirects=True, trust_env=True) as client:
            return client.get(url, headers=headers)
    except httpx.ConnectError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc).upper():
            raise
        bundle = "/etc/ssl/certs/ca-certificates.crt"
        if not Path(bundle).exists():
            raise
        # Some minimal Linux images ship a Python without a usable CA store.
        with httpx.Client(
            timeout=30, follow_redirects=True, trust_env=True, verify=bundle
        ) as client:
            return client.get(url, headers=headers)


def _normalise_github_url(url: str) -> str:
    """Turn github.com/owner/repo[/tree/branch/sub] into a codeload tarball URL."""
    parsed = urlparse(url)
    if parsed.hostname not in {"github.com", "www.github.com"}:
        return url
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return url
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        branch = parts[3]
        # codeload serves a tarball of the whole branch; the skill root is
        # discovered from SKILL.md inside the archive.
        return f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{branch}"
    return f"https://codeload.github.com/{owner}/{repo}/tar.gz/HEAD"


def _safe_extract_member(root: Path, member_name: str) -> None:
    target = (root / member_name).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise SkillError(f"Archive contains an unsafe path: {member_name}")


def _check_skill_size(root: Path) -> None:
    files = [p for p in root.rglob("*") if p.is_file()]
    if len(files) > MAX_SKILL_FILES:
        raise SkillError(
            f"This skill folder has {len(files)} files; the limit is {MAX_SKILL_FILES}.",
            hint="Pick a smaller skill, or tidy the folder before installing.",
        )
    total = sum(p.stat().st_size for p in files)
    if total > MAX_SKILL_BYTES:
        raise SkillError(
            f"This skill folder is {total / 1024 / 1024:.1f} MB; the limit is 32 MB.",
            hint="Skills should contain instructions and small scripts, not datasets or media.",
        )


def _remote_entry(root: Path, base: Path, source: str = "") -> dict[str, str]:
    try:
        text = (root / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        meta = _parse_simple_yaml(FRONTMATTER.match(text).group(1)) if FRONTMATTER.match(text) else {}
    except (OSError, AttributeError):
        meta = {}
    rel = str(root.relative_to(base)) if root != base else "."
    return {
        "name": str(meta.get("name") or root.name),
        "path": rel,
        "description": str(meta.get("description") or "")[:160],
        "install_as": f"{source}#{root.name}" if source and rel != "." else (source or str(root)),
    }


def _extract_tar_files(archive: tarfile.TarFile, members: list[tarfile.TarInfo], staging: Path) -> None:
    """Extract only regular files (no links, no devices), paths already vetted."""
    for member in members:
        target = staging / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            continue
        with source, target.open("wb") as dst:
            shutil.copyfileobj(source, dst)


def _find_skill_roots(staging: Path) -> list[Path]:
    """Every folder in an archive that contains a SKILL.md, shallowest first."""
    candidates = [p.parent for p in staging.rglob("SKILL.md")]
    if not candidates:
        raise SkillError(
            "No SKILL.md found in the downloaded archive.",
            hint="A skill folder needs a SKILL.md at its root (see `jarvis skills new`).",
        )
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates


def _select_skill_root(roots: list[Path], source: str, staging: Path) -> Path:
    """Choose the intended skill when a repo holds a collection of them.

    Supports ``repo#sub/path`` to point at one skill inside a collection.
    """
    if "#" in source:
        wanted = source.split("#", 1)[1].strip().strip("/").lower()
        for root in roots:
            rel = str(root.relative_to(staging)).lower()
            if rel == wanted or rel.endswith("/" + wanted) or root.name.lower() == wanted:
                return root
        names = ", ".join(sorted(r.name for r in roots)[:12])
        raise SkillError(
            f"No skill matching '{wanted}' in this archive.",
            hint=f"Available: {names}{' …' if len(roots) > 12 else ''}",
        )
    if len(roots) > 1:
        # A collection: require an explicit choice rather than guessing.
        raise SkillError(
            f"This repository is a collection of {len(roots)} skills.",
            hint="Install a specific one with: jarvis skills install <repo> --list  then  "
                 "jarvis skills install <repo>#<skill-name>",
        )
    return roots[0]


def export_skill(skill: Skill, destination: Path) -> Path:
    """Bundle a skill into a zip you can share or re-install."""
    destination = destination.with_suffix(".zip") if destination.suffix != ".zip" else destination
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((skill.path or Path(".")).rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{skill.name}/{path.relative_to(skill.path)}")
    return destination


def default_skills() -> list[dict[str, str]]:
    """Starter skills written by `jarvis init` so day one feels alive."""
    return [
        {
            "name": "daily-journal",
            "description": "Capture a short daily journal entry with a mood and one highlight.",
            "triggers": ["journal", "diary", "log my day"],
            "body": (
                "When the user wants to journal, ask at most one clarifying question, then write "
                "a warm 4–8 sentence entry in their voice using `take_note`. Include date, mood, "
                "what happened, and one thing to carry forward. Never invent events."
            ),
        },
        {
            "name": "file-organiser",
            "description": "Tidy the workspace downloads folder into dated subfolders.",
            "triggers": ["tidy", "organise files", "clean up downloads"],
            "body": (
                "Use `list_files` to inspect the folder, propose a small plan, then move files with "
                "`run_shell` `mv` (each move needs approval). Never delete anything; ask first."
            ),
        },
        {
            "name": "meeting-notes",
            "description": "Turn messy meeting notes into decisions, owners and next steps.",
            "triggers": ["meeting notes", "summarise meeting", "action items"],
            "body": (
                "Given notes, produce: Decisions, Owners + Actions (table), Open questions, and "
                "a 3-line recap suitable for reading aloud. Mark anything you inferred."
            ),
        },
    ]


def install_default_skills(cfg: Config) -> list[str]:
    manager = SkillManager(cfg)
    created = []
    for spec in default_skills():
        target = manager.dir / spec["name"]
        if target.exists():
            continue
        target.mkdir(parents=True)
        triggers = ", ".join(spec["triggers"])
        text = (
            f"---\nname: {spec['name']}\ndescription: {spec['description']}\n"
            f"version: 1.0.0\nauthor: jarvis\ntriggers: [{triggers}]\n---\n\n{spec['body']}\n"
        )
        (target / "SKILL.md").write_text(text, encoding="utf-8")
        created.append(spec["name"])
    if created:
        manager._save_index({s.name: s.to_dict() for s in manager.list()})
    return created


__all__ = [
    "Skill",
    "SkillManager",
    "SecurityReport",
    "parse_skill_markdown",
    "scan_folder",
    "export_skill",
    "install_default_skills",
    "run_skill_script",
]
