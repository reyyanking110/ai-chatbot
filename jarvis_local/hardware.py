"""Hardware detection and local model recommendations.

Everything here is best-effort and read-only: we look at RAM, CPU and GPU
information that the operating system already exposes.  No data leaves the
machine.  The resulting profile is used by ``jarvis init`` to pick model
sizes that actually fit, with headroom for the operating system and context
windows.
"""

from __future__ import annotations

import json
import math
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field

from .errors import JarvisError


@dataclass
class SystemProfile:
    os_name: str
    os_version: str
    arch: str
    cpu: str
    cpu_cores: int
    ram_gb: float
    gpu: str | None = None
    vram_gb: float | None = None
    unified_memory: bool = False
    accelerator: str = "cpu"  # cpu | cuda | rocm | metal | vulkan
    is_laptop: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def compute_label(self) -> str:
        if self.accelerator == "metal":
            return f"Apple Silicon unified memory ({self.ram_gb:.0f} GB)"
        if self.gpu and self.vram_gb:
            return f"{self.gpu} ({self.vram_gb:.0f} GB VRAM)"
        if self.gpu:
            return self.gpu
        return f"CPU only ({self.cpu_cores} cores, {self.ram_gb:.0f} GB RAM)"

    def budget_gb(self, override: float | None = None) -> float:
        """How much memory a model may realistically use."""
        if override and override > 0:
            return float(override)
        if self.unified_memory:
            # Leave room for the OS, the terminal, a browser and the editor.
            return max(2.0, self.ram_gb * 0.68)
        if self.accelerator in {"cuda", "rocm"} and self.vram_gb:
            # Slight overhead for the KV cache; Ollama spills to RAM but slowly.
            return max(2.0, self.vram_gb * 0.9)
        if self.accelerator == "vulkan" and self.vram_gb:
            return max(2.0, self.vram_gb * 0.8)
        return max(2.0, self.ram_gb * 0.55)

    def summary(self) -> str:
        parts = [
            f"{self.os_name} {self.os_version} · {self.arch}",
            f"{self.cpu_cores} CPU cores · {self.ram_gb:.0f} GB RAM",
            self.compute_label,
        ]
        return "\n".join(parts)


def _run(cmd: list[str], timeout: float = 4.0) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass

    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out and out.isdigit():
            return round(int(out) / 1024**3, 1)
    if system == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(re.findall(r"\d+", line)[0])
                        return round(kb / 1024**2, 1)
        except OSError:
            pass
    if system == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return round(stat.ullTotalPhys / 1024**3, 1)
        except Exception:
            pass
    return 8.0


def _cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith(("model name", "hardware")):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if system == "Windows":
        out = _run(["wmic", "cpu", "get", "name"])
        if out:
            lines = [x.strip() for x in out.splitlines() if x.strip() and x.strip() != "Name"]
            if lines:
                return lines[0]
    return platform.processor() or platform.machine() or "unknown CPU"


def _nvidia() -> tuple[str | None, float | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if not out:
        return None, None
    first = out.splitlines()[0]
    try:
        name, mem = [part.strip() for part in first.rsplit(",", 1)]
        return name, round(float(mem) / 1024, 1)
    except (ValueError, IndexError):
        return (first or None), None


def _apple_gpu() -> str | None:
    out = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=8)
    if not out:
        return None
    try:
        data = json.loads(out)
        items = data.get("SPDisplaysDataType") or []
        if items:
            return items[0].get("sppci_model") or items[0].get("_name")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def _linux_amd_gpu() -> str | None:
    out = _run(["lspci"])
    if not out:
        return None
    for line in out.splitlines():
        if re.search(r"(VGA|3D|Display).*(AMD|ATI|Radeon)", line, re.I):
            return line.split(":", 2)[-1].strip()
    return None


def _windows_gpu() -> tuple[str | None, float | None]:
    out = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json",
        ]
    )
    if not out:
        return None, None
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        if data:
            name = data[0].get("Name")
            ram = data[0].get("AdapterRAM") or 0
            vram = round(ram / 1024**3, 1) if ram else None
            return name, vram
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None, None


def _ensure_ollama_hint(profile: SystemProfile) -> None:
    if not shutil.which("ollama"):
        profile.notes.append(
            "Ollama is not installed. Install it from https://ollama.com/download, "
            "then run `jarvis doctor`."
        )


def detect_hardware() -> SystemProfile:
    system = platform.system()
    machine = platform.machine().lower()
    cores = 0
    try:
        import os

        cores = os.cpu_count() or 0
    except Exception:
        pass
    cores = cores or 1
    ram = _ram_gb()
    is_apple = system == "Darwin" and machine in {"arm64", "aarch64"}

    profile = SystemProfile(
        os_name=system,
        os_version=platform.release(),
        arch=platform.machine() or "unknown",
        cpu=_cpu_name(),
        cpu_cores=cores,
        ram_gb=ram,
        unified_memory=is_apple,
        is_laptop=False,
    )

    if is_apple:
        profile.gpu = _apple_gpu() or "Apple GPU"
        profile.vram_gb = ram  # shared
        profile.accelerator = "metal"
        profile.notes.append(
            "Apple Silicon shares memory between CPU and GPU; the model budget "
            "reserves about a third of RAM for macOS and your apps."
        )
    else:
        name, vram = _nvidia()
        if name:
            profile.gpu, profile.vram_gb = name, vram
            profile.accelerator = "cuda"
            if shutil.which("rocminfo"):
                pass
        else:
            amd = _linux_amd_gpu() if system == "Linux" else None
            win_gpu, win_vram = _windows_gpu() if system == "Windows" else (None, None)
            gpu = amd or win_gpu
            if gpu:
                profile.gpu = gpu
                profile.vram_gb = win_vram
                if shutil.which("rocm-smi") or shutil.which("rocminfo"):
                    profile.accelerator = "rocm"
                else:
                    profile.accelerator = "vulkan" if win_gpu else "cpu"
            else:
                profile.accelerator = "cpu"

        if profile.accelerator == "cpu":
            profile.notes.append(
                "No compatible GPU detected, so Jarvis will recommend smaller models "
                "that stay comfortable on CPU. Ollama still uses all cores."
            )
        elif profile.accelerator == "vulkan" and not profile.vram_gb:
            profile.notes.append(
                "A GPU was found but its VRAM could not be read; recommendations assume "
                "a 6 GB budget. Set hardware.budget_gb to override."
            )
            profile.vram_gb = 6.0

    if system == "Darwin":
        battery = _run(["pmset", "-g", "batt"])
        profile.is_laptop = bool(battery and "InternalBattery" in battery)
    elif system == "Linux":
        try:
            import glob

            profile.is_laptop = bool(glob.glob("/sys/class/power_supply/BAT*"))
        except Exception:
            pass
    if profile.is_laptop:
        profile.notes.append(
            "Laptop detected: sustained model use drains battery and heats the machine. "
            "Smaller models are gentler; `jarvis models` can switch any time."
        )

    _ensure_ollama_hint(profile)
    return profile


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCard:
    tag: str
    params: str
    size_gb: float
    roles: tuple[str, ...]
    note: str = ""

    @property
    def short(self) -> str:
        return f"{self.tag} ({self.params}, ~{self.size_gb:g} GB)"


CATALOG: tuple[ModelCard, ...] = (
    ModelCard("qwen3:0.6b", "0.6B", 0.5, ("fast",), "Tiny router/summariser; instant on CPU."),
    ModelCard("qwen3:1.7b", "1.7B", 1.4, ("fast", "general"), "Snappy generalist for weak machines."),
    ModelCard("llama3.2:3b", "3B", 2.0, ("fast", "general"), "Dependable tiny model with 128k context."),
    ModelCard("qwen3:4b", "4B", 2.6, ("fast", "general", "coding"), "Strong reasoning per gigabyte."),
    ModelCard("gemma3:4b", "4B", 3.3, ("general", "vision"), "Multimodal entry model."),
    ModelCard("qwen2.5-coder:1.5b", "1.5B", 1.0, ("coding",), "Autocomplete-scale code model."),
    ModelCard("qwen2.5-coder:7b", "7B", 4.7, ("coding",), "Solid FIM/code editing at 8 GB."),
    ModelCard("qwen3:8b", "8B", 5.2, ("general", "coding"), "The 8 GB-class all-rounder."),
    ModelCard("gemma3:12b", "12B", 8.1, ("general", "vision"), "Excellent writing and vision at 12 GB."),
    ModelCard("qwen2.5-coder:14b", "14B", 9.0, ("coding",), "Big step up for agentic coding."),
    ModelCard("qwen3:14b", "14B", 9.3, ("general",), "Dense 14B with strong tool use."),
    ModelCard("qwen3:30b-a3b", "30B MoE / 3B active", 18.0, ("general", "coding"), "MoE speed with 30B quality."),
    ModelCard("gemma3:27b", "27B", 17.0, ("general", "vision"), "Best dense writer that fits 24 GB."),
    ModelCard("qwen2.5-coder:32b", "32B", 20.0, ("coding",), "Top open coder for 24 GB GPUs."),
    ModelCard("qwen3-coder:30b", "30B MoE", 19.0, ("coding",), "Agentic coding, 256k context, fast MoE."),
    ModelCard("qwen3:32b", "32B", 20.0, ("general",), "GPT-4-class everyday reasoning on one GPU."),
    ModelCard("llama3.3:70b", "70B", 43.0, ("general",), "Flagship dense model for 48 GB+."),
    ModelCard("nomic-embed-text", "137M", 0.27, ("embed",), "Default 768-dim embedding model."),
    ModelCard("mxbai-embed-large", "334M", 0.67, ("embed",), "Higher-quality 1024-dim embeddings."),
)

EMBED_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
}


@dataclass
class ModelPlan:
    general: str
    coding: str
    fast: str
    embed: str
    num_ctx: int
    budget_gb: float
    rationale: str
    notes: list[str] = field(default_factory=list)

    @property
    def tags(self) -> list[str]:
        return [self.general, self.coding, self.fast, self.embed]

    def download_gb(self) -> float:
        return round(sum(card.size_gb for card in CATALOG if card.tag in self.tags), 1)


def card_for(tag: str) -> ModelCard | None:
    for card in CATALOG:
        if card.tag == tag:
            return card
    return None


def _best(role: str, budget: float) -> ModelCard | None:
    fits = [c for c in CATALOG if role in c.roles and c.size_gb <= budget]
    return max(fits, key=lambda c: c.size_gb) if fits else None


def recommend_models(budget_gb: float, ram_gb: float = 16.0) -> ModelPlan:
    """Pick the strongest models that fit the memory budget."""
    budget = max(1.0, float(budget_gb))

    general = _best("general", budget)
    coding = _best("coding", budget)
    fast = _best("fast", min(budget, 4.0)) or next(c for c in CATALOG if c.tag == "qwen3:0.6b")
    embed = card_for("nomic-embed-text")

    notes: list[str] = []
    if general is None:
        general = card_for("qwen3:0.6b")
        notes.append("The memory budget is very small; using a 0.6B model. Close apps or add RAM.")
    if coding is None:
        coding = card_for("qwen3:0.6b")
        notes.append("No code model fits the current budget; the tiny model handles simple edits.")
    if budget >= 30:
        num_ctx = 32768
    elif budget >= 15:
        num_ctx = 16384
    elif budget >= 7:
        num_ctx = 8192
    else:
        num_ctx = 4096

    if general.size_gb > budget * 0.85:
        notes.append(
            f"{general.tag} uses most of the estimated budget; generation will be slower "
            f"while other apps are open."
        )
    if ram_gb < 8:
        notes.append("Under 8 GB system RAM: expect small models only, and keep other apps closed.")

    rationale = (
        f"≈{budget:.0f} GB usable model memory → {general.tag} for general work, "
        f"{coding.tag} for code, {fast.tag} for quick routing/summaries, "
        f"and {embed.tag} for local embeddings."
    )
    return ModelPlan(
        general=general.tag,
        coding=coding.tag,
        fast=fast.tag,
        embed=embed.tag,
        num_ctx=num_ctx,
        budget_gb=round(budget, 1),
        rationale=rationale,
        notes=notes,
    )


def plan_from_config(cfg) -> ModelPlan:
    """Build the plan the config effectively uses (explicit models win)."""
    if cfg.hardware.budget_gb:
        budget = cfg.hardware.budget_gb
    else:
        try:
            budget = detect_hardware().budget_gb()
        except JarvisError:
            budget = 8.0
    base = recommend_models(budget)
    plan = ModelPlan(
        general=cfg.models.general or base.general,
        coding=cfg.models.coding or base.coding,
        fast=cfg.models.fast or base.fast,
        embed=cfg.models.embed or base.embed,
        num_ctx=cfg.models.num_ctx or base.num_ctx,
        budget_gb=round(budget, 1),
        rationale=base.rationale,
        notes=list(base.notes),
    )
    return plan


def bytes_to_gb(value: int) -> float:
    return round(value / 1024**3, 1)


def human_gb(value: float) -> str:
    return f"{value:.1f} GB" if not math.isclose(value, round(value), abs_tol=0.05) else f"{value:.0f} GB"
