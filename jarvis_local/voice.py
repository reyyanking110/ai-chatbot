"""Text-to-speech with whatever the machine already has.

Engines, in order of preference (``voice.engine = auto``):

* **say** — built into macOS, good voices, works offline.
* **piper** — a fast neural TTS; pass ``--model /path/to/voice.onnx``.
* **espeak / espeak-ng** — ubiquitous, robotic but dependable on Linux.
* Downloaded cloud TTS is never used: speech is synthesised locally.

Every call also writes a file to ``<data>/audio`` so you can replay it.
"""

from __future__ import annotations

import datetime as dt
import platform
import shutil
import subprocess
import threading
from pathlib import Path

from .config import Config
from .errors import JarvisError
from .tools.base import ToolResult
from .util import human_bytes, run_capture

DEFAULT_MAC_VOICES = ["Samantha", "Alex", "Daniel", "Karen", "Moira", "Tessa", "Fred"]


def available_engines(cfg: Config | None = None) -> list[str]:
    engines = []
    if shutil.which("say"):
        engines.append("say")
    if shutil.which("piper"):
        engines.append("piper")
    for name in ("espeak-ng", "espeak"):
        if shutil.which(name):
            engines.append("espeak")
            break
    return engines


class VoiceEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.audio_dir = cfg.data_root / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen | None = None
        self.last_file: Path | None = None

    # ------------------------------------------------------------------ setup
    def engine(self) -> str:
        wanted = (self.cfg.voice.engine or "auto").lower()
        present = available_engines()
        if wanted == "none":
            return "none"
        if wanted != "auto":
            if wanted == "espeak" and "espeak" in present:
                return "espeak"
            if wanted in present:
                return wanted
            raise JarvisError(
                f"Requested TTS engine '{wanted}' is not installed.",
                hint=f"Available here: {', '.join(present) or 'none'}. "
                "Install espeak-ng or piper, or set voice.engine = auto.",
            )
        return present[0] if present else "none"

    def describe(self) -> str:
        engine = self.engine()
        if engine == "none":
            return "No TTS engine found (macOS `say` or `espeak-ng` gives you a voice)."
        if engine == "say":
            return f"macOS say, voice {self._mac_voice() or 'system default'}"
        if engine == "piper":
            return f"Piper ({self.cfg.voice.piper_model or 'no model set'})"
        return "espeak-ng (robotic but offline)"

    def _mac_voice(self) -> str:
        return (self.cfg.voice.voice or "").strip()

    # --------------------------------------------------------------- speaking
    def speak(self, text: str, play: bool | None = None) -> Path | None:
        text = clean_for_speech(text)
        if not text:
            raise JarvisError("Nothing to speak: the text was empty after cleanup.")
        self._transcript = text
        engine = self.engine()
        if engine == "none":
            raise JarvisError(
                "No text-to-speech engine is available.",
                hint="macOS: nothing to do. Linux: `sudo apt install espeak-ng`. "
                "Or set voice.engine = none and read the transcript instead.",
            )
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        word = "_".join(text.split()[:4]).lower()
        word = "".join(ch for ch in word if ch.isalnum() or ch == "_")[:40] or "speech"
        should_play = self.cfg.voice.play if play is None else play

        if engine == "say":
            path = self.audio_dir / f"{stamp}-{word}.aiff"
            cmd = ["say", "-r", str(max(80, min(self.cfg.voice.rate, 400)))]
            if self._mac_voice():
                cmd += ["-v", self._mac_voice()]
            cmd += ["-o", str(path), text]
            code, _, err = run_capture(cmd, timeout=120)
            if code != 0:
                raise JarvisError(f"say failed: {err.strip()[:300]}")
            self.last_file = path
            if should_play:
                self._play_file(path)
            return path

        if engine == "piper":
            model = self.cfg.voice.piper_model
            if not model:
                raise JarvisError(
                    "Piper needs a voice model.",
                    hint="Set voice.piper_model to a .onnx file, e.g. "
                    "~/piper/en_US-lessac-medium.onnx",
                )
            path = self.audio_dir / f"{stamp}-{word}.wav"
            try:
                proc = subprocess.run(
                    ["piper", "--model", str(Path(model).expanduser()), "--output_file", str(path)],
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=180,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise JarvisError(f"piper failed: {exc}") from exc
            if proc.returncode != 0:
                raise JarvisError(f"piper failed: {proc.stderr.strip()[:300]}")
            self.last_file = path
            if should_play:
                self._play_file(path)
            return path

        # espeak
        path = self.audio_dir / f"{stamp}-{word}.wav"
        binary = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
        cmd = [binary, "-w", str(path), "-s", str(max(80, min(self.cfg.voice.rate, 450)))]
        if self.cfg.voice.voice:
            cmd += ["-v", self.cfg.voice.voice]
        cmd.append(text)
        code, _, err = run_capture(cmd, timeout=120)
        if code != 0 or not path.exists():
            raise JarvisError(f"{binary} failed: {err.strip()[:300]}")
        self.last_file = path
        if should_play:
            self._play_file(path)
        return path

    def _play_file(self, path: Path) -> None:
        # Play via a per-platform CLI; failures fall back to "file saved".
        players: list[list[str]] = []
        if platform.system() == "Darwin":
            players.append(["afplay", str(path)])
        if shutil.which("ffplay"):
            players.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)])
        if shutil.which("aplay"):
            players.append(["aplay", "-q", str(path)])
        if shutil.which("paplay"):
            players.append(["paplay", str(path)])
        for cmd in players:
            if shutil.which(cmd[0]):
                try:
                    self._proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                    )
                    return
                except OSError:
                    continue
        # No player: keep the file, tell nobody; the transcript is what matters.

    def wait(self, timeout: float = 120) -> None:
        if self._proc is not None:
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    def last_transcript(self) -> str:
        return getattr(self, "_transcript", "")

    def speak_rich(self, text: str) -> ToolResult:
        path = self.speak(text)
        if path is None:
            return ToolResult("Spoken aloud (no file saved).")
        return ToolResult(
            f"Spoken aloud. Audio saved to {path} ({human_bytes(path.stat().st_size)})."
        )


def clean_for_speech(text: str) -> str:
    """Turn markdown-ish text into something TTS reads naturally."""
    import re

    text = re.sub(r"```.*?```", " code block omitted. ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"(\*\*|__|\*|_)(.+?)\1", r"\2", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)
    text = re.sub(r"[|]{1}", ", ", text)
    text = re.sub(r"-{3,}", "\n", text)
    text = re.sub(r"\[(\d+)\]", "", text)
    text = re.sub(r"https?://\S+", "a link", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def speak_async(engine: VoiceEngine, text: str) -> threading.Thread:
    """Speak in the background so the CLI stays responsive."""
    thread = threading.Thread(target=_safe_speak, args=(engine, text), daemon=True)
    thread.start()
    return thread


def _safe_speak(engine: VoiceEngine, text: str) -> None:
    try:
        engine.speak(text)
    except JarvisError:
        pass
