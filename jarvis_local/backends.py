"""Model backends.

Jarvis talks to one of:

* **Ollama** (default) — easiest local runtime, native tool calling.
* **OpenAI-compatible** — llama.cpp server, LM Studio, vLLM, text-generation-webui.
* **Mock** — deterministic, offline responses for demos, CI, and `jarvis demo`.

All backends implement :class:`BaseBackend`.  Messages use the OpenAI chat
shape: ``{"role": ..., "content": ...}`` plus optional ``tool_calls`` on
assistant messages and ``tool_call_id``/``name`` on tool messages.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import Config
from .errors import BackendError

Message = dict[str, Any]
ToolSpec = dict[str, Any]


@dataclass
class ChatResult:
    content: str
    model: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class HealthStatus:
    ok: bool
    detail: str
    models: list[str] = field(default_factory=list)
    version: str = ""


class BaseBackend(Protocol):
    name: str

    def health(self) -> HealthStatus: ...

    def list_models(self) -> list[str]: ...

    def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.3,
        options: dict[str, Any] | None = None,
    ) -> ChatResult: ...

    def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token English, ~2 for CJK/code)."""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    return max(1, int((len(text) - cjk) / 4 + cjk * 0.8))


def context_budget(num_ctx: int, reserve_ratio: float = 0.55) -> int:
    return max(256, int(num_ctx * reserve_ratio))


def extract_json(text: str) -> Any | None:
    """Pull the first plausible JSON object/array out of model output."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for chunk in candidates:
        chunk = chunk.strip()
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = chunk.find(opener)
            end = chunk.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(chunk[start : end + 1])
                except json.JSONDecodeError:
                    continue
    return None


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaBackend:
    name = "ollama"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.host = cfg.ollama.host.rstrip("/")
        self.timeout = float(cfg.ollama.timeout_s)

    def _client(self) -> httpx.Client:
        # Trust the local endpoint only; proxies must not intercept localhost.
        return httpx.Client(
            base_url=self.host,
            timeout=self.timeout,
            trust_env=False,
            headers={"Content-Type": "application/json"},
        )

    def health(self) -> HealthStatus:
        try:
            with self._client() as client:
                resp = client.get("/api/tags", timeout=3.0)
                resp.raise_for_status()
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                version = ""
                try:
                    v = client.get("/api/version", timeout=2.0)
                    if v.status_code == 200:
                        version = v.json().get("version", "")
                except httpx.HTTPError:
                    pass
                detail = f"Ollama at {self.host}"
                if version:
                    detail += f" (v{version})"
                return HealthStatus(ok=True, detail=detail, models=models, version=version)
        except httpx.HTTPError as exc:
            return HealthStatus(
                ok=False,
                detail=f"Cannot reach Ollama at {self.host}: {exc.__class__.__name__}",
            )

    def list_models(self) -> list[str]:
        return self.health().models

    def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.3,
        options: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_ollama_message(m) for m in messages],
            "stream": False,
            "options": {"temperature": temperature, **(options or {})},
        }
        if tools:
            payload["tools"] = [_ollama_tool(t) for t in tools]
        started = time.monotonic()
        try:
            with self._client() as client:
                resp = client.post("/api/chat", json=payload)
                if resp.status_code >= 400:
                    raise BackendError(
                        _explain_ollama_error(resp, model),
                        hint="Run `jarvis doctor` to check the backend and installed models.",
                    )
                data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendError(
                f"Ollama request failed: {exc}",
                hint="Is `ollama serve` running? Try `jarvis doctor`.",
            ) from exc
        return _parse_ollama_response(data, model, time.monotonic() - started)

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with self._client() as client:
            for text in texts:
                try:
                    resp = client.post("/api/embeddings", json={"model": model, "prompt": text})
                except httpx.HTTPError as exc:
                    raise BackendError(f"Embedding request failed: {exc}") from exc
                if resp.status_code == 404:
                    raise BackendError(
                        f"Embedding model '{model}' is not installed.",
                        hint=f"Run `ollama pull {model}`.",
                    )
                if resp.status_code >= 400:
                    raise BackendError(
                        f"Embedding failed ({resp.status_code}): {resp.text[:200]}",
                        hint=f"Check that `{model}` is an embedding model.",
                    )
                data = resp.json()
                vectors.append([float(x) for x in data.get("embedding", [])])
        return vectors


def _ollama_message(message: Message) -> dict[str, Any]:
    role = message.get("role", "user")
    out: dict[str, Any] = {"role": role, "content": message.get("content") or ""}
    if role == "assistant" and message.get("tool_calls"):
        out["tool_calls"] = [
            {
                "function": {
                    "name": tc.get("function", {}).get("name", tc.get("name", "")),
                    "arguments": tc.get("function", {}).get("arguments", {}),
                }
            }
            for tc in message["tool_calls"]
        ]
    if role == "tool":
        out["tool_name"] = message.get("name") or message.get("tool_name") or "tool"
        # Ollama expects content as a string; keep it compact.
        content = message.get("content") or ""
        out["content"] = content if isinstance(content, str) else json.dumps(content)
        out.pop("tool_call_id", None)
    if "images" in message:
        out["images"] = message["images"]
    return out


def _ollama_tool(tool: ToolSpec) -> dict[str, Any]:
    if "function" in tool:
        return tool
    return {"type": "function", "function": tool}


def _explain_ollama_error(resp: httpx.Response, model: str) -> str:
    body = ""
    try:
        body = resp.json().get("error", "")
    except (json.JSONDecodeError, AttributeError):
        body = resp.text[:300]
    if resp.status_code == 404 and ("model" in body.lower() or "not found" in body.lower()):
        return f"Model '{model}' is not installed in Ollama."
    if "does not support tools" in body.lower() or "tools" in body.lower() and "support" in body.lower():
        return f"Model '{model}' does not support tool calling."
    return f"Ollama returned HTTP {resp.status_code}: {body}"


def _parse_ollama_response(data: dict[str, Any], model: str, elapsed: float) -> ChatResult:
    message = data.get("message") or {}
    calls: list[dict[str, Any]] = []
    for idx, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"value": args}
        calls.append(
            {
                "id": f"call_{idx}_{abs(hash(json.dumps(args, sort_keys=True))) % 10**8}",
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": args},
            }
        )
    return ChatResult(
        content=message.get("content") or "",
        model=data.get("model", model),
        tool_calls=calls,
        raw=data,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible servers (llama.cpp, LM Studio, vLLM, ...)
# ---------------------------------------------------------------------------

class OpenAICompatibleBackend:
    name = "openai-compatible"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.base_url = cfg.openai.base_url.rstrip("/")
        self.api_key = cfg.openai.api_key or "local"
        self.timeout = float(cfg.ollama.timeout_s)

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            trust_env=False,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

    def health(self) -> HealthStatus:
        try:
            with self._client() as client:
                resp = client.get("/models", timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("id", "") for m in data.get("data", [])]
                    return HealthStatus(
                        ok=True, detail=f"OpenAI-compatible server at {self.base_url}", models=models
                    )
                return HealthStatus(
                    ok=False, detail=f"Server responded HTTP {resp.status_code} at {self.base_url}"
                )
        except httpx.HTTPError as exc:
            return HealthStatus(
                ok=False,
                detail=f"Cannot reach {self.base_url}: {exc.__class__.__name__}",
            )

    def list_models(self) -> list[str]:
        return self.health().models

    def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.3,
        options: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_openai_message(m) for m in messages],
            "temperature": temperature,
            "max_tokens": (options or {}).get("num_predict", 2048),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        started = time.monotonic()
        try:
            with self._client() as client:
                resp = client.post("/chat/completions", json=payload)
                if resp.status_code >= 400:
                    raise BackendError(
                        f"Local server returned HTTP {resp.status_code}: {resp.text[:300]}",
                        hint=f"Check that model '{model}' is loaded in your local server.",
                    )
                data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendError(
                f"OpenAI-compatible request failed: {exc}",
                hint=f"Is your local server running at {self.base_url}?",
            ) from exc
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = []
        for idx, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(
                {
                    "id": tc.get("id") or f"call_{idx}",
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                }
            )
        return ChatResult(
            content=message.get("content") or "",
            model=data.get("model", model),
            tool_calls=calls,
            raw=data,
            elapsed_s=time.monotonic() - started,
        )

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        try:
            with self._client() as client:
                resp = client.post("/embeddings", json={"model": model, "input": texts})
                if resp.status_code >= 400:
                    raise BackendError(
                        f"Embedding request failed ({resp.status_code}): {resp.text[:200]}",
                        hint="Ensure your server exposes an /embeddings endpoint.",
                    )
                data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"Embedding request failed: {exc}") from exc
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [[float(v) for v in item.get("embedding", [])] for item in items]


def _openai_message(message: Message) -> dict[str, Any]:
    role = message.get("role", "user")
    out: dict[str, Any] = {"role": role, "content": message.get("content") or ""}
    if role == "assistant" and message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
        if not out["content"]:
            out["content"] = None
    if role == "tool":
        out["tool_call_id"] = message.get("tool_call_id", "call_0")
        out["name"] = message.get("name", "tool")
    return out


# ---------------------------------------------------------------------------
# Mock backend — deterministic, no network, no model download
# ---------------------------------------------------------------------------

class MockBackend:
    """Answers from a small scripted brain.

    It is intentionally dumb, but it exercises the *real* agent loop, tool
    execution, memory, RAG citations and rendering, so the whole app can be
    explored before a single model is downloaded.
    """

    name = "mock"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg
        self.calls = 0

    def health(self) -> HealthStatus:
        return HealthStatus(
            ok=True,
            detail="Mock backend (no model needed — for demos and tests)",
            models=["mock-1"],
        )

    def list_models(self) -> list[str]:
        return ["mock-1"]

    def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.3,
        options: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls += 1
        system = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "system")
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            {"content": ""},
        )
        question = (last_user.get("content") or "").strip()
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        available = {t.get("function", t).get("name", ""): t for t in (tools or [])}

        # 0. Digest requests: compose a readable briefing from the raw agenda.
        if "You write" in system and "morning briefing" in system and "CALENDAR EVENTS" in question:
            return ChatResult(content=_mock_digest(question), model="mock-1")

        # 1. Tool results already in hand: summarise them.
        if tool_messages:
            body = tool_messages[-1].get("content", "")
            return ChatResult(
                content=(
                    "[mock mode] The tool ran and returned:\n\n> "
                    + _shorten(str(body), 900).replace("\n", "\n> ")
                    + "\n\n_Connect Ollama and pick a real model for a natural-language answer._"
                ),
                model="mock-1",
            )

        # 2. Research: quote the retrieved sources with real citation markers.
        if "# Sources" in system:
            sources = _mock_sources(system)
            if sources:
                question_words = {
                    w for w in re.findall(r"[a-z]{4,}", question.lower())
                } - {"what", "when", "where", "which", "would", "could", "should", "about", "does"}
                scored = []
                for label, body in sources:
                    overlap = len(question_words & {w.lower() for w in re.findall(r"[A-Za-z]{4,}", body)})
                    scored.append((overlap, label, body))
                scored.sort(key=lambda item: item[0], reverse=True)
                picked = [item for item in scored if item[0] > 0][:2] or scored[:1]
                quoted = "\n\n".join(
                    f"**{label}**\n\n> {_shorten(body, 420)}"
                    for _overlap, label, body in picked
                )
                return ChatResult(
                    content=(
                        "[mock mode] Here is what your documents say — quoted verbatim, with "
                        "citations:\n\n" + quoted.replace("[n]", "")
                        + "\n\n_The mock model only quotes; a local 8B+ model reasons over these "
                        "passages and answers in prose._"
                    ),
                    model="mock-1",
                )

        # 3. Coding: demonstrate a real tool call (gated by the same approvals).
        # Detect the coding role either by its system prompt or by the presence
        # of the file/shell toolset (the mock is also used in tests directly).
        coding_tools = {"write_file", "run_shell", "run_python"} & set(available)
        if ("senior engineer working inside the user's workspace" in system or len(coding_tools) >= 2) \
                and "write_file" in available:
            filename = "mock_hello.py"
            return ChatResult(
                content="",
                model="mock-1",
                tool_calls=[
                    _mock_call(
                        "write_file",
                        {
                            "path": filename,
                            "content": (
                                '"""Created by Jarvis in mock mode."""\n\n'
                                "def greet(name: str = 'world') -> str:\n"
                                "    return f'hello, {name}!'\n\n\n"
                                "if __name__ == '__main__':\n"
                                "    print(greet())\n"
                            ),
                        },
                    )
                ],
            )

        # 4. Slash-style demo commands.
        if question.startswith("!demo"):
            return ChatResult(
                content=(
                    "This is **mock mode** — Jarvis is wired up end to end, but answers come "
                    "from a stub, not a model.\n\n"
                    "When you are ready for real intelligence:\n\n"
                    "1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`\n"
                    "2. `jarvis init` — it detects your hardware and pulls fitting models\n"
                    "3. `jarvis chat` — same interface, local model"
                ),
                model="mock-1",
            )

        return ChatResult(
            content=(
                f"[mock mode] You said: “{_shorten(question, 240)}”\n\n"
                "I can route this to a real local model once Ollama is running. "
                "Try `jarvis doctor`, `jarvis models`, or `jarvis demo` for a guided tour."
            ),
            model="mock-1",
        )

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        # Deterministic hashed bag-of-words vectors; good enough for keyword-ish recall.
        return [_hashed_vector(t, 64) for t in texts]


def _mock_sources(system: str) -> list[tuple[str, str]]:
    import re

    blocks = re.findall(r"\[(\d+)\] ([^\n]+)\nfile: ([^\n]+)\n---\n(.*?)(?=\n\[\d+\] |\Z)", system, re.S)
    out = []
    for number, label, _path, body in blocks:
        body = body.strip()
        if body:
            out.append((f"[{number}] {label.strip()}", body))
    return out


def _mock_digest(raw: str) -> str:
    events = []
    mails = []
    section = ""
    for line in raw.splitlines():
        if line.startswith("CALENDAR EVENTS"):
            section = "events"
            continue
        if line.startswith("UNREAD / IMPORTANT EMAIL"):
            section = "mail"
            continue
        if line.startswith("TODAY:"):
            continue
        if line.strip().startswith("- ") and section == "events":
            events.append(line.strip()[2:])
        elif line.strip().startswith("- [") and section == "mail":
            mails.append(line.strip()[2:])
    lines = [
        "[mock mode] Good morning. Here is the shape of your day.",
        "",
        f"You have {len(events)} events and {len(mails)} emails flagged for a look.",
        "",
        "**Your calendar**",
    ]
    for event in events[:6]:
        lines.append(f"- {event}")
    if not events:
        lines.append("- Nothing scheduled.")
    lines += ["", "**Email that needs you**"]
    for mail in mails[:5]:
        lines.append(f"- {mail}")
    if not mails:
        lines.append("- Nothing urgent.")
    lines += [
        "",
        "_The mock model assembles this from raw data without writing prose. "
        "A local model turns the same data into a natural spoken briefing._",
    ]
    return "\n".join(lines)


def _mock_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "call_mock",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class LocalHashEmbedder:
    """Deterministic hashing embedder — no model, no network.

    This is the fallback when the configured embedding model has not been
    pulled yet. It is not a neural model: it captures word overlap with a
    hashed bag-of-words (plus bigrams), which still improves recall over pure
    keyword search for paraphrases that share words. Once Ollama's embedding
    model is available, Jarvis uses real vectors instead.
    """

    def __init__(self, model: str = "local-hash-384", dims: int = 768) -> None:
        self.model = model
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        import re

        vector = [0.0] * self.dims
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for token in grams:
            digest = _hash_token(token)
            idx = digest % self.dims
            vector[idx] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]


def _hash_token(token: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest()[:6], "little")


def embedding_dimension(model: str) -> int:
    from .hardware import EMBED_DIMENSIONS

    return EMBED_DIMENSIONS.get(model.split(":")[0], 768)


def _hashed_vector(text: str, dims: int) -> list[float]:
    import hashlib

    vector = [0.0] * dims
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[idx] += sign
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


def _shorten(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def make_backend(cfg: Config) -> BaseBackend:
    backend = (cfg.backend or "ollama").lower()
    if backend == "mock":
        return MockBackend(cfg)
    if backend == "ollama":
        return OllamaBackend(cfg)
    if backend in {"openai", "openai-compatible", "llamacpp", "lmstudio", "vllm"}:
        return OpenAICompatibleBackend(cfg)
    raise BackendError(
        f"Unknown backend '{cfg.backend}'.",
        hint="Use one of: ollama, openai-compatible, mock.",
    )


def describe_backend(backend: BaseBackend) -> str:
    status = backend.health()
    return f"{backend.name}: {status.detail}"
