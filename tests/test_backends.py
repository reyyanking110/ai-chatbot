"""Backend protocol tests using recorded Ollama payloads and a local fake server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jarvis_local.backends import (
    MockBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
    _parse_ollama_response,
    make_backend,
)

OLLAMA_TOOL_RESPONSE = {
    "model": "qwen3:8b",
    "created_at": "2026-11-26T09:00:00Z",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "notes.md", "start_line": 1},
                }
            }
        ],
    },
    "done": True,
}

OLLAMA_TEXT_RESPONSE = {
    "model": "qwen3:8b",
    "message": {"role": "assistant", "content": "Hello — all local."},
    "done": True,
}

OLLAMA_STRING_ARGS_RESPONSE = {
    "model": "qwen3:8b",
    "message": {
        "role": "assistant",
        "content": "looking",
        "tool_calls": [
            {"function": {"name": "run_shell", "arguments": '{"command": "ls -la"}'}}
        ],
    },
}

OLLAMA_EMBED_RESPONSE = {"embedding": [0.1, 0.2, 0.3]}


def test_parse_tool_call():
    result = _parse_ollama_response(OLLAMA_TOOL_RESPONSE, "qwen3:8b", 1.2)
    assert result.wants_tools
    call = result.tool_calls[0]
    assert call["function"]["name"] == "read_file"
    assert call["function"]["arguments"]["path"] == "notes.md"
    assert call["id"].startswith("call_")


def test_parse_plain_text():
    result = _parse_ollama_response(OLLAMA_TEXT_RESPONSE, "qwen3:8b", 0.5)
    assert not result.wants_tools
    assert result.content == "Hello — all local."
    assert result.model == "qwen3:8b"


def test_parse_string_arguments_are_decoded():
    result = _parse_ollama_response(OLLAMA_STRING_ARGS_RESPONSE, "m", 0.0)
    assert result.tool_calls[0]["function"]["arguments"] == {"command": "ls -la"}


class _FakeOllama(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def log_message(self, *args):  # quiet
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/tags":
            self._send({"models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text"}]})
        elif self.path == "/api/version":
            self._send({"version": "0.9.9"})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).calls.append({"path": self.path, "payload": payload})
        if self.path == "/api/chat":
            if payload.get("tools"):
                self._send(OLLAMA_TOOL_RESPONSE)
            else:
                self._send(OLLAMA_TEXT_RESPONSE)
        elif self.path == "/api/embeddings":
            self._send(OLLAMA_EMBED_RESPONSE)
        else:
            self._send({"error": "unknown endpoint"}, 404)


@pytest.fixture()
def fake_ollama(sandbox):
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cfg = sandbox["cfg"]
    cfg.ollama.host = f"http://127.0.0.1:{server.server_port}"
    _FakeOllama.calls = []
    yield cfg
    server.shutdown()
    server.server_close()


def test_ollama_health_and_models(fake_ollama):
    backend = OllamaBackend(fake_ollama)
    status = backend.health()
    assert status.ok
    assert "qwen3:8b" in status.models
    assert status.version == "0.9.9"


def test_ollama_chat_with_tools_roundtrip(fake_ollama):
    backend = OllamaBackend(fake_ollama)
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    result = backend.chat(
        "qwen3:8b",
        [{"role": "user", "content": "read notes"}],
        tools,
        options={"num_ctx": 4096},
    )
    assert result.tool_calls
    sent = _FakeOllama.calls[-1]["payload"]
    assert sent["options"]["num_ctx"] == 4096
    assert sent["tools"][0]["function"]["name"] == "read_file"
    assert sent["messages"][0]["role"] == "user"


def test_ollama_chat_without_tools(fake_ollama):
    backend = OllamaBackend(fake_ollama)
    result = backend.chat("qwen3:8b", [{"role": "user", "content": "hi"}])
    assert result.content == "Hello — all local."
    assert "tools" not in _FakeOllama.calls[-1]["payload"]


def test_ollama_embed(fake_ollama):
    backend = OllamaBackend(fake_ollama)
    vectors = backend.embed("nomic-embed-text", ["hello", "world"])
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]


def test_ollama_tool_message_serialisation(fake_ollama):
    backend = OllamaBackend(fake_ollama)
    backend.chat(
        "qwen3:8b",
        [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": {"path": "x"}}}],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "contents"},
        ],
    )
    sent = _FakeOllama.calls[-1]["payload"]["messages"]
    assert sent[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert sent[2]["role"] == "tool"
    assert sent[2]["tool_name"] == "read_file"
    assert "tool_call_id" not in sent[2]


def test_ollama_unreachable_gives_friendly_error(sandbox):
    cfg = sandbox["cfg"]
    cfg.ollama.host = "http://127.0.0.1:9"  # nothing listens here
    backend = OllamaBackend(cfg)
    status = backend.health()
    assert not status.ok
    with pytest.raises(Exception) as exc:
        backend.chat("m", [{"role": "user", "content": "x"}])
    assert "ollama serve" in str(exc.value).lower() or "request failed" in str(exc.value).lower()


def test_make_backend_selection(sandbox):
    cfg = sandbox["cfg"]
    assert isinstance(make_backend(cfg), MockBackend)
    cfg.backend = "ollama"
    assert isinstance(make_backend(cfg), OllamaBackend)
    cfg.backend = "openai-compatible"
    assert isinstance(make_backend(cfg), OpenAICompatibleBackend)


def test_ollama_message_conversion_keeps_images():
    from jarvis_local.backends import _ollama_message

    msg = _ollama_message({"role": "user", "content": "what is this?", "images": ["b64"]})
    assert msg["images"] == ["b64"]


class _FakeOpenAI(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def log_message(self, *args):
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._send({"object": "list", "data": [{"id": "qwen2.5-coder-7b"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).calls.append({"path": self.path, "payload": payload, "auth": self.headers.get("Authorization")})
        if self.path.rstrip("/") == "/v1/chat/completions":
            if payload.get("tools"):
                self._send(
                    {
                        "model": payload["model"],
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_abc",
                                            "type": "function",
                                            "function": {
                                                "name": "list_files",
                                                "arguments": json.dumps({"path": "."}),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                    }
                )
            else:
                self._send(
                    {
                        "model": payload["model"],
                        "choices": [{"message": {"role": "assistant", "content": "Local llama.cpp here."}}],
                    }
                )
        elif self.path.rstrip("/") == "/v1/embeddings":
            self._send({"data": [{"index": 0, "embedding": [0.5, 0.5]}]})
        else:
            self._send({"error": "unknown"}, 404)


@pytest.fixture()
def fake_openai(sandbox):
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cfg = sandbox["cfg"]
    cfg.backend = "openai-compatible"
    cfg.openai.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    cfg.openai.model = "qwen2.5-coder-7b"
    cfg.openai.api_key = "local-test"
    _FakeOpenAI.calls = []
    yield cfg
    server.shutdown()
    server.server_close()


def test_openai_health(fake_openai):
    backend = OpenAICompatibleBackend(fake_openai)
    status = backend.health()
    assert status.ok
    assert "qwen2.5-coder-7b" in status.models


def test_openai_chat_with_tools(fake_openai):
    backend = OpenAICompatibleBackend(fake_openai)
    tools = [{"type": "function", "function": {"name": "list_files", "parameters": {}}}]
    result = backend.chat("qwen2.5-coder-7b", [{"role": "user", "content": "list files"}], tools)
    assert result.tool_calls[0]["function"]["name"] == "list_files"
    assert result.tool_calls[0]["function"]["arguments"] == {"path": "."}
    sent = _FakeOpenAI.calls[-1]
    assert sent["auth"] == "Bearer local-test"
    assert sent["payload"]["tool_choice"] == "auto"


def test_openai_plain_chat_and_embed(fake_openai):
    backend = OpenAICompatibleBackend(fake_openai)
    result = backend.chat("qwen2.5-coder-7b", [{"role": "user", "content": "hi"}])
    assert result.content == "Local llama.cpp here."
    vectors = backend.embed("nomic-embed-text", ["hello"])
    assert vectors == [[0.5, 0.5]]


def test_openai_tool_message_shape(fake_openai):
    backend = OpenAICompatibleBackend(fake_openai)
    backend.chat(
        "m",
        [
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "call_abc", "name": "list_files", "content": "a.txt"},
        ],
    )
    msgs = _FakeOpenAI.calls[-1]["payload"]["messages"]
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "call_abc"
    assert msgs[1]["name"] == "list_files"


def test_full_agent_loop_against_openai_server(fake_openai):
    """The real runtime drives a real (fake, local) OpenAI-compatible server."""
    from jarvis_local.agents import get_agent
    from jarvis_local.runtime import AgentRuntime, build_context

    cfg = fake_openai
    ctx = build_context(cfg, OpenAICompatibleBackend(cfg))
    runtime = AgentRuntime(ctx, get_agent("code"))
    result = runtime.run("what is in the workspace?")
    assert result.tool_calls >= 1
    assert any(step.name == "list_files" for step in result.steps)
    ctx.memory.close()
    ctx.state.close()
