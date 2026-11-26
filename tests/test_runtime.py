from __future__ import annotations

from jarvis_local.agents import get_agent
from jarvis_local.backends import MockBackend, estimate_tokens, extract_json
from jarvis_local.runtime import AgentRuntime, build_context, deny_all


def _runtime(sandbox, agent_name="chat", approve=deny_all):
    cfg = sandbox["cfg"]
    ctx = build_context(cfg, MockBackend(cfg), approve=approve)
    return ctx, AgentRuntime(ctx, get_agent(agent_name))


def test_chat_turn(sandbox):
    ctx, runtime = _runtime(sandbox)
    result = runtime.run("hello there")
    assert "mock mode" in result.text.lower()
    assert result.elapsed_s >= 0
    ctx.memory.close()
    ctx.state.close()


def test_code_agent_writes_file_with_approval(sandbox):
    ctx, runtime = _runtime(sandbox, "code", approve=lambda kind, prompt: True)
    result = runtime.run("!code write a hello script")
    assert result.tool_calls == 1
    created = sandbox["workspace"] / "mock_hello.py"
    assert created.exists()
    assert "hello" in created.read_text()
    ctx.memory.close()
    ctx.state.close()


def test_code_agent_respects_denied_approval(sandbox):
    ctx, runtime = _runtime(sandbox, "code", approve=lambda kind, prompt: False)
    result = runtime.run("!code write a hello script")
    assert result.tool_calls == 1
    assert not (sandbox["workspace"] / "mock_hello.py").exists()
    assert "declined" in result.text.lower()
    ctx.memory.close()
    ctx.state.close()


def test_research_agent_uses_library(sandbox):
    from jarvis_local.documents import Library

    docs = sandbox["library"]
    (docs / "facts.md").write_text("The capital of the fictional region is Zanziburg.\n")
    cfg = sandbox["cfg"]
    lib = Library(cfg)
    lib.index([docs])
    ctx = build_context(cfg, MockBackend(cfg), approve=deny_all, library=lib)
    runtime = AgentRuntime(ctx, get_agent("research"))
    result = runtime.run("What is the capital?")
    assert result.hits
    assert "Zanziburg" in result.text
    lib.close()
    ctx.memory.close()
    ctx.state.close()


def test_runtime_handles_unknown_tool(sandbox):
    from jarvis_local.runtime import AgentRuntime

    cfg = sandbox["cfg"]
    ctx = build_context(cfg, MockBackend(cfg))
    steps = []
    ctx.on_step = steps.append

    class RogueAgent:
        name = "chat"

        def model_for(self, cfg):
            return "mock-1"

        def prepare(self, ctx, user_input):
            from jarvis_local.agents.base import Prepared

            return Prepared(system="s", user="u")

    class RogueBackend(MockBackend):
        def chat(self, model, messages, tools=None, **kw):
            from jarvis_local.backends import ChatResult

            if not any(m.get("role") == "tool" for m in messages):
                return ChatResult("", tool_calls=[{"id": "c1", "function": {"name": "not_a_tool", "arguments": {}}}])
            return ChatResult("recovered")

    ctx.backend = RogueBackend(cfg)
    runtime = AgentRuntime(ctx, RogueAgent())
    result = runtime.run("do something odd")
    assert result.text == "recovered"
    assert any(not s.ok for s in steps)
    ctx.memory.close()
    ctx.state.close()


def test_backend_error_is_friendly(sandbox):
    from jarvis_local.errors import BackendError

    cfg = sandbox["cfg"]
    ctx = build_context(cfg, MockBackend(cfg))

    class Failing(MockBackend):
        def chat(self, *a, **k):
            raise BackendError("Ollama request failed", hint="Is `ollama serve` running?")

    ctx.backend = Failing(cfg)
    runtime = AgentRuntime(ctx, get_agent("chat"))
    result = runtime.run("hi")
    assert "couldn't reach" in result.text.lower()
    assert "ollama serve" in result.text
    ctx.memory.close()
    ctx.state.close()


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('sure! [1,2,3] done') == [1, 2, 3]
    assert extract_json("no json here") is None


def test_token_estimate():
    assert 5 < estimate_tokens("hello world this is a sentence") < 20
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好世界") >= 2


def test_mock_tool_calling_shape():
    backend = MockBackend()
    tools = [
        {"type": "function", "function": {"name": name}}
        for name in ("write_file", "run_shell", "run_python", "read_file")
    ]
    result = backend.chat("mock-1", [{"role": "user", "content": "make a script"}], tools=tools)
    assert result.tool_calls
    assert result.tool_calls[0]["function"]["name"] == "write_file"
    # A tool result gets summarised rather than triggering another call.
    followup = backend.chat(
        "mock-1",
        [
            {"role": "user", "content": "make a script"},
            {"role": "tool", "content": "Created mock_hello.py", "name": "write_file"},
        ],
        tools=tools,
    )
    assert not followup.tool_calls and "mock_hello.py" in followup.content
