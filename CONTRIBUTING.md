# Contributing to Jarvis Local

Thanks for helping. This project is deliberately small, dependency-light and
local-first; those constraints are features.

## Ground rules

- **No network calls at import time** and no telemetry, ever.
- **No new required dependencies** without a strong reason. `typer`, `rich`,
  `httpx` and `platformdirs` are the whole core. Anything else goes behind an
  optional extra.
- **Privacy**: never send user content, file paths or memory to a remote service.
  The only outbound calls allowed are ones the user explicitly initiated (a URL
  they asked to fetch, a skill they asked to install, a model pull).
- **Safety first**: any new tool with side effects must go through
  `Policy`/approval. Do not add a tool that writes outside the workspace.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # offline: no model, no network
ruff check jarvis_local tests
```

## Adding an agent

1. Create `jarvis_local/agents/my_agent.py` subclassing `Agent` (see
   `research.py` for retrieval, `code.py` for tools).
2. Register it at module level: `register(MyAgent())`.
3. Import it in `jarvis_local/agents/__init__.py`.
4. If it needs a new tool, add it in `jarvis_local/tools/` with a JSON schema and
   an `agents=` tuple.
5. Add a test in `tests/` and a row to the README table.

## Adding a tool

Tools are `Tool(name, description, parameters, func, category, agents)`. The
function receives `ctx` and keyword arguments matching `parameters`. Raise
`PolicyError`/`JarvisError` for expected failures — the runtime turns them into
readable tool results.

## Adding a backend

Implement `health`, `list_models`, `chat` and `embed`; return `ChatResult` with
OpenAI-shaped `tool_calls`. Register it in `make_backend`.

## Skills

Skills live outside the codebase. Read `README.md#skills` and see
`examples/skills/` for a working script-backed skill and a prompt-only one.

## Pull requests

- Keep diffs focused; describe behaviour changes and include a test.
- Run `pytest` and `ruff check` before pushing.
- For user-facing changes, update `README.md` and `CHANGELOG.md`.
