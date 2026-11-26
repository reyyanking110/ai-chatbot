# Architecture

Jarvis Local is a small stdlib-first application. This document explains how the
pieces fit, so you can extend or audit it without reading every file.

## Layers

```
┌──────────────────────────── CLI (cli.py) ────────────────────────────┐
│  Typer commands, Rich rendering, approvals, chat loop, /commands     │
└───────────────┬──────────────────────────────────────────────────────┘
                │ builds
┌───────────────▼────────────── runtime.py ────────────────────────────┐
│  RunContext (config, policy, memory, library, voice, skills)         │
│  AgentRuntime: messages → model → tool calls → results → answer      │
└───┬───────────────┬────────────────┬─────────────────┬───────────────┘
    │               │                │                 │
┌───▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│ agents/ │  │ backends.py │  │  tools/     │  │ policy.py   │
│ prompts │  │ Ollama      │  │ files shell │  │ paths,      │
│ + prep  │  │ OpenAI-like │  │ system kb   │  │ commands,   │
│         │  │ mock        │  │             │  │ audit       │
└─────────┘  └─────────────┘  └─────────────┘  └─────────────┘
                    │                 │
        ┌───────────▼──────┐  ┌───────▼────────┐
        │ documents.py     │  │ memory.py      │
        │ SQLite FTS5 +    │  │ memories,      │
        │ trigram + vector │  │ sessions       │
        └──────────────────┘  └────────────────┘
```

## The agent loop

`AgentRuntime.run(user_input)`:

1. Ask the agent to `prepare(ctx, user_input)`. This returns a system prompt
   (persona + role rules + memory + skills + retrieved sources) and the user
   message.
2. Fit history into the context window (`_fit_context`), trimming whole
   tool-call exchanges so a tool result is never orphaned.
3. Call the backend with the tool schemas allowed for that agent.
4. If the model requests tools: append the assistant message, execute each call
   (policy → approval → tool), append tool results, loop.
5. Stop on a final message or `agent.max_steps`.

Every step is emitted through `ctx.on_step` so the CLI can render progress, and
written through `ctx.policy.audit`.

## Why SQLite for retrieval

For a personal corpus (tens of thousands of chunks) an embedded database is
faster to start and easier to back up than a vector server.

* `chunks_fts` — FTS5 with `porter unicode61`, BM25 ranking for English and most
  European languages.
* `chunks_trgm` — FTS5 trigram index for substring matching: Chinese/Japanese
  (no spaces), and Arabic-script languages like Urdu where articles attach to
  words.
* `vectors` — float32 blobs from the configured embedding model, cosine search in
  Python. Never mixed across models; if the active model differs from the one
  that built the index, semantic search pauses and says so.
* Fusion — reciprocal rank fusion between keyword/substring and vector results.
  Each hit carries `doc_id`, `ordinal`, `page` and `line` so citations point at
  real locations.

## Safety boundaries

| Boundary | Enforced in |
| --- | --- |
| Reads limited to workspace + library + extras | `Policy.resolve_read` |
| Writes only in the workspace | `Policy.resolve_write` |
| Catastrophic commands denied | `Policy.inspect_command` |
| Mutations need approval | `Tool.needs_approval` + policy verdicts |
| Secrets stripped from subprocess env | `Policy.scrubbed_env` |
| Skills scanned and confirmed | `skills.scan_folder` + `SkillManager.install` |
| Audit trail | `Policy.audit` → `logs/audit.jsonl` |

Approval is a *human* control, not a sandbox. Running Jarvis with
`security.approval = auto` is equivalent to trusting the model with your user
account; do that only in controlled environments.

## The mock backend

`MockBackend` is not a stub for UI tests only. It follows the real protocol: it
can request tools, it reads retrieved sources from the prompt, and it writes
deterministic text. That is how `jarvis demo` exercises the whole pipeline —
retrieval, citations, the digest renderer and approvals — with no model at all.

## Extending

* New agent: subclass `Agent`, implement `prepare`, call `register`.
* New tool: add a `Tool` with a JSON schema; scope it with `agents=`.
* New backend: implement `health/list_models/chat/embed` and return `ChatResult`.
* New skill: no code changes; see `examples/skills/`.
