# ◈ Jarvis Local

**A private, local-first personal AI assistant for your terminal.**

Everything runs on your machine: the models, the memory, the document index, the
speech. No accounts, no API keys, no data leaving your laptop. Jarvis detects
your hardware, picks the strongest local models that actually fit, and gives you
a small team of specialist agents:

| Agent | What it does | Command |
| --- | --- | --- |
| **chat** | Everyday assistant with long-term memory | `jarvis chat` |
| **research** | Answers questions from *your* documents, with citations | `jarvis research "..."` |
| **code** | Reads/writes files, runs shell + Python (approval-gated) | `jarvis code "..."` |
| **digest** | Gmail + Calendar → written and **spoken** morning briefing | `jarvis digest --speak` |

It also installs **skills** — small instruction/script bundles — from GitHub,
scans them for risk, and learns to use them when relevant.

```
You › what did we decide about pricing at the offsite?

[1] q4-planning-offsite · §1 — We agreed to keep the Pro tier at $29/month
    for existing customers through Q1. New customers move to usage-based pricing.

Sources
[1] q4-planning-offsite — §1 · ~/Documents/q4-planning-offsite.md
```

---

## Quick start (5 minutes)

Requirements: **Python 3.11+** and, for real intelligence, **[Ollama](https://ollama.com/download)**.

```bash
git clone https://github.com/reyyanking110/ai-chatbot.git
cd ai-chatbot

python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[google,docs]"                      # google = Gmail/Calendar, docs = PDF/DOCX

jarvis init          # detects hardware, picks models, writes config, (optionally) pulls them
jarvis doctor        # verifies every moving part with concrete fixes
jarvis ask "hello"   # one question
jarvis chat          # interactive session
```

First run: `jarvis ask`, `jarvis chat` and `jarvis digest` will walk you through
setup automatically if no config exists yet.

### No model, no account, no network? Try the tour

```bash
jarvis demo
```

This indexes sample documents, runs real retrieval, generates a digest from a
sample calendar, and shows the skill system — all with a deterministic mock
backend, so you can evaluate the workflow before downloading gigabytes.

---

## Why local-first, and what that means here

| Concern | How Jarvis handles it |
| --- | --- |
| Model inference | Ollama / llama.cpp on your machine. Anything OpenAI-compatible works. |
| Documents | Indexed into one local SQLite file (`library.db`). Embeddings computed locally. |
| Memory | SQLite (`state.db`). Delete it and Jarvis forgets everything. |
| Speech | macOS `say`, or `piper` / `espeak-ng` locally. No cloud TTS. |
| Gmail / Calendar | OAuth **read-only** scopes; tokens never leave your device. |
| Network | Only when *you* ask: a `web_fetch` URL, a skill download, a model pull. Nothing phones home. |
| Audit | Every tool call appended to `logs/audit.jsonl`. |

### Safety model (the shell agent is powerful — deliberately)

- **Writes** are confined to one workspace folder (`~/jarvis-workspace` by default).
- **Reads** are limited to the workspace, your library folders, and extras you configure.
- **Shell commands**: read-only commands run directly; anything that mutates
  files/state shows a preview and asks. Catastrophic patterns (`rm -rf /`, piping
  downloads into `sh`, disk formatting) are blocked outright.
- **Python** runs in a subprocess with a timeout and a scrubbed environment
  (tokens, keys and passwords are stripped).
- **Skills** are scanned for scripts, network access, credential strings and
  shell calls before install, and are never auto-run.
- Approval modes: `confirm` (default), `auto` (for trusted batch use), `read-only`.

---

## How the agents work

```
                 ┌──────────────┐
  your message → │ agent prompt │ → model (local) → tool call? ──→ policy + approval ──┐
                 └──────────────┘        ↑                                          │
                        final answer ←───┴──────── tool result (files, shell, search) ┘
```

- **chat** — general conversation, `remember` / `search_memory`, notes, clock, machine info.
- **research** — hybrid retrieval (BM25 + local embeddings) over your documents, then
  an answer with `[n]` citations mapped to file + page/line. If the documents don't
  cover it, it says so instead of inventing.
- **code** — the full tool loop over files, shell and Python, with every write and
  mutating command approved. Ends with what changed and how it was verified.
- **digest** — fetches calendar events and unread mail (read-only), writes a
  spoken-style briefing, saves it to `briefings/`, and can read it aloud. Runs
  with sample data until you connect Google.

### Morning digest

```bash
jarvis digest                 # text + saved markdown
jarvis digest --speak         # also reads it aloud (say / piper / espeak-ng)
jarvis schedule install -t 07:30   # cron entry, every morning
```

Google is optional. Setup is a three-minute OAuth walkthrough:

```bash
jarvis google setup   # instructions + creates a placeholder secret file
#  1. create a Google Cloud project
#  2. enable Gmail API + Google Calendar API
#  3. OAuth consent screen → External → add yourself as a test user
#  4. create an OAuth client ID of type "Desktop app", download the JSON
#  5. save it over the placeholder file
jarvis google auth    # browser flow, or paste a code in a headless terminal
jarvis google test    # verify read access
```

Scopes are `gmail.readonly` and `calendar.readonly` only. Jarvis cannot send,
delete or modify anything.

### Research with citations

```bash
jarvis library index ~/Documents ~/Notes --add   # incremental; --add remembers these roots
jarvis library stats
jarvis research "what are the risks we flagged for the billing migration?"
```

Supported: Markdown, text, PDF, DOCX, EPUB, HTML, CSV/TSV, JSON, RST, org.
Scanned PDFs need OCR first — Jarvis will tell you when a file has no text layer.

---

## Model selection by hardware

`jarvis init` measures RAM, CPU and GPU (CUDA / ROCm / Apple Metal / Vulkan), then
picks the largest models that fit with headroom for the OS and context window:

| Usable model memory | General | Coding | Fast | Context |
| --- | --- | --- | --- | --- |
| ≤ 4 GB | `llama3.2:3b` | `qwen2.5-coder:1.5b` | `qwen3:0.6b` | 4k |
| ~8 GB | `qwen3:8b` | `qwen2.5-coder:7b` | `qwen3:4b` | 8k |
| ~15 GB | `qwen3:14b` | `qwen2.5-coder:14b` | `qwen3:4b` | 16k |
| ~22 GB | `qwen3:30b-a3b` | `qwen3-coder:30b` | `qwen3:4b` | 32k |
| 40 GB+ | `llama3.3:70b` | `qwen3-coder:30b` | `qwen3:8b` | 32k |

Apple Silicon shares memory between CPU and GPU, so Jarvis reserves about a third
of RAM for macOS and your apps. Embeddings always use `nomic-embed-text` (137M).

```bash
jarvis models plan              # why these models, for this machine
jarvis models current           # what is in use
jarvis models switch coding qwen3-coder:30b
jarvis models pull              # download whatever is missing
```

Prefer llama.cpp / LM Studio / vLLM? `jarvis config set backend openai-compatible`
and `jarvis config set openai.base_url http://127.0.0.1:8080/v1`.

---

## Skills — getting better over time

A skill is a folder with a `SKILL.md`:

```markdown
---
name: world-clock
description: Show local time in several cities and suggest a call window.
version: 1.0.0
triggers: [world clock, time in, call time]
run: scripts/world_clock.py     # optional — exposes a skill_world_clock tool
---

When the user asks for the time somewhere, run the tool with the cities...
```

```bash
jarvis skills install https://github.com/you/your-skill     # GitHub repo or archive URL
jarvis skills install ./my-skill                            # local folder
jarvis skills list                                          # state + risk score
jarvis skills show world-clock                              # instructions + scan
jarvis skills new water-reminder -d "Nudge me to drink water"
jarvis skills export world-clock                            # zip to share
```

Install prints a **security scan** (scripts, network imports, credential-looking
strings, shell calls) and asks before enabling. Skill scripts run only when the
model calls them, and every call passes through the same approval gate as the
shell agent. Two working examples live in [`examples/skills/`](examples/skills/).

---

## Command map

```
jarvis                     welcome / next steps        jarvis init         hardware-aware setup
jarvis doctor [-—deep]     diagnostics + fixes          jarvis demo         offline guided tour
jarvis ask "..."           one-shot answer               jarvis chat         interactive session
jarvis research "..."      documents + citations         jarvis code "..."   files / shell / python
jarvis digest [--speak]    morning briefing              jarvis schedule     cron install/show/remove

jarvis library index|search|stats|list|reset            jarvis agents       list the agents
jarvis models plan|current|list|pull|switch              jarvis sessions     past conversations
jarvis skills    list|install|remove|enable|disable|show|scan|new|export|run|review
jarvis google    status|setup|auth|test                  jarvis memory  list|add|search|forget|clear
jarvis config    show|path|set|edit|reset                jarvis logs         audit trail
jarvis note "..."                                          jarvis --version
```

In `jarvis chat`: `/help`, `/clear`, `/speak`, `/sources`, `/memory`, `/remember`,
`/note`, `/model`, `/doctor`, `/quit`.

Useful config keys (all via `jarvis config set`):

| Key | Default | Meaning |
| --- | --- | --- |
| `assistant_name` | `Jarvis` | What it calls itself |
| `models.general` / `.coding` / `.fast` / `.embed` | auto | Model per role |
| `models.num_ctx` | auto | Context window |
| `security.approval` | `confirm` | `confirm` / `auto` / `read-only` |
| `security.allow_shell` | `true` | Master switch for shell + Python |
| `paths.workspace` | `~/jarvis-workspace` | The only folder Jarvis may write |
| `paths.library` | `~/Documents` | Folders to index |
| `voice.engine` | `auto` | `say` / `piper` / `espeak` / `none` |
| `voice.voice` | system | e.g. `Samantha` (macOS) |
| `library.use_embeddings` | `true` | Add semantic search to BM25 |

---

## Where things live

```
~/.config/jarvis/config.toml        configuration (0600)
~/.config/jarvis/google_client_secret.json
~/.local/share/jarvis/
├── library.db                      document index (FTS5 + embeddings)
├── state.db                        memory, sessions, conversations
├── google_token.json               OAuth refresh token (0600)
├── skills/                         installed skills
├── briefings/                      saved digests
├── audio/                          spoken files
└── logs/audit.jsonl                every tool call

~/jarvis-workspace/                 the only folder the code agent writes to
```

Everything is plain SQLite, Markdown and AIFF/WAV — back it up, encrypt it, or
delete it without asking anyone. Set `JARVIS_CONFIG_DIR` / `JARVIS_DATA_DIR` to
relocate.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q                     # 123 tests, no network, no model needed
```

```
jarvis_local/
├── cli.py            Typer commands + Rich rendering
├── config.py         TOML schema, validation, persistence
├── hardware.py       detection + the model catalogue + recommendations
├── backends.py       Ollama / OpenAI-compatible / deterministic mock
├── runtime.py        the agent loop, context fitting, approvals
├── agents/           chat, research, code, digest (prompts + data prep)
├── documents.py      SQLite FTS5 + trigram + embeddings, chunking, citations
├── policy.py         path, command and audit rules (one choke point)
├── skills.py         SKILL.md parsing, scanning, install, usage tracking
├── voice.py          local TTS (say / piper / espeak-ng) + transcripts
├── integrations/     Google (read-only Gmail/Calendar) + demo source
└── tools/            files, shell, system, knowledge (search/memory/notes/speech)
```

The mock backend is a first-class citizen: it drives the real tools, retrieval and
digest pipeline, which is how the test suite runs fast and offline.

## Learn more

- [`docs/architecture.md`](docs/architecture.md) — how the agent loop, retrieval and
  safety boundaries fit together.
- [`docs/privacy.md`](docs/privacy.md) — exactly what is stored, what leaves the
  machine, and what the threat model is (and is not).
- [`examples/skills/`](examples/skills/) — a script-backed skill and a prompt-only one.
- [`CHANGELOG.md`](CHANGELOG.md) · [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Limitations, honestly

- Not a security boundary. Approvals reduce accidents; they are not a sandbox.
  Run Jarvis as a normal user, and keep `security.approval = confirm` unless you
  mean it.
- Scanned PDFs and images are not OCR'd.
- Very small models (≤3B) follow multi-step tool instructions unreliably; the
  coding agent is noticeably better at 8B+.
- Google is read-only by design.
- Apple Silicon uses Ollama's Metal backend; Linux GPU acceleration depends on
  the drivers Ollama supports.

## License

MIT — see [LICENSE](LICENSE).
