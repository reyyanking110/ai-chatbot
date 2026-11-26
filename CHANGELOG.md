# Changelog

All notable changes to Jarvis Local. This project uses [semantic versioning].

## [0.1.0] — 2026-11-26

First public release. A private, local-first assistant that runs in your terminal.

### Agents
- **chat** — conversational assistant with long-term memory, notes, clock and
  machine info. `remember` / `search_memory` tools persist facts in SQLite.
- **research** — hybrid retrieval (FTS5 BM25 + local embeddings) over your own
  documents with inline `[n]` citations mapped to file and page/line. Refuses to
  invent sources when the answer isn't in your library. Trigram index covers
  Arabic-script languages (Urdu/Arabic), Chinese and Japanese.
- **code** — file, shell and Python tools with a full tool-calling loop. Writes
  are confined to the workspace; mutating commands are previewed and approved;
  verification output is reported honestly.
- **digest** — read-only Gmail + Google Calendar → spoken-style morning briefing,
  saved as Markdown and optionally rendered to audio. Works with demo data until
  Google is connected.

### Local-first by design
- Ollama (default), any OpenAI-compatible local server, or a deterministic mock
  backend for demos and tests. No cloud model calls.
- SQLite for everything: `library.db` (documents, chunks, vectors) and
  `state.db` (memory, sessions, conversations). Delete the files to forget.
- Speech through macOS `say`, `piper` or `espeak-ng`. No cloud TTS.

### Hardware-aware setup
- Detects RAM, CPU, CUDA/ROCm/Metal/Vulkan GPUs and laptop power state.
- A model catalogue with per-role recommendations (general, coding, fast,
  embeddings) that fit the memory budget with context-window headroom.
- `jarvis init` writes config, offers model pulls, indexes documents, installs
  starter skills. `jarvis doctor` verifies everything with concrete fixes.

### Skills
- `SKILL.md` bundles with optional local scripts, installed from GitHub repos,
  archives or folders. Collection repositories supported with `--list` and
  `repo#skill` selection.
- Every install is scanned (scripts, network imports, credential strings, shell
  calls) and confirmed. Scripts never run on their own; calls go through the
  same approval gate as the shell agent.
- Skill usage is tracked and `jarvis skills review` suggests new skills from
  repeated patterns in the audit log.

### Safety
- Writes limited to one workspace folder; reads limited to configured roots.
- Denylist for catastrophic shell patterns; approval modes `confirm`, `auto` and
  `read-only`.
- Python runs with a timeout in a subprocess with secrets stripped from the
  environment.
- Every tool call is appended to `logs/audit.jsonl`.

### Tested
- 100 offline tests (no network, no model downloads): config round-trips, policy
  enforcement, chunking, multilingual retrieval, incremental indexing, the agent
  loop, Ollama wire format against a fake server, skill parsing/scanning/
  installation, memory, speech cleanup and the CLI.

[semantic versioning]: https://semver.org/
