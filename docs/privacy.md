# Privacy and data handling

Jarvis is designed so that the default answer to "where did my data go?" is
"nowhere".

## What leaves your machine

Only actions **you** initiate:

| Action | Destination | What is sent |
| --- | --- | --- |
| `jarvis models pull` | ollama.com / your Ollama daemon | model name only |
| `jarvis skills install <url>` | that URL (GitHub by default) | an HTTP GET |
| a `web_fetch` tool call | the URL the model/user asked for | a GET request |
| `jarvis google auth` / digest | Google APIs | OAuth + read requests |
| Ollama on another host | `ollama.host` you configured | prompts |

Nothing else. There is no analytics, no crash reporting, no update check, no
telemetry endpoint in the codebase.

## What is stored locally

| File | Contents | Protection |
| --- | --- | --- |
| `config.toml` | settings | mode 0600 |
| `library.db` | document text, chunks, embeddings | your filesystem |
| `state.db` | memories, chat history | your filesystem |
| `google_token.json` | OAuth refresh token | mode 0600 |
| `google_client_secret.json` | OAuth client (not a secret in the traditional sense) | mode 0600 |
| `logs/audit.jsonl` | tool calls, redacted arguments | your filesystem |
| `audio/` | synthesised speech | your filesystem |
| `skills/` | installed skills + scripts | your filesystem |
| `~/.jarvis_history` | interactive input history | your home directory |

Back up, encrypt or delete any of these. Deleting `library.db` or `state.db`
permanently forgets the corresponding data — no server copy exists.

## What the model sees

The model runs locally, so the prompt is only in your RAM. Still, be deliberate:

- The research agent puts **retrieved passages** into the prompt — only passages
  relevant to your question, bounded by context size.
- The digest agent puts **calendar titles/times and email headers/snippets** in
  the prompt. Email bodies are not fetched; only Gmail's snippet.
- The chat agent puts **relevant memories** in the prompt.
- The code agent puts **file contents you or the model chose to read** in the
  prompt.

If you point `openai.base_url` at a remote server, those prompts leave the
machine. Jarvis doesn't stop you, but the config and README say "local" for a
reason.

## Google specifics

- Scopes requested: `gmail.readonly`, `calendar.readonly`. The code cannot send,
  delete or modify mail or events, and there is no code path that requests those
  scopes.
- The refresh token lives in your data directory with 0600 permissions.
- `jarvis google auth` uses your own OAuth client, so tokens are issued to you,
  not to the project author.
- Revoke anytime at <https://myaccount.google.com/permissions>.

## Shell and code execution

- Subprocesses get an environment with likely secret variables removed
  (`*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*API_KEY*`, `*AUTH*`, ...).
- The audit log redacts bearer tokens, API keys and private-key blocks.
- Filesystem writes are limited to the workspace; reads are limited to
  configured roots and pass through symlink resolution.

## Threat model (and what it is not)

This is a personal tool that reduces accidents. It is **not** a security sandbox:

- A malicious skill you approve can run code as you.
- A model with `approval = auto` can run any non-denylisted command.
- Prompt injection in a document can try to steer the model; approval gates and
  the denylist limit the blast radius, but review what you index.

Run Jarvis as a normal user (never root), keep approvals on, and prefer the
smallest library scope that answers your questions.
