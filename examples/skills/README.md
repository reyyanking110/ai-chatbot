# Example skills

Two complete skills you can install and study. Both are safe to run: the first
uses only the Python standard library, the second is prompt-only.

```bash
jarvis skills install examples/skills/world-clock --yes
jarvis skills list
jarvis ask "what time is it in Karachi and London?"      # triggers the skill
jarvis skills run world-clock "Karachi, London, Tokyo"   # run the script directly
```

## `world-clock`
A script-backed skill: `SKILL.md` describes when to use it and
`scripts/world_clock.py` produces the answer. Shows the `run:` front-matter key
and how triggers map to real conversation.

## `weekly-review`
A prompt-only skill: instructions that shape how the assistant uses its own
tools (`list_notes`, `read_file`, `take_note`) to produce a structured review.

## Writing your own

```bash
jarvis skills new my-skill -d "What it helps with"
# edit <data>/skills/my-skill/SKILL.md
jarvis skills show my-skill
jarvis skills export my-skill      # share as a zip
```

Front matter fields: `name` (lowercase, dashes), `description`, `version`,
`author`, `triggers` (phrases that activate the skill), and optional `run`
(relative path to a script; Jarvis exposes it as `skill_<name>` behind the usual
approval gate). Everything after the front matter is injected into the prompt
when the skill is relevant.

Shipping a skill:

1. Push a folder with `SKILL.md` to GitHub.
2. `jarvis skills install https://github.com/you/your-skill`
3. For a repo with several skills, list them with `--list` and install one with
   `https://github.com/you/collection#skill-name`.
