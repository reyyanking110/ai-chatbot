---
name: weekly-review
description: Turn the week's notes and digests into a short, honest review.
version: 1.0.0
author: jarvis-local
triggers: [weekly review, review my week, retrospective, week]
---

# Weekly review skill

When the user asks for a weekly review:

1. Use `list_notes` and `list_files` to find notes and briefings from the last
   seven days (`notes/` in the workspace and the briefings passed in context).
2. Read the most relevant ones with `read_file`; do not read more than eight files.
3. Produce:
   - **Wins** — three bullets, each tied to a source you actually read.
   - **Loose ends** — anything mentioned but not finished.
   - **One question** — the most useful question for next week.
4. Keep it under 250 words. Do not invent events that are not in the notes.
5. Offer to save the review with `take_note` titled "Weekly review".
