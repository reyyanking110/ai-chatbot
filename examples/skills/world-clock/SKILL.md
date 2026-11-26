---
name: world-clock
description: Show the local time in several cities and suggest a polite time to call.
version: 1.0.0
author: jarvis-local
triggers: [world clock, time in, what time, call time, timezone]
run: scripts/world_clock.py
---

# World clock skill

When the user asks for the time somewhere, or when it is a good time to call:

1. Run the `skill_world_clock` tool with the city or region from the request
   (for example, "Karachi and London"). If no place is given, use the cities
   in the user's memory, or ask once.
2. Present the result as a compact list ordered west → east.
3. If the user is trying to schedule a call, add one sentence saying which
   overlapping hours are reasonable for everyone.

The script uses only Python's standard library and needs no network access.
