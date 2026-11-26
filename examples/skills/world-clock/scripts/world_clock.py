#!/usr/bin/env python3
"""World clock — standard library only, no network, no secrets.

Usage:
    python3 world_clock.py "Karachi, London, New York"
"""

from __future__ import annotations

import datetime as dt
import sys
import zoneinfo

CITIES = {
    "karachi": "Asia/Karachi",
    "lahore": "Asia/Karachi",
    "islamabad": "Asia/Karachi",
    "dubai": "Asia/Dubai",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo",
    "singapore": "Asia/Singapore",
    "sydney": "Australia/Sydney",
    "new york": "America/New_York",
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "toronto": "America/Toronto",
    "sao paulo": "America/Sao_Paulo",
    "cape town": "Africa/Johannesburg",
}


def resolve(request: str) -> list[tuple[str, str]]:
    wanted = [part.strip().lower() for part in request.replace(";", ",").split(",") if part.strip()]
    found = [(name, zone) for name, zone in CITIES.items() if any(name == w or name in w for w in wanted)]
    if not found:
        found = [("Karachi", CITIES["karachi"]), ("London", CITIES["london"])]
    seen: set[str] = set()
    unique = []
    for name, zone in found:
        if zone in seen:
            continue
        seen.add(zone)
        unique.append((name.title(), zone))
    return unique


def main() -> int:
    request = " ".join(sys.argv[1:]) or sys.stdin.read()
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for name, zone in resolve(request):
        local = now.astimezone(zoneinfo.ZoneInfo(zone))
        rows.append((local.utcoffset(), name, local))
    rows.sort()
    print(f"UTC now: {now:%Y-%m-%d %H:%M}")
    for offset, name, local in rows:
        hours = offset.total_seconds() / 3600 if offset else 0
        sign = "+" if hours >= 0 else "−"
        print(f"{name:<14} {local:%a %H:%M}  (UTC{sign}{abs(hours):g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
