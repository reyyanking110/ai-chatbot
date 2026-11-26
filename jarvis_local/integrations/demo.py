"""Deterministic sample agenda so the morning digest works offline."""

from __future__ import annotations

import datetime as dt
import hashlib

from .base import Agenda, CalendarEvent, MailItem


class DemoSource:
    name = "demo"

    def is_configured(self) -> bool:
        return True

    def describe(self) -> str:
        return "Demo data (no Gmail/Calendar connection needed)"

    def fetch(self, horizon_hours: int = 24, max_mails: int = 12) -> Agenda:
        now = dt.datetime.now().astimezone()
        today = now.date()
        seed = int(hashlib.sha1(today.isoformat().encode()).hexdigest()[:6], 16)
        base = now.replace(minute=0, second=0, microsecond=0)
        tomorrow = {"Friday": "Project review", "Monday": "Weekly planning"}.get(
            now.strftime("%A"), "Focus block"
        )

        def at(hour: int, minute: int = 0) -> dt.datetime:
            return (base + dt.timedelta(days=1)).replace(hour=hour, minute=minute)

        events = [
            CalendarEvent("Morning stand-up", at(9, 15), at(9, 30), location="Meet", attendees=["team@example.com"]),
            CalendarEvent(f"{tomorrow} — design sync", at(11), at(12), location="Zoom", attendees=["ada@example.com", "sam@example.com"]),
            CalendarEvent("Lunch with Priya", at(13), at(14), location="Noodle House"),
            CalendarEvent("Deep work: Q4 roadmap", at(15), at(17)),
            CalendarEvent("Dentist", at(17, 30), at(18, 15), location="Clifton Dental"),
        ]
        if seed % 2:
            events.insert(2, CalendarEvent("1:1 with manager", at(10, 30), at(10, 55), location="Office"))

        mail = [
            MailItem(
                subject="Re: Q4 roadmap — needs your sign-off by Thursday",
                sender="Ada Lovelace <ada@example.com>",
                snippet="The revised scope looks good. Two open questions on pricing before we lock it...",
                received=base - dt.timedelta(hours=2),
                important=True,
            ),
            MailItem(
                subject="Invoice INV-2041 is due tomorrow",
                sender="Billing <billing@example.com>",
                snippet="A friendly reminder that invoice INV-2041 (PKR 42,000) is due on the 27th.",
                received=base - dt.timedelta(hours=5),
            ),
            MailItem(
                subject="Standup notes + action items",
                sender="Sam Rivera <sam@example.com>",
                snippet="Summarised yesterday: API migration unblocked, design review moved to Thursday...",
                received=base - dt.timedelta(hours=11),
            ),
            MailItem(
                subject="Your flight to Islamabad is confirmed",
                sender="Travel <no-reply@example-travel.com>",
                snippet="Booking ref 7QK2ZP. Check-in opens 24 hours before departure.",
                received=base - dt.timedelta(hours=20),
                unread=False,
            ),
        ]
        return Agenda(events=events[: max(3, min(len(events), 8))], mail=mail[:max_mails], date=today, source="demo")
