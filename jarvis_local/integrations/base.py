"""Common shapes for agenda sources (Gmail/Calendar or the offline demo)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class CalendarEvent:
    title: str
    start: dt.datetime | None
    end: dt.datetime | None = None
    location: str = ""
    attendees: list[str] = field(default_factory=list)
    description: str = ""
    all_day: bool = False
    calendar: str = "primary"

    @property
    def duration_min(self) -> int:
        if not self.start or not self.end:
            return 0
        return max(1, int((self.end - self.start).total_seconds() // 60))


@dataclass
class MailItem:
    subject: str
    sender: str
    snippet: str
    received: dt.datetime | None
    unread: bool = True
    important: bool = False
    labels: list[str] = field(default_factory=list)
    thread_id: str = ""

    @property
    def sender_name(self) -> str:
        name = self.sender.split("<")[0].strip().strip('"')
        return name or self.sender


@dataclass
class Agenda:
    events: list[CalendarEvent] = field(default_factory=list)
    mail: list[MailItem] = field(default_factory=list)
    date: dt.date = field(default_factory=dt.date.today)
    source: str = "demo"
    error: str = ""

    @property
    def busy_minutes(self) -> int:
        return sum(e.duration_min for e in self.events if not e.all_day)

    @property
    def is_empty(self) -> bool:
        return not self.events and not self.mail


class AgendaSource(Protocol):
    name: str

    def is_configured(self) -> bool: ...

    def fetch(self, horizon_hours: int = 24, max_mails: int = 12) -> Agenda: ...

    def describe(self) -> str: ...
