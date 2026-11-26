"""Optional third-party integrations (all read-only by default)."""

from .base import AgendaSource, CalendarEvent, MailItem
from .demo import DemoSource
from .google import GoogleSource, google_status

__all__ = [
    "AgendaSource",
    "CalendarEvent",
    "MailItem",
    "DemoSource",
    "GoogleSource",
    "google_status",
]
