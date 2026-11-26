"""The morning digest agent: Gmail + Calendar → written and spoken briefing."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..config import Config
from ..errors import IntegrationError
from ..integrations.base import Agenda, CalendarEvent, MailItem
from ..integrations.demo import DemoSource
from ..integrations.google import GoogleSource
from .base import Agent, Prepared, register

DIGEST_RULES = """
# Your role
You write {name}'s morning briefing. It will be **read aloud**, so it must sound natural
when spoken: no markdown tables, no link syntax, no long lists of symbols.

# Structure (keep it under ~350 words)
1. One-sentence framing of the day ("You have {event_count} events and {mail_count} unread
   emails worth a look.").
2. The day, in chronological order: time, what, with whom, and which one needs prep.
   Mention travel/commute concerns and back-to-back stretches.
3. Email triage: only the messages that need action today. For each: sender, subject in
   plain words, and the action. Skip newsletters and noise.
4. "Worth noting": conflicts, deadlines, or anything that needs a decision. Include the
   most useful suggestion for the morning.

# Rules
- Times are 24-hour local time. Say "at 9:15", not "09:15:00".
- Never invent attendees, emails or deadlines that are not in the data.
- If there is nothing important in email, say so in one sentence.
- End with a short, warm one-liner. No sign-off boilerplate.
"""


@dataclass
class AgendaBundle:
    agenda: Agenda
    events: list[CalendarEvent] = field(default_factory=list)
    mail: list[MailItem] = field(default_factory=list)
    text: str = ""
    source: str = "demo"


class DigestAgent(Agent):
    name = "digest"
    title = "Morning digest"
    description = (
        "Pulls today's Gmail and Google Calendar (read-only), writes a spoken-style briefing "
        "and saves it as text + audio."
    )

    def __init__(self) -> None:
        self._cache: dict[str, AgendaBundle] = {}

    def model_for(self, cfg: Config) -> str:
        return cfg.models.fast if cfg.models.fast else cfg.models.general

    def system_prompt(self, ctx) -> str:
        return super().system_prompt(ctx)

    # ------------------------------------------------------------ data fetch
    def fetch_agenda(self, ctx, force_demo: bool = False) -> AgendaBundle:
        force_demo = force_demo or bool(getattr(ctx, "digest_force_demo", False))
        key = f"{force_demo}:{dt.date.today().isoformat()}:{dt.datetime.now().hour}"
        if key in self._cache:
            return self._cache[key]
        source = None
        if not force_demo:
            google = GoogleSource(ctx.cfg)
            if google.is_configured():
                source = google
        if source is None:
            source = DemoSource()
        agenda = source.fetch(
            horizon_hours=ctx.cfg.google.hours_back,
            max_mails=ctx.cfg.google.max_mails,
        )
        bundle = AgendaBundle(
            agenda=agenda,
            events=agenda.events,
            mail=agenda.mail,
            source=source.name,
            text=render_agenda_text(agenda),
        )
        self._cache[key] = bundle
        return bundle

    # -------------------------------------------------------------- prompting
    def prepare(self, ctx, user_input: str) -> Prepared:
        bundle = self.fetch_agenda(ctx)
        system = self.system_prompt(ctx) + DIGEST_RULES.format(
            name=ctx.cfg.assistant_name,
            event_count=len(bundle.events),
            mail_count=len(bundle.mail),
        )
        if bundle.agenda.error:
            system += f"\n# Data caveat\nSome data failed to load: {bundle.agenda.error}\n"
        if bundle.source == "demo":
            system += (
                "\nThe data below is DEMO data (no Google account connected). Write the digest "
                "normally, and add one short spoken sentence at the start saying this is a "
                "sample while Google is not connected.\nIf the user asked for the digest to be "
                "spoken, use the `speak` tool at the end.\n"
            )
        memories = ctx.memory.relevant("preferences briefing morning", limit=4)
        if memories:
            system += "\n# User preferences\n" + "\n".join(f"- {m}" for m in memories)
        user = (
            f"Here is the raw agenda for {bundle.agenda.date.strftime('%A %d %B %Y')}:\n\n"
            f"{bundle.text}\n\n"
            f"Write the spoken morning briefing. The user's message: {user_input}"
        )
        if any(word in user_input.lower() for word in ("read", "aloud", "speak", "voice", "audio")):
            user += "\n\nAfter writing it, call the speak tool with the final script so it is read aloud."
        return Prepared(system=system, user=user)


def render_agenda_text(agenda: Agenda) -> str:
    lines: list[str] = []
    lines.append("CALENDAR EVENTS")
    if agenda.events:
        for event in agenda.events:
            when = "all day" if event.all_day else _time_range(event)
            who = f"  with {', '.join(event.attendees)}" if event.attendees else ""
            where = f"  at {event.location}" if event.location else ""
            lines.append(f"- {when} — {event.title}{where}{who}")
    else:
        lines.append("- (no events)")
    lines.append("")
    lines.append("UNREAD / IMPORTANT EMAIL")
    if agenda.mail:
        for item in agenda.mail:
            flag = "IMPORTANT" if item.important else ("unread" if item.unread else "read")
            received = item.received.strftime("%a %H:%M") if item.received else "unknown time"
            lines.append(f"- [{flag}] {item.sender_name} — {item.subject} ({received})")
            lines.append(f"    preview: {item.snippet[:220]}")
    else:
        lines.append("- (no matching email)")
    lines.append("")
    lines.append(
        f"TODAY: {agenda.date.strftime('%A %d %B %Y')} · "
        f"{len(agenda.events)} events · {agenda.busy_minutes} busy minutes · "
        f"{sum(1 for m in agenda.mail if m.important)} flagged emails"
    )
    return "\n".join(lines)


def _time_range(event: CalendarEvent) -> str:
    if not event.start:
        return "time TBD"
    if not event.end:
        return event.start.strftime("%H:%M")
    if event.start.date() != dt.date.today() and event.start.date() != (dt.date.today() + dt.timedelta(days=1)):
        return event.start.strftime("%a %d %b %H:%M")
    return f"{event.start.strftime('%H:%M')}–{event.end.strftime('%H:%M')}"


def _assert_google_ready(ctx) -> None:
    google = GoogleSource(ctx.cfg)
    if not google.is_configured():
        raise IntegrationError(
            "Google is not configured.",
            hint="Run `jarvis google setup` for the three-minute OAuth walkthrough.",
        )


register(DigestAgent())
