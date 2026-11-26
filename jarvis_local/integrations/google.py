"""Read-only Gmail + Google Calendar access.

Design choices that keep this local-first and safe:

* Only ``gmail.readonly`` and ``calendar.readonly`` scopes are requested.
  Jarvis cannot send, delete or modify mail or events — even if asked.
* OAuth runs on your machine. ``jarvis google auth`` prints a URL, you paste
  back a code. The refresh token is stored in your data directory with
  ``0600`` permissions and never leaves the device.
* Google responses are summarised locally; message bodies are only fetched
  when the digest is generated.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from ..config import Config
from ..errors import IntegrationError
from ..util import parse_date
from .base import Agenda, CalendarEvent, MailItem

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
]

SETUP_HELP = """\
Google access needs a one-time OAuth client. It takes about three minutes:

  1. Open https://console.cloud.google.com/projectcreate and create a project
     (call it "jarvis-local").
  2. APIs & Services → Library → enable **Gmail API** and **Google Calendar API**.
  3. APIs & Services → OAuth consent screen → External → add your own email as a
     "Test user". (Testing mode is fine; you never need to publish the app.)
  4. Credentials → Create credentials → OAuth client ID → **Desktop app**.
  5. Download the JSON and save it as:

       {secret_path}

  6. Run:  jarvis google auth

Everything stays on this machine; Jarvis only ever reads.
"""


def _import_google():
    try:
        from google.auth.transport.requests import Request  # noqa: F401
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise IntegrationError(
            "Google libraries are not installed.",
            hint="Run: pip install 'jarvis-local[google]'  (or: pip install -e '.[google]')",
        ) from exc
    return Request, Credentials, InstalledAppFlow, build


def google_status(cfg: Config) -> dict:
    secret = cfg.google_client_secret
    token = cfg.google_token_file
    have_libs = True
    try:
        _import_google()
    except IntegrationError:
        have_libs = False
    return {
        "libraries": have_libs,
        "client_secret": secret.exists(),
        "client_secret_path": secret,
        "token": token.exists(),
        "token_path": token,
        "ready": have_libs and secret.exists() and token.exists(),
    }


class GoogleSource:
    name = "google"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._creds = None

    # ------------------------------------------------------------------- auth
    def is_configured(self) -> bool:
        status = google_status(self.cfg)
        return bool(status["libraries"] and status["client_secret"])

    def describe(self) -> str:
        status = google_status(self.cfg)
        if not status["libraries"]:
            return "Google libraries missing (pip install 'jarvis-local[google]')"
        if not status["client_secret"]:
            return "Google not set up — run `jarvis google setup`"
        if not status["token"]:
            return "Authorise with `jarvis google auth`"
        return "Gmail + Calendar connected (read-only)"

    def credentials(self, interactive: bool = False):
        Request, Credentials, InstalledAppFlow, _ = _import_google()
        token_path = self.cfg.google_token_file
        creds = None
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except (ValueError, json.JSONDecodeError):
                creds = None
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self._save_token(creds)
                return creds
            except Exception:  # noqa: BLE001 - token revoked or offline
                creds = None
        if not interactive:
            raise IntegrationError(
                "Google authorisation is missing or expired.",
                hint="Run `jarvis google auth` once in your terminal.",
            )
        secret_path = self.cfg.google_client_secret
        if not secret_path.exists():
            raise IntegrationError(
                "Google client secret file not found.",
                hint=SETUP_HELP.format(secret_path=secret_path),
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), SCOPES)
        creds = self._run_local_flow(flow)
        self._save_token(creds)
        return creds

    def _run_local_flow(self, flow):
        """Prefer a loopback browser flow; fall back to copy/paste everywhere."""
        # ``run_local_server`` opens a browser and listens on localhost. In a
        # terminal-only session we fall back to the console flow automatically.
        try:
            return flow.run_local_server(
                port=0,
                open_browser=True,
                authorization_prompt_message="",
                success_message=oauth_ready_message().replace("<", "&lt;"),
            )
        except Exception:  # noqa: BLE001 - headless / no browser / port blocked
            return flow.run_console()

    def _save_token(self, creds) -> None:
        path: Path = self.cfg.google_token_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(creds.to_json(), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass

    # ------------------------------------------------------------------ fetch
    def fetch(self, horizon_hours: int = 24, max_mails: int = 12) -> Agenda:
        agenda = Agenda(date=dt.date.today(), source="google")
        try:
            creds = self.credentials(interactive=False)
        except IntegrationError as exc:
            agenda.error = exc.message
            return agenda
        try:
            agenda.events = self._calendar(creds, horizon_hours)
        except Exception as exc:  # noqa: BLE001
            agenda.error = f"Calendar: {_friendly_error(exc)}"
        try:
            agenda.mail = self._mail(creds, max_mails)
        except Exception as exc:  # noqa: BLE001
            agenda.error = (agenda.error + "\n" if agenda.error else "") + f"Gmail: {_friendly_error(exc)}"
        return agenda

    def _calendar(self, creds, horizon_hours: int) -> list[CalendarEvent]:
        _, _, _, build = _import_google()
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        now = dt.datetime.now().astimezone()
        start = now - dt.timedelta(minutes=30)
        end = start + dt.timedelta(hours=max(1, horizon_hours))
        result = (
            service.events()
            .list(
                calendarId=self.cfg.google.calendar_id or "primary",
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
                maxResults=40,
            )
            .execute()
        )
        events: list[CalendarEvent] = []
        for item in result.get("items", []):
            if item.get("status") == "cancelled":
                continue
            start_raw = item.get("start", {})
            end_raw = item.get("end", {})
            all_day = "date" in start_raw
            start_dt = parse_date(start_raw.get("dateTime") or start_raw.get("date", ""))
            end_dt = parse_date(end_raw.get("dateTime") or end_raw.get("date", ""))
            attendees = [
                a.get("email", "")
                for a in item.get("attendees", [])
                if a.get("email") and not a.get("self") and a.get("responseStatus") != "declined"
            ]
            events.append(
                CalendarEvent(
                    title=item.get("summary", "(no title)"),
                    start=start_dt,
                    end=end_dt,
                    location=item.get("location", "") or "",
                    attendees=attendees,
                    description=item.get("description", "") or "",
                    all_day=all_day,
                    calendar=item.get("organizer", {}).get("email", "primary"),
                )
            )
        return events

    def _mail(self, creds, max_mails: int) -> list[MailItem]:
        _, _, _, build = _import_google()
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        query = self.cfg.google.mail_query or "is:unread newer_than:1d"
        result = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max(1, min(max_mails, 30)))
            .execute()
        )
        items: list[MailItem] = []
        for ref in result.get("messages", []):
            try:
                msg = (
                    service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=ref["id"],
                        format="metadata",
                        metadataHeaders=["From", "Subject", "Date"],
                    )
                    .execute()
                )
            except Exception:  # noqa: BLE001 - skip individual failures
                continue
            headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
            label_ids = msg.get("labelIds", [])
            items.append(
                MailItem(
                    subject=headers.get("subject", "(no subject)"),
                    sender=headers.get("from", "unknown sender"),
                    snippet=(msg.get("snippet") or "").replace("\n", " ").strip(),
                    received=parse_date(headers.get("date", "")) or _from_millis(msg.get("internalDate")),
                    unread="UNREAD" in label_ids,
                    important="IMPORTANT" in label_ids or "STARRED" in label_ids,
                    labels=[lbl for lbl in label_ids if lbl not in {"UNREAD", "INBOX"}][:4],
                )
            )
        items.sort(key=lambda m: (not m.important, m.received or dt.datetime.min.replace(tzinfo=dt.UTC)), reverse=False)
        return items


def _from_millis(value) -> dt.datetime | None:
    try:
        return dt.datetime.fromtimestamp(int(value) / 1000).astimezone()
    except (TypeError, ValueError, OSError):
        return None


def oauth_ready_message() -> str:
    """A tiny local HTML page shown after the browser callback."""
    return (
        "<html><body style=\"font-family: system-ui; padding: 3rem;\">"
        "<h2>Jarvis is connected 🎉</h2>"
        "<p>You can close this tab and return to the terminal.</p>"
        "<p style=\"color:#666\">Scopes granted: read-only Gmail and Calendar.</p>"
        "</body></html>"
    )


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "insufficientPermissions" in text or "403" in text:
        return "permission denied — re-run `jarvis google auth` and grant read access"
    if "invalid_grant" in text:
        return "token expired or revoked — run `jarvis google auth` again"
    if "HttpError 429" in text or "quota" in text.lower():
        return "Google rate limit hit; try again in a minute"
    return text[:300]
