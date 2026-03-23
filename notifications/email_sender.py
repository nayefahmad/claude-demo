"""HTML email sender for the weekly city events digest."""
import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader, select_autoescape

import config
from db import database as db

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_jinja = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)


def _format_date(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value.rstrip("Z"))
        return dt.strftime("%a %b %-d, %-I:%M %p")
    except Exception:
        return value


_jinja.filters["format_date"] = _format_date
_jinja.filters["urlencode"] = quote


def _render_digest(user: dict, events: list[dict],
                   is_personalised: bool = False) -> str:
    template = _jinja.get_template("weekly_digest.html")
    week_label = datetime.utcnow().strftime("%B %-d, %Y")
    return template.render(
        user=user,
        user_id=user["id"],
        events=events,
        city=user.get("city") or config.CITY,
        week_label=week_label,
        feedback_base=config.FEEDBACK_BASE_URL,
        is_personalised=is_personalised,
    )


def send_digest(user: dict, events: list[dict],
                is_personalised: bool = False) -> bool:
    """
    Render and send the weekly digest email to a single user.
    Returns True on success.
    """
    recipient = user["email"]
    if not config.SMTP_USER or not config.SMTP_PASSWORD:
        logger.warning(
            "SMTP credentials not configured – printing digest for %s instead.", recipient
        )
        html = _render_digest(user, events, is_personalised)
        _print_digest_summary(user, events)
        return True   # treat as success in dev mode

    try:
        html = _render_digest(user, events, is_personalised)
        msg = MIMEMultipart("alternative")
        msg["Subject"] = (
            f"Your Weekly City Events Digest – {datetime.utcnow().strftime('%B %-d')}"
        )
        msg["From"] = config.EMAIL_FROM
        msg["To"] = recipient
        text_part = MIMEText(_plain_text(events), "plain")
        html_part = MIMEText(html, "html")
        msg.attach(text_part)
        msg.attach(html_part)

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.sendmail(config.EMAIL_FROM, recipient, msg.as_string())

        logger.info("Digest sent to %s (%d events)", recipient, len(events))
        return True

    except Exception as exc:
        logger.error("Failed to send digest to %s: %s", recipient, exc)
        return False


def _plain_text(events: list[dict]) -> str:
    lines = ["Your Weekly City Events Digest\n", "=" * 40]
    for e in events:
        lines.append(f"\n{e['title']}")
        if e.get("start_dt"):
            lines.append(f"  When: {_format_date(e['start_dt'])}")
        if e.get("venue"):
            lines.append(f"  Where: {e['venue']}")
        if e.get("url"):
            lines.append(f"  Details: {e['url']}")
    lines.append("\n\nPowered by your personal ML recommendations.")
    return "\n".join(lines)


def _print_digest_summary(user: dict, events: list[dict]):
    """Dev-mode: print a readable summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  WEEKLY DIGEST for {user['email']} ({user.get('city', config.CITY)})")
    print(f"  {len(events)} recommended events")
    print(f"{'='*60}")
    for i, e in enumerate(events, 1):
        price = ""
        if e.get("price_min") is not None:
            price = "Free" if e["price_min"] == 0 else f"from ${e['price_min']:.0f}"
        print(f"\n  {i}. {e['title']}")
        print(f"     Category: {e.get('category', 'N/A')}")
        print(f"     When:     {_format_date(e.get('start_dt', ''))}")
        print(f"     Venue:    {e.get('venue', 'TBD')}")
        if price:
            print(f"     Price:    {price}")
        print(f"     URL:      {e.get('url', 'N/A')}")
    print(f"\n{'='*60}\n")
