#!/usr/bin/env python3
"""
City Events ML Service — entry point.

Usage
-----
# Run everything (API server + scheduler) in one process:
    python main.py serve

# Send the digest right now (for testing):
    python main.py digest --email you@example.com

# Add / register a user:
    python main.py add-user --email you@example.com --city "San Francisco"

# Fetch & store fresh events:
    python main.py fetch-events

# Retrain ML models for all users:
    python main.py retrain

# Show next scheduled runs:
    python main.py status
"""
import argparse
import logging
import sys
import threading

import config
from db import database as db
from events import fetcher
from ml import recommender
from notifications import email_sender
from scheduler import run_scheduler

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_serve(args):
    """Start both the API server and the scheduler."""
    db.init_db()
    from api.app import run_api

    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True, name="scheduler")
    scheduler_thread.start()
    logger.info("Scheduler started in background thread.")
    logger.info("Starting API server on port %d…", config.API_PORT)
    run_api()   # blocking


def cmd_digest(args):
    """Send the weekly digest immediately (useful for testing)."""
    db.init_db()
    email = args.email or config.EMAIL_TO
    if not email:
        print("Error: provide --email or set EMAIL_TO in .env")
        sys.exit(1)

    user = db.get_or_create_user(email, city=args.city or config.CITY)
    logger.info("Generating digest for %s…", email)

    # Ensure events are loaded
    events_in_db = db.get_all_events()
    if not events_in_db:
        logger.info("No events in DB – fetching now…")
        raw = fetcher.fetch_events()
        db.upsert_events(raw)

    feedback = db.get_user_feedback(user["id"])
    is_personalised = len(feedback) > 0
    recommended = recommender.recommend_for_user(user["id"])

    if not recommended:
        print("No events available.")
        sys.exit(0)

    sent = email_sender.send_digest(user, recommended, is_personalised=is_personalised)
    db.mark_digest_sent(user["id"], [e["id"] for e in recommended])
    print(f"Digest {'sent' if sent else 'failed'} for {email} ({len(recommended)} events).")


def cmd_add_user(args):
    """Register a new user."""
    db.init_db()
    user = db.get_or_create_user(args.email, city=args.city or config.CITY)
    print(f"User registered: id={user['id']}  email={user['email']}  city={user['city']}")


def cmd_fetch_events(args):
    """Fetch and store the latest events."""
    db.init_db()
    events = fetcher.fetch_events(days_ahead=14)
    db.upsert_events(events)
    print(f"Fetched and stored {len(events)} events.")


def cmd_retrain(args):
    """Retrain ML models for all users."""
    db.init_db()
    users = db.get_all_users()
    if not users:
        print("No users registered.")
        return
    for user in users:
        recommender.train_for_user(user["id"])
        print(f"  Retrained model for user {user['id']} ({user['email']})")
    print(f"Done – retrained {len(users)} models.")


def cmd_status(args):
    """Print a summary of the current system state."""
    db.init_db()
    users = db.get_all_users()
    events = db.get_all_events()
    print(f"Users:  {len(users)}")
    print(f"Events: {len(events)}")
    for u in users:
        fb = db.get_user_feedback(u["id"])
        print(f"  User {u['id']} ({u['email']}): {len(fb)} feedback signals, "
              f"last digest: {u.get('last_digest_at') or 'never'}")
    print(f"\nConfig:")
    print(f"  City:          {config.CITY}")
    print(f"  Digest:        every {config.WEEKLY_DAY} at {config.WEEKLY_HOUR:02d}:00 UTC")
    print(f"  API port:      {config.API_PORT}")
    print(f"  Feedback base: {config.FEEDBACK_BASE_URL}")
    print(f"  SMTP:          {'configured' if config.SMTP_USER else 'NOT configured (dev mode)'}")
    print(f"  Ticketmaster:  {'configured' if config.TICKETMASTER_API_KEY else 'NOT configured'}")
    print(f"  Eventbrite:    {'configured' if config.EVENTBRITE_TOKEN else 'NOT configured'}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="City Events ML Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    sub.add_parser("serve", help="Start API server + scheduler")

    # digest
    p_digest = sub.add_parser("digest", help="Send digest immediately")
    p_digest.add_argument("--email", help="Recipient email")
    p_digest.add_argument("--city", help="City override")

    # add-user
    p_user = sub.add_parser("add-user", help="Register a user")
    p_user.add_argument("--email", required=True)
    p_user.add_argument("--city")

    # fetch-events
    sub.add_parser("fetch-events", help="Fetch and store events")

    # retrain
    sub.add_parser("retrain", help="Retrain ML models")

    # status
    sub.add_parser("status", help="Show system status")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    commands = {
        "serve": cmd_serve,
        "digest": cmd_digest,
        "add-user": cmd_add_user,
        "fetch-events": cmd_fetch_events,
        "retrain": cmd_retrain,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
