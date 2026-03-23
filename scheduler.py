"""
APScheduler-based job runner.

Jobs
----
weekly_digest   – every Monday at 09:00 (configurable): fetch events,
                  run ML recommendations, send email digests.
refresh_events  – every 3 days: pull fresh events into the database.
retrain_models  – every day at 03:00: retrain all user models with
                  accumulated feedback.
"""
import logging
import threading

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import config
from db import database as db
from events import fetcher
from ml import recommender
from notifications import email_sender

logger = logging.getLogger(__name__)
_scheduler = BlockingScheduler(timezone="UTC")


# ---------------------------------------------------------------------------
# Job implementations
# ---------------------------------------------------------------------------

def job_refresh_events():
    """Fetch latest events from APIs and upsert into the database."""
    logger.info("[job] Refreshing events…")
    events = fetcher.fetch_events(days_ahead=14)
    db.upsert_events(events)
    logger.info("[job] Refreshed %d events", len(events))


def job_retrain_models():
    """Retrain recommendation models for all users."""
    logger.info("[job] Retraining models…")
    users = db.get_all_users()
    for user in users:
        try:
            recommender.train_for_user(user["id"])
        except Exception as exc:
            logger.error("[job] Retrain failed for user %d: %s", user["id"], exc)
    logger.info("[job] Retrain complete for %d users", len(users))


def job_weekly_digest():
    """
    Main weekly job: for every subscribed user, generate personalised
    event recommendations and send the digest email.
    """
    logger.info("[job] Running weekly digest…")

    # Ensure events are fresh
    job_refresh_events()

    users = db.get_all_users()
    if not users:
        logger.warning("[job] No users registered – skipping digest.")
        return

    for user in users:
        try:
            feedback = db.get_user_feedback(user["id"])
            is_personalised = len(feedback) > 0

            events = recommender.recommend_for_user(user["id"])
            if not events:
                logger.warning("[job] No events for user %d", user["id"])
                continue

            sent = email_sender.send_digest(user, events, is_personalised=is_personalised)
            if sent:
                db.mark_digest_sent(user["id"], [e["id"] for e in events])

        except Exception as exc:
            logger.error("[job] Digest failed for user %d: %s", user["id"], exc)

    logger.info("[job] Weekly digest complete for %d users", len(users))


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def setup_scheduler():
    day_of_week = {
        "monday": "mon", "tuesday": "tue", "wednesday": "wed",
        "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun",
    }.get(config.WEEKLY_DAY.lower(), "mon")

    _scheduler.add_job(
        job_weekly_digest,
        CronTrigger(day_of_week=day_of_week, hour=config.WEEKLY_HOUR, minute=0),
        id="weekly_digest",
        replace_existing=True,
    )
    _scheduler.add_job(
        job_refresh_events,
        CronTrigger(day="*/3", hour=2, minute=0),
        id="refresh_events",
        replace_existing=True,
    )
    _scheduler.add_job(
        job_retrain_models,
        CronTrigger(hour=3, minute=0),
        id="retrain_models",
        replace_existing=True,
    )
    logger.info(
        "Scheduler configured: digest every %s at %02d:00 UTC",
        config.WEEKLY_DAY, config.WEEKLY_HOUR,
    )


def run_scheduler():
    """Start the blocking scheduler (call in a dedicated thread or process)."""
    setup_scheduler()
    logger.info("Starting scheduler…")
    _scheduler.start()
