"""SQLite database layer for city events service."""
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import config

logger = logging.getLogger(__name__)


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id          TEXT PRIMARY KEY,
                source      TEXT NOT NULL,
                title       TEXT NOT NULL,
                description TEXT,
                category    TEXT,
                start_dt    TEXT,
                end_dt      TEXT,
                venue       TEXT,
                url         TEXT,
                image_url   TEXT,
                price_min   REAL,
                price_max   REAL,
                fetched_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                email           TEXT UNIQUE NOT NULL,
                city            TEXT,
                preferences     TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL,
                last_digest_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id),
                event_id    TEXT NOT NULL REFERENCES events(id),
                signal      TEXT NOT NULL,   -- 'click', 'like', 'dislike', 'attend'
                weight      REAL NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS digests (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER NOT NULL REFERENCES users(id),
                sent_at         TEXT NOT NULL,
                event_ids       TEXT NOT NULL,   -- JSON array
                open_count      INTEGER DEFAULT 0,
                click_count     INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS model_state (
                user_id         INTEGER PRIMARY KEY REFERENCES users(id),
                tfidf_matrix    BLOB,            -- pickled scipy sparse matrix
                event_ids       TEXT,            -- JSON ordered list matching rows
                user_vector     TEXT,            -- JSON array (dense preference vec)
                trained_at      TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_event ON feedback(event_id);
            CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_dt);
        """)
    logger.info("Database initialised at %s", config.DB_PATH)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def upsert_events(events: list[dict]):
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO events
               (id, source, title, description, category, start_dt, end_dt,
                venue, url, image_url, price_min, price_max, fetched_at)
               VALUES (:id,:source,:title,:description,:category,:start_dt,:end_dt,
                       :venue,:url,:image_url,:price_min,:price_max,:fetched_at)
               ON CONFLICT(id) DO UPDATE SET
                   title=excluded.title,
                   description=excluded.description,
                   start_dt=excluded.start_dt,
                   end_dt=excluded.end_dt,
                   fetched_at=excluded.fetched_at
            """,
            events,
        )


def get_upcoming_events(days_ahead: int = 14) -> list[dict]:
    now = datetime.utcnow().isoformat()
    cutoff = datetime.utcnow()
    from datetime import timedelta
    cutoff = (datetime.utcnow() + timedelta(days=days_ahead)).isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE start_dt >= ? AND start_dt <= ? ORDER BY start_dt",
            (now, cutoff),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_events() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM events ORDER BY start_dt").fetchall()
    return [dict(r) for r in rows]


def get_event(event_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_or_create_user(email: str, city: str | None = None) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if row:
            return dict(row)
        conn.execute(
            "INSERT INTO users (email, city, created_at) VALUES (?,?,?)",
            (email, city or config.CITY, datetime.utcnow().isoformat()),
        )
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row)


def get_all_users() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()
    return [dict(r) for r in rows]


def update_user_preferences(user_id: int, prefs: dict):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET preferences=? WHERE id=?",
            (json.dumps(prefs), user_id),
        )


def mark_digest_sent(user_id: int, event_ids: list[str]):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO digests (user_id, sent_at, event_ids) VALUES (?,?,?)",
            (user_id, datetime.utcnow().isoformat(), json.dumps(event_ids)),
        )
        conn.execute(
            "UPDATE users SET last_digest_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), user_id),
        )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

SIGNAL_WEIGHTS = {
    "click": 0.5,
    "like": 1.0,
    "attend": 1.5,
    "dislike": -1.0,
}


def record_feedback(user_id: int, event_id: str, signal: str):
    weight = SIGNAL_WEIGHTS.get(signal, 0.5)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (user_id, event_id, signal, weight, created_at) VALUES (?,?,?,?,?)",
            (user_id, event_id, signal, weight, datetime.utcnow().isoformat()),
        )


def get_user_feedback(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT f.event_id, SUM(f.weight) as total_weight, f.signal
               FROM feedback f
               WHERE f.user_id=?
               GROUP BY f.event_id""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

def save_model_state(user_id: int, tfidf_matrix_bytes: bytes,
                     event_ids: list[str], user_vector: list[float]):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO model_state (user_id, tfidf_matrix, event_ids, user_vector, trained_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET
                   tfidf_matrix=excluded.tfidf_matrix,
                   event_ids=excluded.event_ids,
                   user_vector=excluded.user_vector,
                   trained_at=excluded.trained_at""",
            (user_id, tfidf_matrix_bytes, json.dumps(event_ids),
             json.dumps(user_vector), datetime.utcnow().isoformat()),
        )


def load_model_state(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM model_state WHERE user_id=?", (user_id,)
        ).fetchone()
    return dict(row) if row else None
