"""
Flask feedback API.

Endpoints
---------
GET  /feedback/<user_id>/<event_id>/<signal>
     Record a like / dislike / attend signal, retrain model, redirect to a
     confirmation page.

GET  /track/click/<user_id>/<event_id>
     Record a click signal then redirect to ?redirect= param (the event URL).

GET  /health          Health check.
GET  /unsubscribe/<user_id>   (stub)
GET  /preferences/<user_id>   (stub)
"""
import logging
import threading
from urllib.parse import unquote

from flask import Flask, jsonify, redirect, request, url_for

import config
from db import database as db
from ml import recommender

logger = logging.getLogger(__name__)
app = Flask(__name__)

# We retrain in a background thread so the redirect is instant.
_retrain_lock = threading.Lock()


def _async_retrain(user_id: int):
    def _work():
        with _retrain_lock:
            try:
                recommender.train_for_user(user_id)
                logger.info("Background retrain complete for user %d", user_id)
            except Exception as exc:
                logger.error("Background retrain failed for user %d: %s", user_id, exc)
    t = threading.Thread(target=_work, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Feedback recording
# ---------------------------------------------------------------------------

@app.route("/feedback/<int:user_id>/<event_id>/<signal>")
def record_feedback(user_id: int, event_id: str, signal: str):
    valid_signals = {"like", "dislike", "attend"}
    if signal not in valid_signals:
        return jsonify({"error": f"Unknown signal '{signal}'"}), 400

    event = db.get_event(event_id)
    if not event:
        return jsonify({"error": "Event not found"}), 404

    db.record_feedback(user_id, event_id, signal)
    logger.info("Feedback recorded: user=%d event=%s signal=%s", user_id, event_id, signal)

    # Retrain ML model asynchronously
    _async_retrain(user_id)

    emoji = {"like": "👍", "dislike": "👎", "attend": "🎉"}.get(signal, "✓")
    message = {
        "like": "Got it! We'll recommend more events like this.",
        "dislike": "Noted! We'll show fewer events like this.",
        "attend": "Awesome! Enjoy the event!",
    }.get(signal, "Feedback recorded.")

    return f"""
    <html><body style="font-family:sans-serif;text-align:center;padding:60px">
      <div style="font-size:64px">{emoji}</div>
      <h2 style="margin:16px 0">{message}</h2>
      <p style="color:#71717a">Your recommendations will improve for next week's digest.</p>
      <p style="margin-top:24px">
        <strong>{event['title']}</strong>
      </p>
    </body></html>
    """, 200


# ---------------------------------------------------------------------------
# Click tracking
# ---------------------------------------------------------------------------

@app.route("/track/click/<int:user_id>/<event_id>")
def track_click(user_id: int, event_id: str):
    redirect_url = request.args.get("redirect", "")
    event = db.get_event(event_id)
    if event:
        db.record_feedback(user_id, event_id, "click")
        _async_retrain(user_id)

    if redirect_url:
        return redirect(unquote(redirect_url))
    return jsonify({"status": "click recorded"})


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.route("/unsubscribe/<int:user_id>")
def unsubscribe(user_id: int):
    # In production: set an unsubscribed flag on the user record
    return """
    <html><body style="font-family:sans-serif;text-align:center;padding:60px">
      <h2>You've been unsubscribed</h2>
      <p style="color:#71717a">You will no longer receive weekly digests.</p>
    </body></html>
    """, 200


@app.route("/preferences/<int:user_id>")
def preferences(user_id: int):
    # Stub: in production render a preference form
    return jsonify({"user_id": user_id, "message": "Preferences management coming soon."})


@app.route("/stats/<int:user_id>")
def stats(user_id: int):
    """Return feedback stats for a user (useful for debugging)."""
    feedback = db.get_user_feedback(user_id)
    return jsonify({
        "user_id": user_id,
        "feedback_count": len(feedback),
        "feedback": feedback,
    })


def run_api():
    """Start the feedback API server (blocking)."""
    app.run(host="0.0.0.0", port=config.API_PORT, debug=False)
