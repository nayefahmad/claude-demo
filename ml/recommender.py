"""
Content-based ML recommender with continuous learning from user feedback.

Pipeline
--------
1. Build a TF-IDF matrix over all stored events (title + description + category).
2. For each user, compute a *preference vector* as the weighted average of
   TF-IDF vectors of events they have positively interacted with (and subtract
   vectors of events they disliked).
3. Rank unseen events by cosine similarity to the preference vector.
4. Fall back to popularity / recency ordering for cold-start users.
5. Persist the trained state to the database so each digest benefits from
   all accumulated feedback.
"""
import json
import logging
import pickle
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import config
from db import database as db

logger = logging.getLogger(__name__)


def _event_text(event: dict) -> str:
    """Combine fields into a single string for TF-IDF."""
    parts = [
        event.get("title") or "",
        event.get("category") or "",
        event.get("description") or "",
        event.get("venue") or "",
    ]
    return " ".join(p for p in parts if p).lower()


def _recency_score(event: dict) -> float:
    """Score 1.0 for today, decaying toward 0 over 14 days."""
    try:
        start = datetime.fromisoformat(event["start_dt"].rstrip("Z"))
        delta = (start - datetime.utcnow()).total_seconds() / 86400
        return max(0.0, 1.0 - delta / 14)
    except Exception:
        return 0.5


class EventRecommender:
    """Stateless recommender; state is read/written via the database."""

    def __init__(self, user_id: int):
        self.user_id = user_id
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None          # (n_events, n_features) sparse matrix
        self._event_ids: list[str] = []
        self._user_vector: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, events: list[dict], feedback: list[dict]):
        """
        Train the recommender on all known events and the user's feedback history.

        Parameters
        ----------
        events:   All events stored in the database.
        feedback: List of {event_id, total_weight} rows for this user.
        """
        if not events:
            logger.warning("No events available for training user %d", self.user_id)
            return

        self._event_ids = [e["id"] for e in events]
        corpus = [_event_text(e) for e in events]

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)  # sparse

        # Build user preference vector from feedback
        self._user_vector = self._build_user_vector(feedback)

        # Persist to DB
        matrix_bytes = pickle.dumps(self._matrix)
        user_vec_list = self._user_vector.tolist() if self._user_vector is not None else []
        db.save_model_state(self.user_id, matrix_bytes, self._event_ids, user_vec_list)
        logger.info(
            "Model fitted for user %d: %d events, %d feedback signals",
            self.user_id, len(events), len(feedback),
        )

    def _build_user_vector(self, feedback: list[dict]) -> np.ndarray | None:
        if not feedback:
            return None

        id_to_idx = {eid: i for i, eid in enumerate(self._event_ids)}
        weighted_sum = np.zeros(self._matrix.shape[1])
        total_weight = 0.0

        for fb in feedback:
            idx = id_to_idx.get(fb["event_id"])
            if idx is None:
                continue
            w = float(fb["total_weight"])
            vec = self._matrix[idx].toarray().flatten()
            weighted_sum += w * vec
            total_weight += abs(w)

        if total_weight == 0:
            return None

        user_vec = weighted_sum / total_weight
        # L2 normalise so cosine sim is just a dot product
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec /= norm
        return user_vec

    # ------------------------------------------------------------------
    # Loading persisted state
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load previously trained state from DB. Returns True on success."""
        state = db.load_model_state(self.user_id)
        if not state or not state.get("tfidf_matrix"):
            return False
        try:
            self._matrix = pickle.loads(state["tfidf_matrix"])
            self._event_ids = json.loads(state["event_ids"])
            user_vec_list = json.loads(state["user_vector"])
            self._user_vector = np.array(user_vec_list) if user_vec_list else None
            logger.info("Loaded persisted model for user %d", self.user_id)
            return True
        except Exception as exc:
            logger.warning("Failed to load model state: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def recommend(self, candidate_event_ids: list[str],
                  already_seen_ids: set[str],
                  top_n: int = 10) -> list[str]:
        """
        Return top-N event IDs from candidates, ranked by personalisation score.
        Falls back to recency-based ranking for cold-start users.
        """
        candidates = [eid for eid in candidate_event_ids if eid not in already_seen_ids]
        if not candidates:
            return []

        if self._user_vector is not None and self._matrix is not None:
            return self._personalised_rank(candidates, top_n)
        else:
            return self._coldstart_rank(candidates, top_n)

    def _personalised_rank(self, candidates: list[str], top_n: int) -> list[str]:
        id_to_idx = {eid: i for i, eid in enumerate(self._event_ids)}
        idxs = []
        valid_candidates = []
        for eid in candidates:
            idx = id_to_idx.get(eid)
            if idx is not None:
                idxs.append(idx)
                valid_candidates.append(eid)

        if not idxs:
            return candidates[:top_n]

        candidate_matrix = self._matrix[idxs]
        scores = cosine_similarity(
            candidate_matrix,
            self._user_vector.reshape(1, -1),
        ).flatten()

        ranked = sorted(zip(valid_candidates, scores),
                        key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in ranked[:top_n]]

    def _coldstart_rank(self, candidates: list[str], top_n: int) -> list[str]:
        """For new users: prefer events starting soonest."""
        events = {e["id"]: e for e in db.get_all_events()}
        ranked = sorted(
            candidates,
            key=lambda eid: events.get(eid, {}).get("start_dt", "9999"),
        )
        return ranked[:top_n]


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def train_for_user(user_id: int):
    """Retrain the recommender for a single user using all available data."""
    all_events = db.get_all_events()
    feedback = db.get_user_feedback(user_id)
    rec = EventRecommender(user_id)
    rec.fit(all_events, feedback)


def recommend_for_user(user_id: int, top_n: int = None) -> list[dict]:
    """
    Return a ranked list of upcoming event dicts for `user_id`.
    Retrains in-place if no persisted model exists.
    """
    if top_n is None:
        top_n = config.MAX_RECOMMENDATIONS

    rec = EventRecommender(user_id)
    loaded = rec.load()
    if not loaded:
        # First run: train now
        train_for_user(user_id)
        rec = EventRecommender(user_id)
        rec.load()

    upcoming = db.get_upcoming_events(days_ahead=14)
    if not upcoming:
        upcoming = db.get_all_events()

    feedback = db.get_user_feedback(user_id)
    seen_ids = {fb["event_id"] for fb in feedback if float(fb["total_weight"]) > 0}

    ranked_ids = rec.recommend(
        candidate_event_ids=[e["id"] for e in upcoming],
        already_seen_ids=seen_ids,
        top_n=top_n,
    )

    id_to_event = {e["id"]: e for e in upcoming}
    return [id_to_event[eid] for eid in ranked_ids if eid in id_to_event]
