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

MLflow tracking
---------------
Each call to ``fit()`` logs a new run under the ``city-events-recommender``
experiment:
  - Parameters : TF-IDF hyper-params + dataset sizes
  - Metrics    : vocabulary size, matrix sparsity, user vector norm,
                 positive / negative feedback counts, top-5 mean similarity
  - Artifacts  : serialised TF-IDF vectorizer (sklearn model), user vector JSON
  - Model registry: vectorizer registered as
                    ``city-events-recommender-user-<user_id>`` and promoted to
                    the "Production" alias on every successful retrain.
"""
import json
import logging
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from db import database as db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MLflow experiment bootstrap (runs once on import)
# ---------------------------------------------------------------------------

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
_mlflow_client = MlflowClient()


def _registered_model_name(user_id: int) -> str:
    return f"city-events-recommender-user-{user_id}"


def _ensure_registered_model(name: str):
    """Create the registered model entry if it doesn't exist yet."""
    try:
        _mlflow_client.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        _mlflow_client.create_registered_model(
            name,
            description=f"TF-IDF event recommender for {name}",
        )


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _event_text(event: dict) -> str:
    parts = [
        event.get("title") or "",
        event.get("category") or "",
        event.get("description") or "",
        event.get("venue") or "",
    ]
    return " ".join(p for p in parts if p).lower()


def _recency_score(event: dict) -> float:
    try:
        start = datetime.fromisoformat(event["start_dt"].rstrip("Z"))
        delta = (start - datetime.utcnow()).total_seconds() / 86400
        return max(0.0, 1.0 - delta / 14)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class EventRecommender:
    """Stateless recommender; state is read/written via the database."""

    # TF-IDF hyper-parameters (logged to MLflow on every fit)
    NGRAM_RANGE = (1, 2)
    MAX_FEATURES = 5000
    SUBLINEAR_TF = True
    MIN_DF = 1

    def __init__(self, user_id: int):
        self.user_id = user_id
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._event_ids: list[str] = []
        self._user_vector: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, events: list[dict], feedback: list[dict]):
        """Train the recommender and log everything to MLflow."""
        if not events:
            logger.warning("No events available for training user %d", self.user_id)
            return

        self._event_ids = [e["id"] for e in events]
        corpus = [_event_text(e) for e in events]

        self._vectorizer = TfidfVectorizer(
            ngram_range=self.NGRAM_RANGE,
            max_features=self.MAX_FEATURES,
            sublinear_tf=self.SUBLINEAR_TF,
            min_df=self.MIN_DF,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)
        self._user_vector = self._build_user_vector(feedback)

        # ---- MLflow run ------------------------------------------------
        run_id = self._log_to_mlflow(events, feedback)
        # ----------------------------------------------------------------

        # Persist to DB
        matrix_bytes = pickle.dumps(self._matrix)
        user_vec_list = self._user_vector.tolist() if self._user_vector is not None else []
        db.save_model_state(self.user_id, matrix_bytes, self._event_ids, user_vec_list)

        logger.info(
            "Model fitted for user %d: %d events, %d feedback signals  (mlflow run=%s)",
            self.user_id, len(events), len(feedback), run_id,
        )

    def _log_to_mlflow(self, events: list[dict], feedback: list[dict]) -> str:
        """Log a training run and register the model. Returns the run_id."""
        pos_fb = sum(1 for f in feedback if float(f.get("total_weight", 0)) > 0)
        neg_fb = sum(1 for f in feedback if float(f.get("total_weight", 0)) < 0)
        vocab_size = len(self._vectorizer.vocabulary_)
        total_elements = self._matrix.shape[0] * self._matrix.shape[1]
        sparsity = 1.0 - self._matrix.nnz / total_elements if total_elements else 0.0
        user_vec_norm = (
            float(np.linalg.norm(self._user_vector))
            if self._user_vector is not None else 0.0
        )

        model_name = _registered_model_name(self.user_id)
        _ensure_registered_model(model_name)

        with mlflow.start_run(
            run_name=f"user-{self.user_id}-retrain-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        ) as run:
            # --- params ---
            mlflow.log_params({
                "user_id": self.user_id,
                "ngram_range": str(self.NGRAM_RANGE),
                "max_features": self.MAX_FEATURES,
                "sublinear_tf": self.SUBLINEAR_TF,
                "min_df": self.MIN_DF,
                "num_events": len(events),
                "num_feedback_signals": len(feedback),
            })

            # --- metrics ---
            mlflow.log_metrics({
                "vocabulary_size": vocab_size,
                "matrix_sparsity": sparsity,
                "user_vector_norm": user_vec_norm,
                "positive_feedback_count": pos_fb,
                "negative_feedback_count": neg_fb,
                "is_personalised": float(self._user_vector is not None),
            })

            # --- top-5 mean similarity (quality signal) ---
            if self._user_vector is not None:
                sims = cosine_similarity(
                    self._matrix, self._user_vector.reshape(1, -1)
                ).flatten()
                top5_mean = float(np.mean(np.sort(sims)[::-1][:5]))
                mlflow.log_metric("top5_mean_similarity", top5_mean)

            # --- tags ---
            mlflow.set_tags({
                "city": config.CITY,
                "is_personalised": str(self._user_vector is not None),
            })

            # --- sklearn vectorizer artifact + model registry ---
            mlflow.sklearn.log_model(
                self._vectorizer,
                name="tfidf_vectorizer",
                registered_model_name=model_name,
            )

            # --- user preference vector as JSON artifact ---
            if self._user_vector is not None:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(self._user_vector.tolist(), f)
                    tmp_path = f.name
                mlflow.log_artifact(tmp_path, artifact_path="user_vector")
                Path(tmp_path).unlink(missing_ok=True)

            run_id = run.info.run_id

        # Promote this version to "Production" alias in the registry
        try:
            versions = _mlflow_client.search_model_versions(f"name='{model_name}'")
            latest = max(versions, key=lambda v: int(v.version))
            _mlflow_client.set_registered_model_alias(
                model_name, "production", latest.version
            )
            logger.info(
                "MLflow: registered %s v%s → production alias",
                model_name, latest.version,
            )
        except Exception as exc:
            logger.warning("MLflow model registry update failed: %s", exc)

        return run_id

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
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec /= norm
        return user_vec

    # ------------------------------------------------------------------
    # Loading persisted state
    # ------------------------------------------------------------------

    def load(self) -> bool:
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
        candidates = [eid for eid in candidate_event_ids if eid not in already_seen_ids]
        if not candidates:
            return []

        if self._user_vector is not None and self._matrix is not None:
            ranked, scores = self._personalised_rank(candidates, top_n)
            self._log_recommendation_run(ranked, scores, is_coldstart=False)
            return ranked
        else:
            ranked = self._coldstart_rank(candidates, top_n)
            self._log_recommendation_run(ranked, [], is_coldstart=True)
            return ranked

    def _personalised_rank(self, candidates: list[str],
                            top_n: int) -> tuple[list[str], list[float]]:
        id_to_idx = {eid: i for i, eid in enumerate(self._event_ids)}
        idxs, valid = [], []
        for eid in candidates:
            idx = id_to_idx.get(eid)
            if idx is not None:
                idxs.append(idx)
                valid.append(eid)
        if not idxs:
            return candidates[:top_n], []

        candidate_matrix = self._matrix[idxs]
        scores = cosine_similarity(
            candidate_matrix, self._user_vector.reshape(1, -1)
        ).flatten()
        ranked = sorted(zip(valid, scores), key=lambda x: x[1], reverse=True)
        top = ranked[:top_n]
        return [eid for eid, _ in top], [float(s) for _, s in top]

    def _coldstart_rank(self, candidates: list[str], top_n: int) -> list[str]:
        events = {e["id"]: e for e in db.get_all_events()}
        ranked = sorted(
            candidates,
            key=lambda eid: events.get(eid, {}).get("start_dt", "9999"),
        )
        return ranked[:top_n]

    def _log_recommendation_run(self, ranked: list[str],
                                 scores: list[float], is_coldstart: bool):
        """Log a lightweight inference run to MLflow (non-blocking)."""
        try:
            with mlflow.start_run(
                run_name=f"user-{self.user_id}-recommend-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
            ):
                mlflow.set_tag("run_type", "inference")
                mlflow.set_tag("user_id", str(self.user_id))
                mlflow.log_metrics({
                    "num_recommended": len(ranked),
                    "is_coldstart": float(is_coldstart),
                    "mean_similarity_score": float(np.mean(scores)) if scores else 0.0,
                    "max_similarity_score": float(max(scores)) if scores else 0.0,
                })
        except Exception as exc:
            logger.debug("MLflow inference logging failed (non-fatal): %s", exc)


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
    """Return a ranked list of upcoming event dicts for `user_id`."""
    if top_n is None:
        top_n = config.MAX_RECOMMENDATIONS

    rec = EventRecommender(user_id)
    loaded = rec.load()
    if not loaded:
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
