"""
Isolation Forest anomaly detection for MindKey.

Purpose
-------
Detect sessions that are unusual relative to a user's own historical
behavior.

Important:
This is behavioral anomaly detection, not medical diagnosis.

The detector intentionally focuses on one user's historical feature
distribution. Population-level assumptions are not used here.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from ml.features import validate_features
import joblib

from ml.config import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_ESTIMATORS,
    ISOLATION_FOREST_RANDOM_STATE,
    MINIMUM_BASELINE_SESSIONS,
)

from ml.features import (
    FEATURE_KEYS,
    is_valid_session,
)


# Backwards-compatible name used by the existing tests.
MIN_TRAINING_SESSIONS = MINIMUM_BASELINE_SESSIONS

CONTAMINATION = ISOLATION_FOREST_CONTAMINATION


class InsufficientDataError(Exception):
    """Raised when there are not enough valid sessions for training."""


class AnomalyDetector:
    """
    Per-user Isolation Forest anomaly detector.

    Training:
        Historical valid sessions are used to construct the user's
        behavioral reference distribution.

    Evaluation:
        A new session is scored against that learned distribution.

    Score convention:
        sklearn's score_samples() is higher for more normal observations.
        MindKey negates that score so larger anomaly_score means
        greater behavioral unusualness.

    The normalized anomaly score is scaled to [0, 1] relative to the
    training distribution.
    """

    def __init__(
        self,
        contamination=CONTAMINATION,
        random_state=ISOLATION_FOREST_RANDOM_STATE,
    ):
        self._contamination = contamination
        self._random_state = random_state

        self._model = None

        self._train_anomaly_min = None
        self._train_anomaly_max = None

        self.available_sessions = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, sessions):
        """
        Train the detector on historical sessions.

        Invalid sessions are ignored.

        Returns
        -------
        dict
            Training status and number of valid samples used.
        """

        valid_sessions = [
            session
            for session in sessions
            if is_valid_session(session)
        ]

        self.available_sessions = len(
            valid_sessions
        )

        if (
            self.available_sessions
            < MIN_TRAINING_SESSIONS
        ):
            raise InsufficientDataError(
                f"Need at least "
                f"{MIN_TRAINING_SESSIONS} valid sessions "
                f"for training; received "
                f"{self.available_sessions}."
            )

        X = self._to_feature_matrix(
            valid_sessions
        )

        self._model = IsolationForest(
            n_estimators=(
                ISOLATION_FOREST_ESTIMATORS
            ),
            contamination=self._contamination,
            random_state=self._random_state,
        )

        self._model.fit(X)

        raw_scores = self._model.score_samples(X)

        # Higher values should represent more anomalous behavior.
        anomaly_scores = -raw_scores

        self._train_anomaly_min = float(
            np.min(anomaly_scores)
        )

        self._train_anomaly_max = float(
            np.max(anomaly_scores)
        )

        return {
            "status": "trained",
            "sample_count": self.available_sessions,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, session):
        """
        Evaluate one session against the trained model.

        Returns
        -------
        dict
            Structured anomaly result.
        """

        if self._model is None:
            return {
                "status": "insufficient_data",
                "message": (
                    "The anomaly detector has not "
                    "been trained yet."
                ),
            }

        if not is_valid_session(session):
            return {
                "status": "invalid_session",
                "message": (
                    "Session failed ML validation."
                ),
            }

        X = self._to_feature_matrix(
            [session]
        )

        raw_score = float(
            self._model.score_samples(X)[0]
        )

        anomaly_score = -raw_score

        normalized_score = (
            self._normalize_score(
                anomaly_score
            )
        )

        decision = float(
            self._model.decision_function(X)[0]
        )

        is_anomaly = decision < 0

        return {
            "status": "ok",
            "anomaly_score": anomaly_score,
            "normalized_anomaly_score": (
                normalized_score
            ),
            "is_anomaly": is_anomaly,
            "message": (
                "Session is behaviorally unusual "
                "relative to the user's training "
                "distribution."
                if is_anomaly
                else
                "Session is within the learned "
                "behavioral distribution."
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_feature_matrix(self, sessions):
        """
        Convert sessions into a matrix using the canonical feature order.
        """

        return np.array(
            [
                [
                    float(session[key])
                    for key in FEATURE_KEYS
                ]
                for session in sessions
            ],
            dtype=float,
        )

        # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save the trained detector to disk.
        """

        if self._model is None:
            raise ValueError(
                "Cannot save an untrained detector."
            )

        artifact = {
            "model": self._model,
            "train_anomaly_min": self._train_anomaly_min,
            "train_anomaly_max": self._train_anomaly_max,
            "available_sessions": self.available_sessions,
            "feature_keys": FEATURE_KEYS,
        }

        joblib.dump(
            artifact,
            path,
        )

    @classmethod
    def load(cls, path):
        """
        Load a previously trained detector from disk.
        """

        artifact = joblib.load(path)

        if artifact["feature_keys"] != FEATURE_KEYS:
            raise ValueError(
                "Saved model feature schema does not match "
                "the current feature schema."
            )

        detector = cls()

        detector._model = artifact["model"]

        detector._train_anomaly_min = (
            artifact["train_anomaly_min"]
        )

        detector._train_anomaly_max = (
            artifact["train_anomaly_max"]
        )

        detector.available_sessions = (
            artifact["available_sessions"]
        )

        return detector

    
    def _normalize_score(self, anomaly_score):
        """
        Normalize an anomaly score to [0, 1].

        Normalization is relative to the training distribution.
        """

        if (
            self._train_anomaly_min is None
            or self._train_anomaly_max is None
        ):
            return 0.0

        score_range = (
            self._train_anomaly_max
            - self._train_anomaly_min
        )

        if score_range <= 0:
            return 0.0

        normalized = (
            anomaly_score
            - self._train_anomaly_min
        ) / score_range

        return float(
            np.clip(
                normalized,
                0.0,
                1.0,
            )
        )