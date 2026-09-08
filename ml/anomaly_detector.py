"""Isolation-Forest anomaly detection for MindKey.

This module trains a per-user model on valid historical typing sessions and
evaluates a new session for deviation from that user's normal typing
behavior.

Non-goals (explicit):
- No medical diagnosis. This module never maps an anomaly to any disease
  (Parkinson's, Alzheimer's, dementia, ...). It only measures deviation from
  an individual's own typing pattern.
- No persistence layer and no multi-signal risk assessment yet. Extension
  points are marked with "EXTENSION POINT" comments so those features can be
  added later without restructuring this module.

Score convention (IMPORTANT):
- sklearn's ``IsolationForest.score_samples()`` returns a raw score where
  HIGHER = MORE NORMAL and LOWER = MORE ANOMALOUS.
- MindKey exposes an application-level ``anomaly_score`` where HIGHER =
  MORE ANOMALOUS. We therefore transform the raw model output:
      anomaly_score_raw = -score_samples()
  and min-max normalize it to [0, 1] against the distribution of the user's
  own training sessions. The exposed ``anomaly_score`` is thus
  "more anomalous = higher score", relative to that user's normal range.
- ``is_anomaly`` is derived from the model's ``decision_function``: a value
  below 0 marks the session as an outlier.

Modules in this package use flat imports (``from features import ...``) and
are intended to be run from the ``ml/`` directory, matching the convention
used by the ``backend/`` and ``agent/`` packages.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from features import FEATURE_KEYS, is_valid_session

#: Minimum number of valid historical sessions required to train a model.
#: 10 valid sessions is a chosen prototype minimum, because very small
#: samples (e.g. the current test user's 3 sessions) provide only a weak
#: representation of an individual's normal typing behavior. With fewer
#: sessions we refuse to train and report an "insufficient data" status.
MIN_TRAINING_SESSIONS = 10

#: Expected share of anomalous sessions in a user's history under normal
#: behavior. Prototype assumption used to set the Isolation Forest's
#: contamination; documented rather than hidden.
CONTAMINATION = 0.1


class InsufficientDataError(Exception):
    """Raised when a user has fewer valid sessions than MIN_TRAINING_SESSIONS.

    Callers translate this into a clear "insufficient data" status for the
    user, including how many sessions are available and how many are needed.
    """


class AnomalyDetector:
    """Detects deviation of a typing session from a user's normal behavior.

    Attributes:
        available_sessions: number of valid sessions seen by the most recent
            ``train()`` call (recorded even when training fails, so callers
            can report how much data exists).
    """

    def __init__(self, contamination=CONTAMINATION, random_state=42):
        self._contamination = contamination
        self._random_state = random_state
        self._model = None
        # Bounds of the transformed training scores, used to min-max
        # normalize anomaly_score to [0, 1] against the user's own range.
        self._train_anomaly_min = None
        self._train_anomaly_max = None
        self.available_sessions = 0

    @property
    def is_trained(self):
        """True once ``train()`` has successfully fitted a model."""
        return self._model is not None

    def train(self, sessions):
        """Fit the Isolation Forest on valid historical sessions.

        Args:
            sessions: iterable of typing-session dicts for one user, as
                stored by the backend. Invalid sessions are filtered out via
                ``is_valid_session``.

        Returns:
            A summary dict with ``status`` ("trained") and ``sample_count``.

        Raises:
            InsufficientDataError: if fewer than ``MIN_TRAINING_SESSIONS``
                valid sessions are available. ``available_sessions`` is set
                on the detector either way so callers can report the count.

        Notes:
            - The model learns what "normal" looks like for THIS user from
              their own history only; it never compares users to each other.
            - ``random_state`` is fixed so results are reproducible.
        """
        valid_sessions = [s for s in sessions if is_valid_session(s)]
        self.available_sessions = len(valid_sessions)

        if self.available_sessions < MIN_TRAINING_SESSIONS:
            raise InsufficientDataError(
                f"Only {self.available_sessions} valid session(s) available; "
                f"at least {MIN_TRAINING_SESSIONS} valid historical sessions "
                "are required to train (a chosen prototype minimum: very "
                "small samples provide only a weak representation of an "
                "individual's normal typing behavior)."
            )

        feature_matrix = self._to_feature_matrix(valid_sessions)

        model = IsolationForest(
            n_estimators=100,
            contamination=self._contamination,
            random_state=self._random_state,
        )
        model.fit(feature_matrix)

        # Record the transformed-score distribution of the training set so
        # future sessions can be min-max normalized to [0, 1] against the
        # user's own normal range. The transformation flips the sklearn
        # convention (score_samples: higher = more normal) so that our
        # anomaly_score is higher = more anomalous.
        training_raw_scores = -model.score_samples(feature_matrix)
        self._train_anomaly_min = float(np.min(training_raw_scores))
        self._train_anomaly_max = float(np.max(training_raw_scores))
        self._model = model

        # EXTENSION POINT (persistence): persist self._model (e.g. with
        # joblib) keyed by user_id here, and load an existing model instead
        # of retraining when a persistence check finds one.
        return {
            "status": "trained",
            "sample_count": self.available_sessions,
            "message": f"Model trained on {self.available_sessions} valid sessions.",
        }

    def evaluate(self, session):
        """Score a single new typing session against the trained model.

        Args:
            session: a typing-session dict, as stored by the backend.

        Returns:
            A dict:
            - ``status``: "ok" when a model is trained and the session is
              valid; "insufficient_data" when no model has been trained yet
              (not enough historical data); "invalid_session" when the
              session fails validation.
            - ``anomaly_score``: float in [0, 1], HIGHER = MORE ANOMALOUS,
              min-max normalized against the user's training range.
            - ``anomaly_score_raw``: the untransformed ``-score_samples()``
              value, kept for debugging/verification of the transformation.
            - ``is_anomaly``: bool, True when the session deviates from the
              user's normal typing behavior.
            - ``message`` plus, for the insufficient-data status,
              ``available_sessions`` and ``required_sessions``.
        """
        if not self.is_trained:
            return {
                "status": "insufficient_data",
                "available_sessions": self.available_sessions,
                "required_sessions": MIN_TRAINING_SESSIONS,
                "message": (
                    "Not enough valid typing sessions to build a reliable "
                    f"model. At least {MIN_TRAINING_SESSIONS} valid "
                    "historical sessions are required (a chosen prototype "
                    "minimum: very small samples provide only a weak "
                    "representation of an individual's normal typing "
                    "behavior)."
                ),
            }

        if not is_valid_session(session):
            return {
                "status": "invalid_session",
                "message": (
                    "Session is missing required features or has values "
                    "outside the expected ranges."
                ),
            }

        feature_vector = self._to_feature_matrix([session])
        # score_samples(): higher = more normal. Negate so higher = more
        # anomalous, then normalize to [0, 1] against the user's own range.
        raw_anomaly_score = float(-self._model.score_samples(feature_vector)[0])
        anomaly_score = self._normalize_anomaly_score(raw_anomaly_score)
        # decision_function(): values below 0 mark outliers.
        is_anomaly = bool(self._model.decision_function(feature_vector)[0] < 0)

        # EXTENSION POINT (multi-signal risk assessment): combine this
        # typing-anomaly signal with other signals (e.g. survey responses,
        # appointment data) in a later risk-assessment layer.
        return {
            "status": "ok",
            "anomaly_score": anomaly_score,
            "anomaly_score_raw": raw_anomaly_score,
            "is_anomaly": is_anomaly,
            "message": "Anomalous session" if is_anomaly else "Normal session",
        }

    def _normalize_anomaly_score(self, raw_anomaly_score):
        """Min-max normalize a transformed score to [0, 1] against training.

        Bounds come from the user's own training distribution, so the score
        expresses how anomalous a session is relative to that user's normal
        range. Values outside the training range clamp to 0.0 / 1.0.

        Degenerate case: if all training sessions were identical the training
        range has zero span, and any session that differs at all is scored as
        maximally anomalous (1.0); identical sessions score 0.0.
        """
        if self._train_anomaly_min is None or self._train_anomaly_max is None:
            return None

        span = self._train_anomaly_max - self._train_anomaly_min
        if span <= 0:
            return 1.0 if raw_anomaly_score > self._train_anomaly_max else 0.0

        normalized = (raw_anomaly_score - self._train_anomaly_min) / span
        return float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def _to_feature_matrix(sessions):
        """Build the fixed-order float feature matrix from session dicts."""
        return np.array(
            [[session[key] for key in FEATURE_KEYS] for session in sessions],
            dtype=float,
        )