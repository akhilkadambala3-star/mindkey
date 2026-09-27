"""Replay the production ML step over a synthetic session history.

In production every stored session is scored by ``POST /typing/session``: a
per-user Isolation Forest is trained on the user's *earlier* valid sessions
(at least ``MIN_TRAINING_SESSIONS``) and the new session is evaluated against
it. The Demo Lab replays exactly that, in chronological order, so the ML
results a judge sees are produced by the real model rather than typed in.

The model is deterministic (fixed ``random_state``), so the replay is too.
If scikit-learn is unavailable the replay returns ``None`` and callers keep
their fallback values.
"""

import sys
from pathlib import Path

_ML_DIR = Path(__file__).resolve().parents[2] / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

try:
    from anomaly_detector import MIN_TRAINING_SESSIONS, AnomalyDetector
    from features import is_valid_session

    ML_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the runtime
    ML_AVAILABLE = False

#: Label recorded on demo payloads whose ML results come from this replay.
ML_SOURCE = "isolation_forest_replay"


def replay_anomalies(sessions):
    """``session id -> {"anomaly_score", "is_anomaly"}`` for every scorable session.

    Sessions are processed oldest first. A session is scored only when it is
    itself valid and at least ``MIN_TRAINING_SESSIONS`` valid sessions came
    before it, which is the production rule. Returns ``None`` without ML.
    """
    if not ML_AVAILABLE:
        return None
    ordered = sorted(sessions, key=lambda r: (str(r.get("session_start")), str(r.get("id"))))
    results = {}
    history = []
    for row in ordered:
        if is_valid_session(row):
            if len(history) >= MIN_TRAINING_SESSIONS:
                detector = AnomalyDetector()
                detector.train(history)
                outcome = detector.evaluate(row)
                if outcome.get("status") == "ok":
                    results[row["id"]] = {
                        "anomaly_score": round(float(outcome["anomaly_score"]), 4),
                        "is_anomaly": bool(outcome["is_anomaly"]),
                    }
            history.append(row)
    return results
