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
from datetime import datetime
from pathlib import Path

# The ML layer is the repo-root ``ml`` package (``from ml.… import``), exactly
# as ``routes/typing.py`` imports it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from ml.anomaly_detector import MIN_TRAINING_SESSIONS, AnomalyDetector
    from ml.features import is_valid_session

    ML_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the runtime
    ML_AVAILABLE = False


def _with_session_duration(row):
    """Add ``session_duration`` (seconds) from the timestamps, as the
    production typing route does for stored rows that predate the field."""
    if row.get("session_duration") is not None:
        return row
    try:
        start = datetime.fromisoformat(str(row["session_start"]))
        end = datetime.fromisoformat(str(row["session_end"]))
    except (KeyError, TypeError, ValueError):
        return row
    return {**row, "session_duration": max(0.0, (end - start).total_seconds())}


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
    for raw in ordered:
        row = _with_session_duration(raw)
        if is_valid_session(row):
            if len(history) >= MIN_TRAINING_SESSIONS:
                detector = AnomalyDetector()
                detector.train(history)
                outcome = detector.evaluate(row)
                if outcome.get("status") == "ok":
                    # Stored exactly as production stores it: the raw score
                    # (higher = more unusual for this user) and the flag.
                    results[row["id"]] = {
                        "anomaly_score": round(float(outcome["anomaly_score"]), 4),
                        "is_anomaly": bool(outcome["is_anomaly"]),
                    }
            history.append(row)
    return results
