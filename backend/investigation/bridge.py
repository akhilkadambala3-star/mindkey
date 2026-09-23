"""The ML <-> Agent bridge (Phase 2).

This module is the single place where stored typing sessions are turned into
canonical, unit-explicit evidence rows.

Design decisions
----------------
- **One notion of "valid session".** Validation is reused from
  ``ml/features.py`` (``is_valid_session`` / ``FEATURE_KEYS``) so the evidence
  layer and the Isolation Forest agree on exactly which sessions count. The
  ``ml/`` directory is added to ``sys.path`` using the same guarded approach
  already used by ``backend/routes/typing.py``. ``ml/features.py`` has no
  third-party imports, so this adds no runtime dependency.

- **Mirrored model minimum.** ``MODEL_MINIMUM_SESSIONS`` mirrors
  ``ml/anomaly_detector.MIN_TRAINING_SESSIONS`` (10). It is mirrored as a
  constant rather than imported so that the evidence layer does not pull
  scikit-learn/numpy (which ``anomaly_detector`` imports) into a process that
  only needs to read evidence.

- **No content.** Only the six numeric features, ids, and timestamps are read.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .contracts import CanonicalSession
from .units import STORED_TO_CANONICAL

logger = logging.getLogger(__name__)

#: Mirrors ``ml.anomaly_detector.MIN_TRAINING_SESSIONS`` (a chosen prototype
#: minimum). Mirrored explicitly so reading evidence never imports scikit-learn.
MODEL_MINIMUM_SESSIONS = 10

# Reuse the ML layer's validation as the single source of truth.
_ML_DIR = Path(__file__).resolve().parents[2] / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

try:  # pragma: no cover - exercised implicitly by every test
    from features import FEATURE_KEYS as ML_FEATURE_KEYS
    from features import is_valid_session as _ml_is_valid_session

    ML_VALIDATION_AVAILABLE = True
except ImportError:  # pragma: no cover - ml/ is always present in this repo
    ML_VALIDATION_AVAILABLE = False
    ML_FEATURE_KEYS = (
        "dwell_mean",
        "flight_mean",
        "typing_speed",
        "correction_rate",
        "rhythm_variability",
        "pause_count",
    )

    def _ml_is_valid_session(session):
        """Local fallback mirroring ``ml/features.py:is_valid_session``."""
        for key in ML_FEATURE_KEYS:
            if session.get(key) is None:
                return False
        typing_speed = session["typing_speed"]
        correction_rate = session["correction_rate"]
        if typing_speed <= 0:
            return False
        if correction_rate < 0 or correction_rate > 1:
            return False
        return True


def _as_float(value):
    """Coerce a stored value to float, or None if it is missing/unusable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    """Coerce a stored value to int, or None if it is missing/unusable."""
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _as_datetime(value):
    """Parse a stored timestamp to an aware datetime, or None if unusable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        logger.warning("Unparseable session timestamp; treated as missing")
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def is_valid_stored_session(row):
    """Return True if a stored session row is usable evidence input.

    Delegates to the ML layer's validation so both layers share one definition.
    """
    if not isinstance(row, dict):
        return False
    return bool(_ml_is_valid_session(row))


def to_canonical_session(row):
    """Map a stored session row to a :class:`CanonicalSession`.

    Only the six numeric behavioral features plus ids/timestamps are read; any
    other columns (including any user-authored text) are ignored.
    """
    if not isinstance(row, dict):
        raise TypeError("session row must be a dict")

    session = CanonicalSession(
        session_id=None if row.get("id") is None else str(row.get("id")),
        user_id=None if row.get("user_id") is None else str(row.get("user_id")),
        session_start=_as_datetime(row.get("session_start")),
        session_end=_as_datetime(row.get("session_end")),
        is_valid=is_valid_stored_session(row),
    )

    for stored_key, canonical_key in STORED_TO_CANONICAL.items():
        raw = row.get(stored_key)
        if canonical_key == "pause_count":
            setattr(session, canonical_key, _as_int(raw))
        else:
            setattr(session, canonical_key, _as_float(raw))

    return session


def canonical_feature_value(session, canonical_key):
    """Read a canonical feature value off a :class:`CanonicalSession`."""
    return getattr(session, canonical_key, None)
