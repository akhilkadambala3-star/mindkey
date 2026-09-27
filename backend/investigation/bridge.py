"""The ML <-> Agent bridge (Phase 2).

This module is the single place where stored typing sessions are turned into
canonical, unit-explicit evidence rows.

Design decisions
----------------
- **One notion of "valid session".** Validation is reused from
  ``ml/features.py`` (``is_valid_session`` / ``FEATURE_KEYS``) so the evidence
  layer and the Isolation Forest agree on exactly which sessions count. The
  repository root is added to ``sys.path`` so ``ml`` resolves as a package,
  mirroring ``backend/routes/typing.py``. ``ml/features.py`` has no
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

# Reuse the ML layer's validation as the single source of truth. The ML modules
# live in the repo-root "ml" package and use package-qualified imports, so the
# repository root must be on sys.path for "ml" to resolve as a package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # pragma: no cover - exercised implicitly by every test
    from ml.features import FEATURE_KEYS as ML_FEATURE_KEYS
    from ml.features import is_valid_session as _ml_is_valid_session

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


def _session_duration_seconds(session):
    """Derive the canonical ``session_duration`` feature from stored timestamps.

    Legacy ``typing_sessions`` rows predate the canonical ``session_duration``
    feature and carry only ``session_start``/``session_end``. This mirrors the
    derivation already used by ``backend/routes/typing.py``. Returns None when
    no usable source exists.
    """
    value = session.get("session_duration")
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    start = _as_datetime(session.get("session_start"))
    end = _as_datetime(session.get("session_end"))
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def _with_session_duration(session):
    """Return ``session`` carrying a ``session_duration`` for ML validation.

    A copy is returned so stored rows are never mutated; rows that already
    carry the field are passed through untouched.
    """
    if session.get("session_duration") is not None:
        return session

    duration = _session_duration_seconds(session)
    if duration is None:
        return session

    enriched = dict(session)
    enriched["session_duration"] = duration
    return enriched


def is_valid_stored_session(row):
    """Return True if a stored session row is usable evidence input.

    Delegates to the ML layer's validation so both layers share one definition.
    Legacy rows without the canonical ``session_duration`` feature have it
    derived from their timestamps first (mirroring
    ``backend/routes/typing.py``), so the shared definition applies uniformly.
    """
    if not isinstance(row, dict):
        return False
    return bool(_ml_is_valid_session(_with_session_duration(row)))


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
