"""Deterministic read-only evidence tools (Phase 2).

Each tool has an explicit Pydantic input schema, an explicit Pydantic output
schema, input validation, and structured error handling. Tools never raise on
missing or unavailable data: they return ``available=False`` with an
``unavailable_reason``. Structurally invalid inputs (e.g. ``limit <= 0``) are
rejected up front by Pydantic validation.

Tools are pure with respect to their repository: passing a fake repository
makes them fully deterministic and network-free. All data access goes through
:class:`~investigation.repository.SessionRepository`.

Privacy: only the six numeric behavioral features plus ids/timestamps are read.
No typed content is ever selected or returned.
"""

import logging
from datetime import datetime, timedelta, timezone
from statistics import mean, median, stdev

from pydantic import BaseModel, Field

from .bridge import (
    MODEL_MINIMUM_SESSIONS,
    canonical_feature_value,
    is_valid_stored_session,
    to_canonical_session,
)
from .contracts import CanonicalSession, FeatureStats
from .repository import RepositoryError, default_repository
from .units import CANONICAL_FEATURE_KEYS, label_for, unit_for

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _now():
    return datetime.now(timezone.utc)


def _ensure_aware(value):
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _within(session, start, end):
    """True when a session's start falls in ``[start, end)``.

    Sessions without a parseable timestamp are excluded from windowed math.
    """
    ts = _ensure_aware(session.session_start)
    if ts is None:
        return False
    if start is not None and ts < start:
        return False
    if end is not None and ts >= end:
        return False
    return True


def _load_canonical_sessions(user_id, repository):
    """Fetch and canonicalize all stored sessions for ``user_id``."""
    repo = repository if repository is not None else default_repository()
    rows = repo.list_sessions(user_id)
    return [to_canonical_session(row) for row in (rows or [])]


def _stats_for(canonical_key, sessions):
    """Compute summary statistics for one feature over valid sessions."""
    values = [
        value
        for value in (canonical_feature_value(s, canonical_key) for s in sessions)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not values:
        return FeatureStats(
            key=canonical_key,
            label=label_for(canonical_key),
            canonical_unit=unit_for(canonical_key),
            available=False,
            sample_count=0,
        )
    return FeatureStats(
        key=canonical_key,
        label=label_for(canonical_key),
        canonical_unit=unit_for(canonical_key),
        available=True,
        sample_count=len(values),
        mean=float(mean(values)),
        median=float(median(values)),
        stdev=float(stdev(values)) if len(values) >= 2 else None,
    )


def _stats_by_feature(sessions):
    return [_stats_for(key, sessions) for key in CANONICAL_FEATURE_KEYS]


def _relative_change(baseline_value, recent_value):
    if baseline_value is None or recent_value is None:
        return None
    if baseline_value == 0:
        return None
    return (recent_value - baseline_value) / baseline_value


# ---------------------------------------------------------------------------
# Tool 1: get_ml_evidence
# ---------------------------------------------------------------------------


class MLEvidenceInput(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class MLEvidenceOutput(BaseModel):
    available: bool
    user_id: str
    session_id: str
    status: str | None = None
    anomaly_score: float | None = None
    is_anomaly: bool | None = None
    note: str = ""
    unavailable_reason: str | None = None


def get_ml_evidence(user_id, session_id, repository=None):
    """Retrieve the stored ML anomaly result for one session.

    The anomaly score is returned as a number only. It is never interpreted as
    a condition or a cause.
    """
    params = MLEvidenceInput(user_id=user_id, session_id=session_id)

    try:
        repo = repository if repository is not None else default_repository()
        row = repo.get_anomaly_result(params.session_id)
    except RepositoryError as exc:
        logger.warning("get_ml_evidence: repository unavailable")
        return MLEvidenceOutput(
            available=False,
            user_id=params.user_id,
            session_id=params.session_id,
            unavailable_reason=str(exc) or "repository_unavailable",
        )
    except Exception as exc:  # defensive: never fail the caller
        logger.exception("get_ml_evidence failed unexpectedly")
        return MLEvidenceOutput(
            available=False,
            user_id=params.user_id,
            session_id=params.session_id,
            unavailable_reason=f"unexpected_error: {exc.__class__.__name__}",
        )

    if not row:
        return MLEvidenceOutput(
            available=False,
            user_id=params.user_id,
            session_id=params.session_id,
            note="No stored anomaly result exists for this session.",
            unavailable_reason="no_anomaly_record",
        )

    anomaly_score = row.get("anomaly_score")
    return MLEvidenceOutput(
        available=True,
        user_id=params.user_id,
        session_id=params.session_id,
        status="ok",
        anomaly_score=None if anomaly_score is None else float(anomaly_score),
        is_anomaly=None if row.get("is_anomaly") is None else bool(row.get("is_anomaly")),
        note="Stored ML anomaly result. A score is not a diagnosis.",
    )


# ---------------------------------------------------------------------------
# Tool 2: get_recent_sessions
# ---------------------------------------------------------------------------


class GetRecentSessionsInput(BaseModel):
    user_id: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=500)
    window_days: int | None = Field(default=None, ge=1, le=3650)


class RecentSessionsOutput(BaseModel):
    available: bool
    user_id: str
    count: int = 0
    window_days: int | None = None
    sessions: list[CanonicalSession] = Field(default_factory=list)
    unavailable_reason: str | None = None


def get_recent_sessions(user_id, limit=20, window_days=None, repository=None):
    """Return the most recent sessions, oldest first, in canonical units."""
    params = GetRecentSessionsInput(user_id=user_id, limit=limit, window_days=window_days)

    try:
        sessions = _load_canonical_sessions(params.user_id, repository)
    except RepositoryError as exc:
        logger.warning("get_recent_sessions: repository unavailable")
        return RecentSessionsOutput(
            available=False,
            user_id=params.user_id,
            window_days=params.window_days,
            unavailable_reason=str(exc) or "repository_unavailable",
        )
    except Exception as exc:
        logger.exception("get_recent_sessions failed unexpectedly")
        return RecentSessionsOutput(
            available=False,
            user_id=params.user_id,
            window_days=params.window_days,
            unavailable_reason=f"unexpected_error: {exc.__class__.__name__}",
        )

    if params.window_days is not None:
        cutoff = _now() - timedelta(days=params.window_days)
        sessions = [s for s in sessions if _within(s, cutoff, None)]

    floor = datetime.min.replace(tzinfo=timezone.utc)
    sessions.sort(key=lambda s: _ensure_aware(s.session_start) or floor)
    sessions = sessions[-params.limit :]

    return RecentSessionsOutput(
        available=True,
        user_id=params.user_id,
        count=len(sessions),
        window_days=params.window_days,
        sessions=sessions,
    )


# ---------------------------------------------------------------------------
# Tool 3: get_historical_baseline
# ---------------------------------------------------------------------------

BASELINE_METHOD = "computed_from_valid_history_sliding_window"


class GetHistoricalBaselineInput(BaseModel):
    user_id: str = Field(min_length=1)
    window_days: int = Field(default=30, ge=1, le=3650)


class HistoricalBaselineOutput(BaseModel):
    available: bool
    user_id: str
    window_days: int
    method: str = BASELINE_METHOD
    sample_count: int = 0
    stats: list[FeatureStats] = Field(default_factory=list)
    note: str = ""
    unavailable_reason: str | None = None


def get_historical_baseline(user_id, window_days=30, repository=None):
    """Compute a per-feature baseline from valid sessions in a time window.

    This is a deterministic computation over history, not a stored table. It is
    distinct from (and does not modify) the existing static ``/baseline``
    endpoint.
    """
    params = GetHistoricalBaselineInput(user_id=user_id, window_days=window_days)

    try:
        sessions = _load_canonical_sessions(params.user_id, repository)
    except RepositoryError as exc:
        logger.warning("get_historical_baseline: repository unavailable")
        return HistoricalBaselineOutput(
            available=False,
            user_id=params.user_id,
            window_days=params.window_days,
            unavailable_reason=str(exc) or "repository_unavailable",
        )
    except Exception as exc:
        logger.exception("get_historical_baseline failed unexpectedly")
        return HistoricalBaselineOutput(
            available=False,
            user_id=params.user_id,
            window_days=params.window_days,
            unavailable_reason=f"unexpected_error: {exc.__class__.__name__}",
        )

    cutoff = _now() - timedelta(days=params.window_days)
    valid = [s for s in sessions if s.is_valid and _within(s, cutoff, None)]

    if not valid:
        return HistoricalBaselineOutput(
            available=False,
            user_id=params.user_id,
            window_days=params.window_days,
            note="No valid sessions in the requested window.",
            unavailable_reason="no_valid_sessions_in_window",
        )

    return HistoricalBaselineOutput(
        available=True,
        user_id=params.user_id,
        window_days=params.window_days,
        sample_count=len(valid),
        stats=_stats_by_feature(valid),
        note=(
            f"Computed from {len(valid)} valid sessions in the last "
            f"{params.window_days} days."
        ),
    )


# ---------------------------------------------------------------------------
# Tool 4: calculate_behavioral_drift
# ---------------------------------------------------------------------------


class FeatureDrift(BaseModel):
    key: str
    label: str
    canonical_unit: str
    available: bool
    baseline_value: float | None = None
    recent_value: float | None = None
    absolute_change: float | None = None
    relative_change: float | None = None
    direction: str = "unknown"


class CalculateBehavioralDriftInput(BaseModel):
    user_id: str = Field(min_length=1)
    recent_window_days: int = Field(default=7, ge=1, le=3650)
    baseline_window_days: int = Field(default=30, ge=1, le=3650)
    minimum_samples: int = Field(default=3, ge=1, le=100)


class BehavioralDriftOutput(BaseModel):
    available: bool
    user_id: str
    recent_window_days: int
    baseline_window_days: int
    recent_sample_count: int = 0
    baseline_sample_count: int = 0
    sufficient: bool = False
    features: list[FeatureDrift] = Field(default_factory=list)
    note: str = ""
    unavailable_reason: str | None = None


def calculate_behavioral_drift(
    user_id,
    recent_window_days=7,
    baseline_window_days=30,
    minimum_samples=3,
    repository=None,
):
    """Compare a recent window against the equally sized window before it.

    The baseline window is the ``baseline_window_days`` immediately preceding
    the recent window, so the two do not overlap. Direction is a plain sign of
    the change; it is not a judgement about the cause.
    """
    params = CalculateBehavioralDriftInput(
        user_id=user_id,
        recent_window_days=recent_window_days,
        baseline_window_days=baseline_window_days,
        minimum_samples=minimum_samples,
    )

    try:
        sessions = _load_canonical_sessions(params.user_id, repository)
    except RepositoryError as exc:
        logger.warning("calculate_behavioral_drift: repository unavailable")
        return BehavioralDriftOutput(
            available=False,
            user_id=params.user_id,
            recent_window_days=params.recent_window_days,
            baseline_window_days=params.baseline_window_days,
            unavailable_reason=str(exc) or "repository_unavailable",
        )
    except Exception as exc:
        logger.exception("calculate_behavioral_drift failed unexpectedly")
        return BehavioralDriftOutput(
            available=False,
            user_id=params.user_id,
            recent_window_days=params.recent_window_days,
            baseline_window_days=params.baseline_window_days,
            unavailable_reason=f"unexpected_error: {exc.__class__.__name__}",
        )

    now = _now()
    recent_start = now - timedelta(days=params.recent_window_days)
    baseline_start = recent_start - timedelta(days=params.baseline_window_days)

    recent = [s for s in sessions if s.is_valid and _within(s, recent_start, now)]
    baseline = [s for s in sessions if s.is_valid and _within(s, baseline_start, recent_start)]

    if not recent and not baseline:
        return BehavioralDriftOutput(
            available=False,
            user_id=params.user_id,
            recent_window_days=params.recent_window_days,
            baseline_window_days=params.baseline_window_days,
            recent_sample_count=0,
            baseline_sample_count=0,
            note="No valid sessions in either window.",
            unavailable_reason="no_valid_sessions_in_windows",
        )

    sufficient = (
        len(recent) >= params.minimum_samples
        and len(baseline) >= params.minimum_samples
    )

    features = []
    for key in CANONICAL_FEATURE_KEYS:
        recent_stats = _stats_for(key, recent)
        baseline_stats = _stats_for(key, baseline)
        baseline_value = baseline_stats.mean
        recent_value = recent_stats.mean
        absolute = relative = None
        direction = "unknown"
        if baseline_value is not None and recent_value is not None:
            absolute = recent_value - baseline_value
            relative = _relative_change(baseline_value, recent_value)
            if absolute > 0:
                direction = "increase"
            elif absolute < 0:
                direction = "decrease"
            else:
                direction = "unknown"
        features.append(
            FeatureDrift(
                key=key,
                label=label_for(key),
                canonical_unit=unit_for(key),
                available=baseline_value is not None and recent_value is not None,
                baseline_value=baseline_value,
                recent_value=recent_value,
                absolute_change=absolute,
                relative_change=relative,
                direction=direction,
            )
        )

    return BehavioralDriftOutput(
        available=True,
        user_id=params.user_id,
        recent_window_days=params.recent_window_days,
        baseline_window_days=params.baseline_window_days,
        recent_sample_count=len(recent),
        baseline_sample_count=len(baseline),
        sufficient=sufficient,
        features=features,
        note=(
            "Recent vs. preceding baseline window. Direction is a sign of "
            "change only; it is not an explanation or a diagnosis."
        ),
    )


# ---------------------------------------------------------------------------
# Tool 5: compare_time_windows
# ---------------------------------------------------------------------------


class WindowComparison(BaseModel):
    label: str
    window_days: int
    start: datetime | None = None
    end: datetime | None = None
    sample_count: int = 0
    stats: list[FeatureStats] = Field(default_factory=list)


class CompareTimeWindowsOutput(BaseModel):
    available: bool
    user_id: str
    window_a: WindowComparison | None = None
    window_b: WindowComparison | None = None
    deltas: list[FeatureDrift] = Field(default_factory=list)
    note: str = ""
    unavailable_reason: str | None = None


class CompareTimeWindowsInput(BaseModel):
    user_id: str = Field(min_length=1)
    window_a_days: int = Field(default=7, ge=1, le=3650)
    window_b_days: int = Field(default=30, ge=1, le=3650)


def compare_time_windows(user_id, window_a_days=7, window_b_days=30, repository=None):
    """Compare two adjacent, non-overlapping windows of valid sessions.

    Window A is the most recent ``window_a_days``. Window B is the
    ``window_b_days`` immediately preceding window A.
    """
    params = CompareTimeWindowsInput(
        user_id=user_id, window_a_days=window_a_days, window_b_days=window_b_days
    )

    try:
        sessions = _load_canonical_sessions(params.user_id, repository)
    except RepositoryError as exc:
        logger.warning("compare_time_windows: repository unavailable")
        return CompareTimeWindowsOutput(
            available=False,
            user_id=params.user_id,
            unavailable_reason=str(exc) or "repository_unavailable",
        )
    except Exception as exc:
        logger.exception("compare_time_windows failed unexpectedly")
        return CompareTimeWindowsOutput(
            available=False,
            user_id=params.user_id,
            unavailable_reason=f"unexpected_error: {exc.__class__.__name__}",
        )

    now = _now()
    a_start = now - timedelta(days=params.window_a_days)
    b_start = a_start - timedelta(days=params.window_b_days)

    a_sessions = [s for s in sessions if s.is_valid and _within(s, a_start, now)]
    b_sessions = [s for s in sessions if s.is_valid and _within(s, b_start, a_start)]

    if not a_sessions and not b_sessions:
        return CompareTimeWindowsOutput(
            available=False,
            user_id=params.user_id,
            unavailable_reason="no_valid_sessions_in_windows",
        )

    window_a = WindowComparison(
        label="recent",
        window_days=params.window_a_days,
        start=a_start,
        end=now,
        sample_count=len(a_sessions),
        stats=_stats_by_feature(a_sessions),
    )
    window_b = WindowComparison(
        label="previous",
        window_days=params.window_b_days,
        start=b_start,
        end=a_start,
        sample_count=len(b_sessions),
        stats=_stats_by_feature(b_sessions),
    )

    b_by_key = {s.key: s for s in window_b.stats}
    deltas = []
    for a_stats in window_a.stats:
        b_stats = b_by_key[a_stats.key]
        absolute = relative = None
        direction = "unknown"
        if a_stats.mean is not None and b_stats.mean is not None:
            absolute = a_stats.mean - b_stats.mean
            relative = _relative_change(b_stats.mean, a_stats.mean)
            direction = (
                "increase" if absolute > 0 else "decrease" if absolute < 0 else "unknown"
            )
        deltas.append(
            FeatureDrift(
                key=a_stats.key,
                label=a_stats.label,
                canonical_unit=a_stats.canonical_unit,
                available=a_stats.mean is not None and b_stats.mean is not None,
                baseline_value=b_stats.mean,
                recent_value=a_stats.mean,
                absolute_change=absolute,
                relative_change=relative,
                direction=direction,
            )
        )

    return CompareTimeWindowsOutput(
        available=True,
        user_id=params.user_id,
        window_a=window_a,
        window_b=window_b,
        deltas=deltas,
        note=(
            "Window A is the recent window; window B is the window immediately "
            "before it. Both are valid-sessions-only."
        ),
    )
