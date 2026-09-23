"""Deterministic evidence adapter (Phase 2).

``build_behavioral_evidence(user_id, session_id)`` composes the read-only tools
into a single :class:`~investigation.contracts.BehavioralEvidence` package.

This module is deterministic: no randomness, no LLM, no network beyond the
repository it is given, and no inference of persistence when the history is too
small. It reports what is measured, what is unavailable, and what remains
uncertain -- and never a diagnosis.
"""

import logging
from datetime import datetime, timezone

from .bridge import MODEL_MINIMUM_SESSIONS
from .contracts import (
    BehavioralEvidence,
    ContextEvidence,
    DataQuality,
    PersistenceAssessment,
    SignalEvidence,
    TemporalAnalysis,
    TriggerInfo,
    Uncertainty,
    WindowSummary,
)
from .tools import (
    calculate_behavioral_drift,
    compare_time_windows,
    get_historical_baseline,
    get_ml_evidence,
    get_recent_sessions,
)
from .units import CANONICAL_FEATURE_KEYS, canonical_units, label_for, unit_for

logger = logging.getLogger(__name__)

#: Relative change above which a signal is considered "moved" by the
#: deterministic persistence heuristic. Documented, not hidden.
MOVED_RELATIVE_THRESHOLD = 0.10

#: Minimum number of simultaneously moved signals to suggest persistence.
MIN_CONSISTENT_SIGNALS = 3

#: Minimum recent valid sessions before any persistence claim beyond
#: "insufficient_data" is allowed.
MIN_RECENT_SESSIONS_FOR_PERSISTENCE = 5

DEFAULT_RECENT_WINDOW_DAYS = 7
DEFAULT_BASELINE_WINDOW_DAYS = 30


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_session(sessions, session_id):
    for session in sessions:
        if session.session_id is not None and str(session.session_id) == str(session_id):
            return session
    return None


def _assess_persistence(drift, data_quality):
    """Conservative, deterministic persistence hint.

    Never returns anything other than ``"insufficient_data"`` when the valid
    history is below the model minimum.
    """
    if not data_quality.meets_model_minimum:
        return PersistenceAssessment(
            status="insufficient_data",
            supporting_session_count=drift.recent_sample_count,
            signals_moved=0,
            note=(
                f"Only {data_quality.valid_sessions} valid session(s); at least "
                f"{MODEL_MINIMUM_SESSIONS} are required before persistence can be "
                "assessed."
            ),
        )

    moved = [
        feature
        for feature in drift.features
        if feature.available
        and feature.relative_change is not None
        and abs(feature.relative_change) >= MOVED_RELATIVE_THRESHOLD
    ]
    increases = sum(1 for f in moved if f.direction == "increase")
    decreases = sum(1 for f in moved if f.direction == "decrease")
    same_direction = max(increases, decreases)

    if (
        drift.recent_sample_count >= MIN_RECENT_SESSIONS_FOR_PERSISTENCE
        and same_direction >= MIN_CONSISTENT_SIGNALS
    ):
        status = "persistent_change"
    elif moved:
        status = "recent_variation"
    else:
        status = "stable"

    return PersistenceAssessment(
        status=status,
        supporting_session_count=drift.recent_sample_count,
        signals_moved=len(moved),
        note=(
            "Deterministic heuristic only: counts signals whose recent vs. "
            "baseline relative change exceeds "
            f"{int(MOVED_RELATIVE_THRESHOLD * 100)}%. It is not a diagnosis and "
            "does not establish a cause."
        ),
    )


def _window_summary(label, window_days, session_count, valid_count, start, end):
    return WindowSummary(
        label=label,
        window_days=window_days,
        session_count=session_count,
        valid_session_count=valid_count,
        start=start,
        end=end,
    )


def build_behavioral_evidence(user_id, session_id, repository=None):
    """Build a deterministic :class:`BehavioralEvidence` for one session.

    Args:
        user_id: the user whose history is read.
        session_id: the session that triggered the investigation.
        repository: optional read-only repository (tests inject a fake).

    Returns:
        BehavioralEvidence. Missing or unavailable data is represented
        explicitly; nothing is fabricated.
    """
    if not user_id:
        raise ValueError("user_id is required")
    if not session_id:
        raise ValueError("session_id is required")

    limitations = []

    ml = get_ml_evidence(user_id, session_id, repository=repository)
    recent = get_recent_sessions(user_id, limit=500, repository=repository)
    baseline = get_historical_baseline(
        user_id, window_days=DEFAULT_BASELINE_WINDOW_DAYS, repository=repository
    )
    drift = calculate_behavioral_drift(
        user_id,
        recent_window_days=DEFAULT_RECENT_WINDOW_DAYS,
        baseline_window_days=DEFAULT_BASELINE_WINDOW_DAYS,
        repository=repository,
    )
    windows = compare_time_windows(
        user_id,
        window_a_days=DEFAULT_RECENT_WINDOW_DAYS,
        window_b_days=DEFAULT_BASELINE_WINDOW_DAYS,
        repository=repository,
    )

    sessions = list(recent.sessions) if recent.available else []
    target = _find_session(sessions, session_id)

    # --- Data quality -----------------------------------------------------
    valid_count = sum(1 for s in sessions if s.is_valid)
    invalid_count = len(sessions) - valid_count
    without_timestamp = sum(1 for s in sessions if s.session_start is None)
    meets_minimum = valid_count >= MODEL_MINIMUM_SESSIONS

    issues = []
    if not recent.available:
        issues.append(
            recent.unavailable_reason or "recent sessions are unavailable"
        )
    if valid_count == 0:
        issues.append("no valid sessions are available")
    if valid_count < MODEL_MINIMUM_SESSIONS:
        issues.append(
            f"only {valid_count} valid session(s); at least "
            f"{MODEL_MINIMUM_SESSIONS} are required for model-based evidence"
        )
    if without_timestamp:
        issues.append(
            f"{without_timestamp} session(s) have no usable timestamp and were "
            "excluded from windowed comparisons"
        )
    if not drift.sufficient:
        issues.append(
            "not enough valid sessions in one or both comparison windows for a "
            "stable drift estimate"
        )

    data_quality = DataQuality(
        total_sessions=len(sessions),
        valid_sessions=valid_count,
        invalid_sessions=invalid_count,
        sessions_without_timestamp=without_timestamp,
        model_minimum_sessions=MODEL_MINIMUM_SESSIONS,
        meets_model_minimum=meets_minimum,
        issues=issues,
    )

    # --- Signals ----------------------------------------------------------
    baseline_by_key = {stat.key: stat for stat in baseline.stats}
    drift_by_key = {feature.key: feature for feature in drift.features}

    signals = []
    for key in CANONICAL_FEATURE_KEYS:
        value = None if target is None else _as_float(getattr(target, key, None))

        drift_feature = drift_by_key.get(key)
        baseline_value = None
        if drift_feature is not None:
            baseline_value = drift_feature.baseline_value
        if baseline_value is None:
            stat = baseline_by_key.get(key)
            baseline_value = stat.mean if stat is not None and stat.available else None

        absolute = relative = None
        direction = "unknown"
        if value is not None and baseline_value is not None:
            absolute = value - baseline_value
            if baseline_value != 0:
                relative = absolute / baseline_value
            direction = (
                "increase" if absolute > 0 else "decrease" if absolute < 0 else "unknown"
            )

        signals.append(
            SignalEvidence(
                key=key,
                label=label_for(key),
                canonical_unit=unit_for(key),
                available=value is not None and baseline_value is not None,
                value=value,
                baseline_value=baseline_value,
                absolute_change=absolute,
                relative_change=relative,
                direction=direction,
            )
        )

    # --- Temporal analysis ------------------------------------------------
    a = windows.window_a
    b = windows.window_b
    persistence = _assess_persistence(drift, data_quality)

    temporal_analysis = TemporalAnalysis(
        recent=_window_summary(
            "recent",
            DEFAULT_RECENT_WINDOW_DAYS,
            0 if a is None else a.sample_count,
            0 if a is None else a.sample_count,
            None if a is None else a.start,
            None if a is None else a.end,
        ),
        baseline=_window_summary(
            "baseline",
            DEFAULT_BASELINE_WINDOW_DAYS,
            0 if b is None else b.sample_count,
            0 if b is None else b.sample_count,
            None if b is None else b.start,
            None if b is None else b.end,
        ),
        persistence=persistence,
        recent_feature_stats=[] if a is None else a.stats,
        baseline_feature_stats=[] if b is None else b.stats,
    )

    # --- Context ----------------------------------------------------------
    context = ContextEvidence(
        source="unavailable",
        checkins=[],
        symptoms=None,
        note=(
            "No backend store for check-ins or symptoms exists yet, so no "
            "contextual factors can be retrieved. This is reported as "
            "unavailable rather than inferred."
        ),
    )

    # --- Uncertainty ------------------------------------------------------
    uncertainty_reasons = []
    if not meets_minimum:
        level = "high"
        uncertainty_reasons.append(
            f"valid history ({valid_count}) is below the model minimum "
            f"({MODEL_MINIMUM_SESSIONS})"
        )
    elif not drift.sufficient:
        level = "moderate"
        uncertainty_reasons.append(
            "one or both comparison windows contain too few valid sessions"
        )
    elif persistence.status == "recent_variation":
        level = "moderate"
        uncertainty_reasons.append("variation is recent and may not persist")
    else:
        level = "moderate" if persistence.status == "persistent_change" else "low"

    if not ml.available:
        uncertainty_reasons.append("no stored ML anomaly result for this session")
        if level == "low":
            level = "moderate"

    uncertainty = Uncertainty(level=level, reasons=uncertainty_reasons)

    # --- Limitations ------------------------------------------------------
    limitations.append(
        "This evidence describes measured behavioral signals only. It is not a "
        "medical diagnosis and does not establish a cause."
    )
    limitations.append(
        "Persistence is a deterministic heuristic over relative change across "
        "signals; it is not a clinical finding."
    )
    if not meets_minimum:
        limitations.append(
            "Persistence cannot be assessed from the available history because "
            f"it is below the {MODEL_MINIMUM_SESSIONS}-session model minimum."
        )
    if target is None:
        limitations.append(
            "The triggering session was not found in the retrieved history, so "
            "per-signal current values are unavailable."
        )
    if not ml.available:
        limitations.append(
            "No stored ML anomaly result is available for this session."
        )
    if not baseline.available:
        limitations.append(
            "No valid sessions were available in the baseline window."
        )
    limitations.append(
        "No user-provided contextual factors (sleep, fatigue, stress, symptoms) "
        "are stored server-side yet, so alternative explanations cannot be "
        "confirmed or ruled out."
    )
    limitations.append(
        "Only structured numeric typing features are used; no typed content is "
        "read or processed."
    )

    trigger_start = None if target is None else target.session_start

    return BehavioralEvidence(
        user_id=user_id,
        generated_at=datetime.now(timezone.utc),
        trigger=TriggerInfo(
            source="typing_session",
            session_id=str(session_id),
            session_start=trigger_start,
        ),
        signals=signals,
        temporal_analysis=temporal_analysis,
        data_quality=data_quality,
        context=context,
        uncertainty=uncertainty,
        limitations=limitations,
        units=canonical_units(),
    )
