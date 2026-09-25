from datetime import datetime

from pydantic import BaseModel

from investigation.agent import GroundedReport


class HealthResponse(BaseModel):
    status: str
    message: str


class SupabaseTestResponse(BaseModel):
    status: str
    message: str


class TypingSessionCreate(BaseModel):
    user_id: str
    session_start: datetime
    session_end: datetime
    dwell_mean: float
    flight_mean: float
    typing_speed: float
    correction_rate: float
    rhythm_variability: float
    pause_count: int


class AnomalyResult(BaseModel):
    """Anomaly-detection outcome attached to a stored typing session.

    ``status`` is one of:
    - "ok": a model was trained on the user's valid history and the new
      session was evaluated against it.
    - "insufficient_data": fewer valid historical sessions than the
      required minimum, so no model was trained.
    - "invalid_session": the session failed feature validation.
    - "ml_unavailable": the ML runtime (scikit-learn) is not installed in
      the backend environment.
    - "ml_error": an unexpected failure while retrieving history, training,
      or evaluating.

    ``anomaly_score`` (higher = more anomalous) and ``is_anomaly`` are only
    set when ``status`` is "ok"; ``available_sessions`` and
    ``required_sessions`` are only set when ``status`` is
    "insufficient_data".
    """

    status: str
    message: str
    anomaly_score: float | None = None
    is_anomaly: bool | None = None
    available_sessions: int | None = None
    required_sessions: int | None = None


class TypingSessionResponse(BaseModel):
    status: str
    message: str
    session_id: str
    anomaly: AnomalyResult


# ---------------------------------------------------------------------------
# Phase 4: read-only response models for the documented endpoints
# ---------------------------------------------------------------------------
# Purely additive. These describe the read contract only: sessions emit
# neither ``status`` nor ``consistency`` (Phase 4 decision), and no model
# ever exposes typed content — only timing-derived contract fields. The
# investigation report reuses the frozen ``GroundedReport`` kernel model so
# the served payload and its API schema can never drift apart.


class SessionReadModel(BaseModel):
    """One stored typing session in the dashboard contract shape.

    Stored units: ``typing_speed`` is characters per minute, ``wpm`` is the
    presentation conversion (5 keystrokes per word), dwell/flight arrive in
    milliseconds, ``correction_rate`` is a 0-1 ratio, ``rhythm_variability``
    is seconds and ``pause_count`` is a count.
    """

    session_id: str
    session_start: str | None = None
    session_end: str | None = None
    date: str | None = None
    typing_speed: float | None = None
    wpm: float | None = None
    dwell_mean_ms: float | None = None
    flight_mean_ms: float | None = None
    correction_rate: float | None = None
    rhythm_variability: float | None = None
    pause_count: int | None = None


class BaselineReadModel(BaseModel):
    """The stored personal baseline (dashboard contract, null-safe).

    ``sample_count`` is 0 and every feature is null when no baseline exists,
    so the dashboard can render "no baseline yet" defensively.
    """

    typing_speed: float | None = None
    dwell_mean: float | None = None
    flight_mean: float | None = None
    correction_rate: float | None = None
    rhythm_variability: float | None = None
    pause_count: float | None = None
    wpm: float | None = None
    sample_count: int = 0
    updated_at: str | None = None


class InvestigationReadModel(BaseModel):
    """One investigation outcome: frozen-kernel envelope, JSON-safe.

    ``report`` is the frozen ``GroundedReport`` when a kernel run happened
    and null when there were no stored sessions to investigate;
    ``stop_reason`` uses the engine's fixed vocabulary; ``disclaimer`` is
    always the kernel's fixed non-diagnostic notice.
    """

    schema_version: str
    user_id: str
    session_id: str | None = None
    stop_reason: str
    evidence_digest: str | None = None
    report: GroundedReport | None = None
    timeline: list[str]
    limitations: list[str]
    disclaimer: str