from datetime import datetime

from pydantic import BaseModel


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