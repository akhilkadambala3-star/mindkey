import sys
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from database import supabase
from schemas import AnomalyResult, TypingSessionCreate, TypingSessionResponse

router = APIRouter()

# --- Guarded import of the ML anomaly-detection layer -----------------------
# The ML modules live in the repo-root "ml" package and use package-qualified
# imports (e.g. "from ml.features import ..."), so the repository root must be
# on sys.path for "ml" to resolve as a package. The import is guarded: if the
# ML runtime is unavailable, the backend still starts and typing sessions are
# still stored; the anomaly result then reports status "ml_unavailable". This
# keeps /health, /supabase-test and /baseline/{user_id} working regardless.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from ml.anomaly_detector import AnomalyDetector, MIN_TRAINING_SESSIONS
    from ml.features import is_valid_session

    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False


def _resolve_session_duration(session):
    """Resolve the canonical ``session_duration`` feature for a session.

    The listener now sends ``session_duration`` directly. Stored
    ``typing_sessions`` rows predate that field, so it is derived from the
    ``session_start``/``session_end`` timestamps when absent. Returns None
    when neither source is usable.
    """
    value = session.get("session_duration")
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    start = session.get("session_start")
    end = session.get("session_end")
    if not start or not end:
        return None

    try:
        start_dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except ValueError:
        return None

    return max(0.0, (end_dt - start_dt).total_seconds())


def _with_session_duration(session):
    """Return ``session`` carrying a ``session_duration`` for ML validation.

    A copy is returned so stored/read rows are never mutated. Rows that
    already carry the field are passed through untouched.
    """
    if session.get("session_duration") is not None:
        return session

    duration = _resolve_session_duration(session)
    if duration is None:
        return session

    enriched = dict(session)
    enriched["session_duration"] = duration
    return enriched


def _store_anomaly_result(user_id, session_id, result):
    """Best-effort persistence of an anomaly result.

    Stores exactly: user_id, session_id, anomaly_score, is_anomaly. The
    insert is guarded because a failure here must never fail the already
    stored typing session.
    """
    try:
        supabase.table("anomaly_results").insert(
            {
                "user_id": user_id,
                "session_id": session_id,
                "anomaly_score": result["anomaly_score"],
                "is_anomaly": result["is_anomaly"],
            }
        ).execute()
    except Exception:
        pass


def _evaluate_anomaly(session: TypingSessionCreate, session_id: str) -> AnomalyResult:
    """Score a new typing session against the user's normal behavior.

    Runs after the session has been stored. Never raises: every ML-related
    outcome is reported through ``AnomalyResult.status`` so the endpoint's
    existing success response is preserved.

    Flow:
    1. Reuse ml/features.py validation (single source of truth).
    2. Fetch the user's typing history and filter to valid sessions,
       excluding the just-stored session so it is never in its own
       training pool.
    3. Fewer than MIN_TRAINING_SESSIONS valid historical sessions ->
       "insufficient_data"; no training is attempted.
    4. Otherwise train the Isolation Forest on the valid history and
       evaluate the new session.
    """
    if not _ML_AVAILABLE:
        return AnomalyResult(
            status="ml_unavailable",
            message=(
                "ML anomaly detection is unavailable because scikit-learn "
                "is not installed in the backend runtime."
            ),
        )

    session_data = _with_session_duration(session.model_dump(mode="json"))

    if not is_valid_session(session_data):
        return AnomalyResult(
            status="invalid_session",
            message=(
                "Session is missing required features or has values outside "
                "the expected ranges."
            ),
        )

    try:
        sessions_result = (
            supabase.table("typing_sessions")
            .select("*")
            .eq("user_id", session.user_id)
            .execute()
        )
        sessions = sessions_result.data or []
    except Exception:
        return AnomalyResult(
            status="ml_error",
            message="Failed to retrieve typing history for anomaly detection.",
        )

    # Historical sessions exclude the session that was just stored, so the
    # evaluated session is never part of its own training pool.
    valid_history = [
        enriched
        for enriched in (_with_session_duration(s) for s in sessions)
        if str(enriched.get("id")) != session_id and is_valid_session(enriched)
    ]

    if len(valid_history) < MIN_TRAINING_SESSIONS:
        return AnomalyResult(
            status="insufficient_data",
            message=(
                f"At least {MIN_TRAINING_SESSIONS} valid historical typing "
                "sessions are required for anomaly detection; "
                f"{len(valid_history)} are available."
            ),
            available_sessions=len(valid_history),
            required_sessions=MIN_TRAINING_SESSIONS,
        )

    try:
        detector = AnomalyDetector()
        detector.train(valid_history)
        result = detector.evaluate(session_data)
    except Exception:
        return AnomalyResult(
            status="ml_error",
            message="Failed to train or evaluate the anomaly model.",
        )

    # result["status"] is "ok" here: the session already passed validation.
    _store_anomaly_result(session.user_id, session_id, result)

    return AnomalyResult(
        status=result["status"],
        message=result["message"],
        anomaly_score=result["anomaly_score"],
        is_anomaly=result["is_anomaly"],
    )


@router.post("/typing/session", response_model=TypingSessionResponse)
def create_typing_session(session: TypingSessionCreate):
    # 1. Store the session exactly as before.
    try:
        result = (
            supabase.table("typing_sessions")
            .insert(session.model_dump(mode="json", exclude={"session_duration"}))
            .execute()
        )
        inserted = result.data[0] if result.data else None
        if not inserted or "id" not in inserted:
            raise ValueError("Insert did not return a session id")
        session_id = str(inserted["id"])
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Failed to store typing session",
        )

    # 2. Anomaly detection against the user's valid history. Any outcome
    #    (including failures) is reported inside the response; the session
    #    storage itself has already succeeded.
    anomaly = _evaluate_anomaly(session, session_id)

    return TypingSessionResponse(
        status="ok",
        message="Typing session stored",
        session_id=session_id,
        anomaly=anomaly,
    )