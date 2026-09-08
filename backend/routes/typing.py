import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from database import supabase
from schemas import AnomalyResult, TypingSessionCreate, TypingSessionResponse

router = APIRouter()

# --- Guarded import of the ML anomaly-detection layer -----------------------
# The ML modules in ml/ use flat imports (e.g. "from features import ..."),
# so ml/ itself must be on sys.path for them to resolve. The import is
# guarded: if scikit-learn is not installed in the backend's runtime, the
# backend still starts and typing sessions are still stored; the anomaly
# result then reports status "ml_unavailable". This keeps /health,
# /supabase-test and /baseline/{user_id} working regardless.
_ML_DIR = Path(__file__).resolve().parents[2] / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

try:
    from anomaly_detector import AnomalyDetector, MIN_TRAINING_SESSIONS
    from features import is_valid_session

    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False


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

    session_data = session.model_dump(mode="json")

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
        s for s in sessions
        if str(s.get("id")) != session_id and is_valid_session(s)
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
            .insert(session.model_dump(mode="json"))
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