from datetime import datetime, timezone
from statistics import mean

from fastapi import APIRouter, HTTPException

from database import supabase

router = APIRouter()

FEATURE_KEYS = (
    "dwell_mean",
    "flight_mean",
    "typing_speed",
    "correction_rate",
    "rhythm_variability",
    "pause_count",
)


def _is_valid_session(session):
    for key in FEATURE_KEYS:
        if session.get(key) is None:
            return False

    typing_speed = session["typing_speed"]
    correction_rate = session["correction_rate"]

    if typing_speed <= 0:
        return False
    if correction_rate < 0 or correction_rate > 1:
        return False

    return True


@router.post("/baseline/{user_id}")
def create_or_update_baseline(user_id: str):
    try:
        sessions_result = (
            supabase.table("typing_sessions")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        sessions = sessions_result.data or []
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Failed to calculate baseline",
        )

    valid_sessions = [session for session in sessions if _is_valid_session(session)]
    sample_count = len(valid_sessions)

    if sample_count < 3:
        raise HTTPException(
            status_code=400,
            detail="At least 3 valid typing sessions are required before a personal baseline can be calculated.",
        )

    baseline_values = {
        key: mean(session[key] for session in valid_sessions)
        for key in FEATURE_KEYS
    }
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "user_id": user_id,
        "sample_count": sample_count,
        "updated_at": now,
        **baseline_values,
    }

    try:
        existing_result = (
            supabase.table("baselines")
            .select("id")
            .eq("user_id", user_id)
            .execute()
        )
        existing = existing_result.data or []

        if existing:
            supabase.table("baselines").update(record).eq("id", existing[0]["id"]).execute()
        else:
            supabase.table("baselines").insert(record).execute()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Failed to store baseline",
        )

    return {
        "status": "ok",
        "message": "Personal baseline stored",
        "user_id": user_id,
        "sample_count": sample_count,
        **baseline_values,
    }
