"""Wellbeing check-ins: the user-reported context the investigation reads.

- ``POST /api/users/{user_id}/checkins`` stores one check-in (day + factor,
  optional note). The note is kept for the user but never read by the agent.
- ``GET /api/users/{user_id}/checkins`` lists them, oldest first.

The store is the shared client from ``database`` (Supabase, or the local
SQLite store with ``MINDKEY_STORE=local``). An unreachable store is a 503 with
a generic detail; nothing about the failure is echoed back.
"""

from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException

from investigation.context import CHECKIN_FACTORS, label_for_factor
from schemas import CheckinCreate, CheckinReadModel

router = APIRouter()

_UNAVAILABLE = "Check-ins are temporarily unavailable"


def _client():
    try:
        from database import supabase  # noqa: WPS433 (lazy: no env needed at import)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE) from exc
    return supabase


def _require_user_id(user_id):
    if user_id is None or not str(user_id).strip():
        raise HTTPException(status_code=422, detail="user_id path must not be empty")


def _project(row):
    factor = row.get("factor")
    return {
        "id": str(row.get("id")),
        "date": str(row.get("date"))[:10] if row.get("date") else None,
        "factor": factor,
        "label": label_for_factor(factor),
        "note": row.get("note") or None,
        "created_at": str(row.get("created_at")) if row.get("created_at") else None,
    }


@router.get("/api/users/{user_id}/checkins", response_model=list[CheckinReadModel])
def list_checkins(user_id: str):
    _require_user_id(user_id)
    client = _client()
    try:
        result = client.table("checkins").select("*").eq("user_id", user_id).execute()
    except Exception:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE) from None
    rows = [
        _project(r) for r in (result.data or []) if r.get("factor") in CHECKIN_FACTORS
    ]
    rows.sort(key=lambda r: (r["date"] or "", r["created_at"] or ""))
    return rows


@router.post(
    "/api/users/{user_id}/checkins",
    response_model=CheckinReadModel,
    status_code=201,
)
def create_checkin(user_id: str, checkin: CheckinCreate):
    _require_user_id(user_id)
    today = datetime.now(timezone.utc).date()
    day = checkin.date or today
    # One day of slack covers users ahead of UTC; anything later is a mistake.
    if day > today + timedelta(days=1):
        raise HTTPException(status_code=422, detail="date cannot be in the future")
    record = {
        "user_id": user_id,
        "date": day.isoformat(),
        "factor": checkin.factor,
        "note": (checkin.note or "").strip() or None,
    }
    client = _client()
    try:
        result = client.table("checkins").insert(record).execute()
    except Exception:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE) from None
    stored = (result.data or [record])[0]
    return _project({**record, **stored})
