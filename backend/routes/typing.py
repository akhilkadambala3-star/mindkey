from fastapi import APIRouter, HTTPException

from database import supabase
from schemas import TypingSessionCreate, TypingSessionResponse

router = APIRouter()


@router.post("/typing/session", response_model=TypingSessionResponse)
def create_typing_session(session: TypingSessionCreate):
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

    return TypingSessionResponse(
        status="ok",
        message="Typing session stored",
        session_id=session_id,
    )
