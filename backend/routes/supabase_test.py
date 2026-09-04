from fastapi import APIRouter, HTTPException

from database import supabase
from schemas import SupabaseTestResponse

router = APIRouter()


@router.get("/supabase-test", response_model=SupabaseTestResponse)
def supabase_test():
    try:
        supabase.table("users").select("*").limit(0).execute()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Supabase database communication failed",
        )

    return SupabaseTestResponse(
        status="ok",
        message="Supabase client can communicate with the database",
    )
