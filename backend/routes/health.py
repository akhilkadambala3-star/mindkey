from fastapi import APIRouter

from schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        message="MindKey backend is running",
    )
