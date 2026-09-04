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


class TypingSessionResponse(BaseModel):
    status: str
    message: str
    session_id: str
