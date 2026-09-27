import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.api import router as api_router
from routes.baseline import router as baseline_router
from routes.checkins import router as checkins_router
from routes.demo import router as demo_router
from routes.health import router as health_router
from routes.supabase_test import router as supabase_test_router
from routes.typing import router as typing_router

app = FastAPI(
    title="MindKey Backend",
    description="Privacy-first early-warning API. Not a diagnostic system.",
    version="0.1.0",
)

# Dashboard origins: explicit list (never "*"), no credentials. POST is
# allowed only for wellbeing check-ins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        # Extra deployed dashboard origins (e.g. the Vercel URL), comma separated.
        *[o.strip() for o in os.getenv("MINDKEY_CORS_ORIGINS", "").split(",") if o.strip()],
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(supabase_test_router)
app.include_router(typing_router)
app.include_router(baseline_router)
app.include_router(api_router)
app.include_router(demo_router)
app.include_router(checkins_router)
