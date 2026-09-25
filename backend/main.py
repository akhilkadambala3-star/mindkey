from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.api import router as api_router
from routes.baseline import router as baseline_router
from routes.health import router as health_router
from routes.supabase_test import router as supabase_test_router
from routes.typing import router as typing_router

app = FastAPI(
    title="MindKey Backend",
    description="Privacy-first early-warning API. Not a diagnostic system.",
    version="0.1.0",
)

# Phase 4: read-only API for the dashboard served from :5500. Explicit
# origin list (never "*") and GET only; no credentials are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(supabase_test_router)
app.include_router(typing_router)
app.include_router(baseline_router)
app.include_router(api_router)
