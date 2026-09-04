from fastapi import FastAPI

from routes.baseline import router as baseline_router
from routes.health import router as health_router
from routes.supabase_test import router as supabase_test_router
from routes.typing import router as typing_router

app = FastAPI(
    title="MindKey Backend",
    description="Privacy-first early-warning API. Not a diagnostic system.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(supabase_test_router)
app.include_router(typing_router)
app.include_router(baseline_router)
