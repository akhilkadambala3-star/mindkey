"""Demo Lab routes: the real investigation engine on in-memory scenarios.

Read-only and offline: no store is constructed, so these work without any
Supabase configuration. Every payload is labelled ``data_source: "demo"``.
"""

from fastapi import APIRouter, HTTPException

from services import demo

router = APIRouter()


@router.get("/api/demo/scenarios")
def list_demo_scenarios():
    """The demo scenario catalog, in stable order."""
    return demo.list_scenarios()


@router.get("/api/demo/scenarios/{key}")
def run_demo_scenario(key: str):
    """Run one demo scenario through the real engine (deterministic)."""
    try:
        return demo.scenario_payload(key)
    except demo.UnknownScenario:
        raise HTTPException(status_code=404, detail="Unknown demo scenario") from None
