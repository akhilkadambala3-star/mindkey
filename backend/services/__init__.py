"""Read-only service layer (Phase 4).

Thin, documented projections between stored rows (Supabase, read-only) and
the documented API contract. Nothing in this package writes to a store,
computes a status/threshold, or authors a claim: unit conversion and session
validity are reused from the frozen ``investigation`` kernel, and every
investigation outcome comes verbatim from ``investigation.agent``.
"""

from .investigations import SessionNotFound, run_for_user
from .reads import (
    anomaly_client,
    baseline_client,
    latest_session_id,
    latest_session_id_for,
    list_anomalies,
    list_sessions,
    user_baseline,
)

__all__ = [
    "SessionNotFound",
    "anomaly_client",
    "baseline_client",
    "latest_session_id",
    "latest_session_id_for",
    "list_anomalies",
    "list_sessions",
    "run_for_user",
    "user_baseline",
]
