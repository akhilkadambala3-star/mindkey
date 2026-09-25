"""Run the frozen investigation kernel for a user's read request (Phase 4).

Pass-through only: every substantive output — stop reason, evidence digest,
report, timeline, limitations, disclaimer — comes verbatim from the frozen
``investigation.agent`` pipeline. No threshold, status, hypothesis or claim
is authored here.

Boundary decisions
------------------
- The store is resolved here (not inside the engine) so an unreachable store
  surfaces as ``RepositoryError`` → HTTP 503 instead of a 200 payload.
- An explicitly requested ``session_id`` must exist among the user's stored
  rows; otherwise ``SessionNotFound`` → HTTP 404.
- Without an explicit session the latest stored session is used
  (deterministic: ``session_start``, then id). With no stored sessions there
  is nothing to investigate: an honest, report-less ``evidence_insufficient``
  payload is returned (``report`` null, ``timeline`` empty, the kernel's
  fixed disclaimer present) — nothing is fabricated and nothing fails.
"""

from investigation.agent import (
    ENGINE_SCHEMA_VERSION,
    REPORT_DISCLAIMER,
    build_grounded_report,
    render_timeline_for,
    run_investigation,
)
from investigation.repository import default_repository

from .reads import latest_session_id_for


class SessionNotFound(Exception):
    """The explicitly requested session is not stored for this user."""

    def __init__(self, session_id):
        super().__init__(f"Unknown session: {session_id}")
        self.session_id = str(session_id)


def _empty_payload(user_id):
    """An honest "no stored sessions" outcome for the investigation contract.

    ``stop_reason`` reuses the engine's fixed vocabulary and the disclaimer
    is the kernel's fixed notice; ``report`` stays null because no kernel run
    happened and this layer invents nothing. Top-level ``schema_version`` is
    the engine's (the embedded report carries its own when present).
    """
    return {
        "schema_version": ENGINE_SCHEMA_VERSION,
        "user_id": user_id,
        "session_id": None,
        "stop_reason": "evidence_insufficient",
        "evidence_digest": None,
        "report": None,
        "timeline": [],
        "limitations": [],
        "disclaimer": REPORT_DISCLAIMER,
    }


def run_for_user(
    user_id,
    session_id=None,
    *,
    repository=None,
    clock=None,
    as_of=None,
):
    """Investigate one user's history and return a JSON-safe payload.

    Args:
        user_id: the user whose read-only history is investigated.
        session_id: explicit triggering session; when omitted, the latest
            stored session is used (see ``latest_session_id_for``).
        repository: a read-only repository; defaults to the production one
            (``RepositoryError`` when the store is unavailable).
        clock: optional zero-argument clock for deterministic trace stamps.
        as_of: optional reference time for windowed analysis.

    Returns:
        A dict with exactly the contract keys: ``schema_version``,
        ``user_id``, ``session_id``, ``stop_reason``, ``evidence_digest``,
        ``report``, ``timeline``, ``limitations``, ``disclaimer``.

    Raises:
        SessionNotFound: an explicit ``session_id`` is not stored for the
            user.
        RepositoryError: the store cannot be constructed or read.
    """
    repository = repository if repository is not None else default_repository()
    rows = repository.list_sessions(user_id)

    if session_id is not None:
        target = str(session_id)
        stored = {str(row["id"]) for row in rows if row.get("id") is not None}
        if target not in stored:
            raise SessionNotFound(target)
    else:
        target = latest_session_id_for(rows)
        if target is None:
            return _empty_payload(user_id)

    result = run_investigation(
        user_id,
        target,
        repository=repository,
        clock=clock,
        as_of=as_of,
    )
    report = build_grounded_report(result, clock=clock)
    timeline = render_timeline_for(report, result, clock=clock)

    return {
        "schema_version": result.schema_version,
        "user_id": result.user_id,
        "session_id": str(result.session_id),
        "stop_reason": result.stop_reason,
        "evidence_digest": result.evidence_digest,
        "report": report.model_dump(mode="json"),
        "timeline": list(timeline),
        "limitations": list(report.limitations),
        "disclaimer": report.disclaimer,
    }
