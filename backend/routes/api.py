"""Read-only API routes (Phase 4): the documented dashboard endpoints.

One router, three GET endpoints. The handlers only validate input and map
errors — every projection lives in ``services.reads`` and every
investigation outcome comes from the frozen ``investigation.agent`` kernel
via ``services.investigations``.

Error mapping (generic details, mirroring ``routes/typing.py`` style):
- 200: an honest outcome, *including* insufficient data (never an error).
- 404: an explicitly requested ``session_id`` not stored for the user.
- 422: malformed path (empty ``user_id``) or unusable query values.
- 503: the store cannot be constructed or read (nothing is leaked).

Phase 4 boundaries: no authentication is added, sessions emit neither
``status`` nor ``consistency``, and nothing here writes to any store.
"""

from collections.abc import Callable
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from investigation.repository import (
    RepositoryError,
    SessionRepository,
    default_repository,
)
from schemas import BaselineReadModel, InvestigationReadModel, SessionReadModel
from services import investigations, reads

router = APIRouter()

#: Fixed, generic 503 details (never echo ids, values, or store errors).
_SESSIONS_UNAVAILABLE = "Failed to read typing sessions"
_BASELINE_UNAVAILABLE = "Failed to read baseline"
_INVESTIGATION_UNAVAILABLE = "Failed to read investigation data"

_USER_ID_REQUIRED = "user_id path must not be empty"
_AS_OF_INVALID = "as_of must be an ISO-8601 timestamp"
_SESSION_ID_EMPTY = "session_id must not be empty"


def get_repository():
    """Dependency: the read-only store, or ``None`` when unavailable.

    Returning ``None`` rather than raising lets each endpoint translate the
    failure into its own generic 503 detail inside its own handler.
    """
    try:
        return default_repository()
    except RepositoryError:
        return None


def get_baseline_client():
    """Dependency: the Supabase client for baseline SELECTs, or ``None``."""
    try:
        return reads.baseline_client()
    except RepositoryError:
        return None


def get_clock():
    """Dependency: the trace clock; ``None`` lets the kernel use its own.

    The seam exists so tests can inject a frozen clock for determinism.
    """
    return None


def _require_user_id(user_id):
    """Reject a malformed path before any store is touched (→ 422)."""
    if user_id is None or not str(user_id).strip():
        raise HTTPException(status_code=422, detail=_USER_ID_REQUIRED)


def _parse_as_of(value):
    """Parse the optional ``as_of`` query value (ISO-8601), or raise 422.

    A naive timestamp is read as UTC — the same convention the frozen
    evidence bridge uses for stored timestamps.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text:
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
    raise HTTPException(status_code=422, detail=_AS_OF_INVALID)


def _default_repository_or_503(detail):
    """Construct the production store; a failure is a generic 503."""
    try:
        return default_repository()
    except RepositoryError as exc:
        raise HTTPException(status_code=503, detail=detail) from exc


def _default_client_or_503():
    """Construct the production baseline client; a failure is a generic 503."""
    try:
        return reads.baseline_client()
    except RepositoryError as exc:
        raise HTTPException(status_code=503, detail=_BASELINE_UNAVAILABLE) from exc


@router.get("/api/users/{user_id}/sessions", response_model=list[SessionReadModel])
def list_user_sessions(
    user_id: str,
    repository: SessionRepository | None = Depends(get_repository),
):
    """Every stored, valid session for ``user_id``, oldest first.

    Unknown users return an empty array; an unreachable store is a 503.
    """
    _require_user_id(user_id)
    if repository is None:
        repository = _default_repository_or_503(_SESSIONS_UNAVAILABLE)
    try:
        return reads.list_sessions(user_id, repository=repository)
    except RepositoryError:
        raise HTTPException(status_code=503, detail=_SESSIONS_UNAVAILABLE) from None


@router.get("/api/users/{user_id}/baseline", response_model=BaselineReadModel)
def get_user_baseline(
    user_id: str,
    client: object | None = Depends(get_baseline_client),
):
    """The stored personal baseline for ``user_id``.

    Missing baseline → 200 with ``sample_count`` 0 and null features;
    an unreachable store is a 503.
    """
    _require_user_id(user_id)
    if client is None:
        client = _default_client_or_503()
    try:
        return reads.user_baseline(user_id, client=client)
    except RepositoryError:
        raise HTTPException(status_code=503, detail=_BASELINE_UNAVAILABLE) from None


@router.get(
    "/api/users/{user_id}/investigation",
    response_model=InvestigationReadModel,
)
def run_user_investigation(
    user_id: str,
    session_id: str | None = None,
    as_of: str | None = None,
    repository: SessionRepository | None = Depends(get_repository),
    clock: Callable[[], datetime] | None = Depends(get_clock),
):
    """Run the frozen, deterministic investigation for one user.

    Outcomes are honest, never partial: insufficient data returns 200 with an
    inconclusive report. Only an unknown explicit session (404), an unusable
    query value (422), or an unreachable store (503) are errors.
    """
    _require_user_id(user_id)
    if session_id is not None and not session_id.strip():
        raise HTTPException(status_code=422, detail=_SESSION_ID_EMPTY)
    reference = _parse_as_of(as_of)
    if repository is None:
        repository = _default_repository_or_503(_INVESTIGATION_UNAVAILABLE)
    try:
        return investigations.run_for_user(
            user_id,
            session_id.strip() if session_id is not None else None,
            repository=repository,
            clock=clock,
            as_of=reference,
        )
    except investigations.SessionNotFound:
        raise HTTPException(status_code=404, detail="Unknown session") from None
    except RepositoryError:
        raise HTTPException(
            status_code=503, detail=_INVESTIGATION_UNAVAILABLE
        ) from None
