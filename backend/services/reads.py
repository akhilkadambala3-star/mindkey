"""Read-only projections for the documented dashboard endpoints (Phase 4).

Scope and invariants
--------------------
- **Read-only.** ``list_sessions`` / ``latest_session_id`` use only the
  ``SessionRepository`` protocol, and ``user_baseline`` performs a single
  SELECT against the ``baselines`` table. Nothing in this module writes.
- **One notion of a valid session.** Stored rows are filtered with
  ``investigation.bridge.is_valid_stored_session`` — the same validation the
  evidence layer and the Isolation Forest already use.
- **No duplicated unit math.** ``wpm`` and the millisecond fields come
  exclusively from ``investigation.units.to_wpm`` / ``to_milliseconds``
  (the dashboard's own presentation conversions).
- **No derived statuses.** The session projection deliberately emits neither
  ``status`` nor ``consistency``: neither exists server-side, and inventing
  either would add a decision this layer does not own.
- **Failures are explicit.** An unreachable store raises ``RepositoryError``;
  the route layer maps that to HTTP 503 with a generic detail.

Unit reminder (see ``investigation.units``): ``typing_speed`` is stored in
characters per minute, ``dwell_mean`` / ``flight_mean`` /
``rhythm_variability`` in seconds, ``correction_rate`` as a 0–1 ratio and
``pause_count`` as a count.
"""

from datetime import datetime

from investigation import units
from investigation.bridge import is_valid_stored_session
from investigation.repository import RepositoryError, default_repository


def _resolve(repository):
    """Return the injected repository, or construct the production one."""
    if repository is not None:
        return repository
    return default_repository()


def _timestamp(value):
    """JSON-safe stored timestamp: datetimes become ISO strings, others echo."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _row_order(row):
    """Sort key: oldest stored session first, ties broken by id (as strings)."""
    return (str(row.get("session_start") or ""), str(row.get("id") or ""))


def _identifiable(row):
    """A stored row can take part in the contract only if it has an id."""
    return row.get("id") is not None


def _project_session(row):
    """Map one valid stored row onto the documented session contract.

    Conversions reuse ``investigation.units``; stored values are echoed, not
    recomputed, and rounding stays a presentation concern. There is no
    ``status`` key and no ``consistency`` key — by Phase 4 decision.
    """
    typing_speed = row.get("typing_speed")
    start = _timestamp(row.get("session_start"))
    text = str(start) if start is not None else None
    return {
        "session_id": str(row["id"]),
        "session_start": start,
        "session_end": _timestamp(row.get("session_end")),
        "date": text[:10] if text and len(text) >= 10 else None,
        "typing_speed": typing_speed,
        "wpm": units.to_wpm(typing_speed),
        "dwell_mean_ms": units.to_milliseconds(row.get("dwell_mean")),
        "flight_mean_ms": units.to_milliseconds(row.get("flight_mean")),
        "correction_rate": row.get("correction_rate"),
        "rhythm_variability": row.get("rhythm_variability"),
        "pause_count": row.get("pause_count"),
    }


def list_sessions(user_id, repository=None):
    """Stored, valid sessions for ``user_id`` in the dashboard contract shape.

    Oldest first (``session_start``, ties broken by id). Unknown users and
    empty stores yield ``[]`` — an honest outcome, not an error. An
    unreachable store raises ``RepositoryError``.

    Rows failing ``is_valid_stored_session`` are omitted: the read model
    exposes measured features, never unvalidated data.
    """
    repo = _resolve(repository)
    rows = repo.list_sessions(user_id)
    sessions = [
        row for row in rows
        if _identifiable(row) and is_valid_stored_session(row)
    ]
    sessions.sort(key=_row_order)
    return [_project_session(row) for row in sessions]


def latest_session_id_for(rows):
    """Latest stored session id in ``rows`` (deterministic), or ``None``.

    Order mirrors ``list_sessions``: greatest ``session_start``, ties broken
    by id. Unlike the read model this considers *stored* rows, not only valid
    ones: an investigation may start from the true newest session and let the
    kernel report any data-quality problem itself.
    """
    candidates = [row for row in rows if _identifiable(row)]
    if not candidates:
        return None
    return str(max(candidates, key=_row_order)["id"])


def latest_session_id(user_id, repository=None):
    """The id of the latest stored session for ``user_id``, or ``None``."""
    repo = _resolve(repository)
    return latest_session_id_for(repo.list_sessions(user_id))


def baseline_client():
    """The Supabase client used for the read-only baseline SELECT.

    Imported lazily so this module never requires environment variables at
    import time; an unavailable client raises ``RepositoryError``.
    """
    try:
        from database import supabase  # noqa: WPS433 (intentional lazy import)
    except Exception as exc:  # env missing / client construction failed
        raise RepositoryError(f"Supabase client is unavailable: {exc}") from exc
    return supabase


def _empty_baseline():
    """The dashboard-defensive baseline: zero samples, every feature null."""
    return {
        "typing_speed": None,
        "dwell_mean": None,
        "flight_mean": None,
        "correction_rate": None,
        "rhythm_variability": None,
        "pause_count": None,
        "wpm": None,
        "sample_count": 0,
        "updated_at": None,
    }


def _project_baseline(row):
    """Map the stored ``baselines`` row onto the documented contract.

    ``wpm`` comes from ``investigation.units.to_wpm`` — the single source of
    the CPM → WPM conversion.
    """
    typing_speed = row.get("typing_speed")
    sample_count = row.get("sample_count")
    try:
        sample_count = 0 if sample_count is None else int(sample_count)
    except (TypeError, ValueError):
        sample_count = 0
    return {
        "typing_speed": typing_speed,
        "dwell_mean": row.get("dwell_mean"),
        "flight_mean": row.get("flight_mean"),
        "correction_rate": row.get("correction_rate"),
        "rhythm_variability": row.get("rhythm_variability"),
        "pause_count": row.get("pause_count"),
        "wpm": units.to_wpm(typing_speed),
        "sample_count": sample_count,
        "updated_at": row.get("updated_at"),
    }


def user_baseline(user_id, client=None):
    """The stored personal baseline for ``user_id`` (read-only SELECT).

    No baseline row is an honest outcome, not an error: the contract shape is
    returned with ``sample_count`` 0 and null features so the dashboard can
    render "no baseline yet". An unreachable store raises ``RepositoryError``.
    """
    if client is None:
        client = baseline_client()
    try:
        result = (
            client.table("baselines")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise RepositoryError("Failed to read baseline") from exc
    rows = result.data or []
    if not rows:
        return _empty_baseline()
    return _project_baseline(rows[0])
