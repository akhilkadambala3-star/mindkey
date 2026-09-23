"""Read-only data access for the evidence layer (Phase 2).

The evidence layer must be independently testable without a live Supabase
instance. It therefore depends on a small ``SessionRepository`` protocol:

- Production code uses :class:`SupabaseSessionRepository`, which imports the
  Supabase client **lazily** so that importing this package (or running the
  unit tests) never requires environment variables or network access.
- Tests and the demo supply a fake in-memory implementation.

The repository is strictly read-only: it never inserts, updates, or deletes.
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class RepositoryError(Exception):
    """Raised when the underlying data store cannot be reached or read."""


class SessionRepository(Protocol):
    """Read-only access to stored typing sessions and anomaly results."""

    def list_sessions(self, user_id: str) -> list[dict]:
        """Return all stored typing-session rows for ``user_id``."""
        ...

    def get_anomaly_result(self, session_id: str) -> dict | None:
        """Return the stored anomaly-result row for ``session_id``, if any."""
        ...


class SupabaseSessionRepository:
    """Read-only Supabase implementation of :class:`SessionRepository`."""

    def __init__(self, client=None):
        if client is not None:
            self._client = client
            return

        # Lazy import: keeps this package importable without Supabase env vars.
        try:
            from database import supabase  # noqa: WPS433 (intentional lazy import)
        except Exception as exc:  # env missing / client construction failed
            raise RepositoryError(
                f"Supabase client is unavailable: {exc}"
            ) from exc

        self._client = supabase

    def list_sessions(self, user_id: str) -> list[dict]:
        try:
            result = (
                self._client.table("typing_sessions")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
        except Exception as exc:
            logger.warning("Failed to read typing_sessions for a user: %s", exc)
            raise RepositoryError("Failed to read typing sessions") from exc

        return list(result.data or [])

    def get_anomaly_result(self, session_id: str) -> dict | None:
        try:
            result = (
                self._client.table("anomaly_results")
                .select("*")
                .eq("session_id", session_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            logger.warning("Failed to read anomaly_results for a session: %s", exc)
            raise RepositoryError("Failed to read anomaly result") from exc

        rows = result.data or []
        return dict(rows[0]) if rows else None


def default_repository():
    """Construct the production repository (Supabase, lazily imported)."""
    return SupabaseSessionRepository()
