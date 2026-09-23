"""Fixtures for the Phase 2 tests.

All data here is synthetic and exists only to exercise code paths. Stored row
shapes mirror the columns written by the existing pipeline (``typing_sessions``
and ``anomaly_results``), so the tests exercise the real mapping code.
"""

from datetime import datetime, timedelta, timezone

from investigation.repository import RepositoryError

DEFAULT_USER = "user-1"


def _dt(days_ago, hour=0):
    base = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return base.replace(hour=hour, minute=0, second=0, microsecond=0)


def make_session(
    session_id="S0001",
    user_id=DEFAULT_USER,
    days_ago=0,
    hour=0,
    dwell_mean=0.12,
    flight_mean=0.08,
    typing_speed=285.0,
    correction_rate=0.06,
    rhythm_variability=0.19,
    pause_count=4,
):
    """A stored typing-session row with typical values (stored units)."""
    start = _dt(days_ago, hour)
    end = start + timedelta(seconds=20)
    return {
        "id": session_id,
        "user_id": user_id,
        "session_start": start.isoformat(),
        "session_end": end.isoformat(),
        "dwell_mean": dwell_mean,
        "flight_mean": flight_mean,
        "typing_speed": typing_speed,
        "correction_rate": correction_rate,
        "rhythm_variability": rhythm_variability,
        "pause_count": pause_count,
    }


def make_invalid_session(**overrides):
    """A stored row that fails validation (typing_speed <= 0 by default)."""
    overrides.setdefault("typing_speed", 0.0)
    return make_session(**overrides)


def make_history(
    count,
    user_id=DEFAULT_USER,
    span_days=40,
    prefix="S",
    **overrides,
):
    """A list of valid sessions, oldest first, spread evenly over ``span_days``."""
    sessions = []
    for i in range(count):
        if count > 1:
            days_ago = span_days - (i * span_days / (count - 1))
        else:
            days_ago = 0
        sessions.append(
            make_session(
                session_id=f"{prefix}{i + 1:04d}",
                user_id=user_id,
                days_ago=days_ago,
                **overrides,
            )
        )
    return sessions


class FakeRepository:
    """In-memory read-only repository used in place of Supabase."""

    def __init__(self, sessions=None, anomalies=None):
        self.sessions = list(sessions or [])
        self.anomalies = dict(anomalies or {})

    def list_sessions(self, user_id):
        # Return copies so tests cannot mutate fixture state through a tool.
        return [dict(row) for row in self.sessions if row.get("user_id") == user_id]

    def get_anomaly_result(self, session_id):
        row = self.anomalies.get(session_id)
        return dict(row) if row else None


class FailingRepository:
    """Repository that simulates an unreachable data store."""

    def list_sessions(self, user_id):
        raise RepositoryError("database unavailable")

    def get_anomaly_result(self, session_id):
        raise RepositoryError("database unavailable")
