"""Shared support for the Phase 4 API tests (Phase 4).

Not a test module: unittest discovery only collects ``test*.py``, so this
file is imported, never run on its own. It provides the injected fakes the
routes and services need in place of a live store, the frozen clock for
deterministic runs, and the exact key sets of the documented contract.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

from tests import fixtures

#: The instant every deterministic Phase 4 run is pinned to.
FROZEN_NOW = datetime(2026, 9, 24, 9, 42, tzinfo=timezone.utc)


def frozen_clock():
    """A zero-argument clock pinned to FROZEN_NOW."""
    return FROZEN_NOW


#: Exact key sets of the documented read contract. A response carrying any
#: extra key — notably ``status`` or ``consistency`` — fails a contract test.
SESSION_KEYS = frozenset(
    {
        "session_id",
        "session_start",
        "session_end",
        "date",
        "typing_speed",
        "wpm",
        "dwell_mean_ms",
        "flight_mean_ms",
        "correction_rate",
        "rhythm_variability",
        "pause_count",
    }
)

BASELINE_KEYS = frozenset(
    {
        "typing_speed",
        "dwell_mean",
        "flight_mean",
        "correction_rate",
        "rhythm_variability",
        "pause_count",
        "wpm",
        "sample_count",
        "updated_at",
    }
)

INVESTIGATION_KEYS = frozenset(
    {
        "schema_version",
        "user_id",
        "session_id",
        "stop_reason",
        "evidence_digest",
        "report",
        "timeline",
        "limitations",
        "disclaimer",
    }
)


def baseline_row(user_id=fixtures.DEFAULT_USER, **overrides):
    """A stored ``baselines`` row shaped like routes/baseline.py writes it."""
    row = {
        "user_id": user_id,
        "typing_speed": 285.0,
        "dwell_mean": 0.12,
        "flight_mean": 0.08,
        "correction_rate": 0.06,
        "rhythm_variability": 0.19,
        "pause_count": 4.0,
        "sample_count": 12,
        "updated_at": "2026-09-20T08:00:00+00:00",
    }
    row.update(overrides)
    return row


class ReadOnlyRepository:
    """A repository wrapper that fails the test on any non-read access.

    Only the two protocol reads are implemented; any other attribute lookup
    raises immediately, so an attempted write can never go unnoticed.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def list_sessions(self, user_id):
        self.calls.append(("list_sessions", user_id))
        return self.inner.list_sessions(user_id)

    def get_anomaly_result(self, session_id):
        self.calls.append(("get_anomaly_result", session_id))
        return self.inner.get_anomaly_result(session_id)

    def __getattr__(self, name):
        raise AssertionError(f"unexpected repository access: {name}")


class FakeBaselineClient:
    """Just enough Supabase client for the read-only baseline SELECT.

    Only the chain the service performs is implemented
    (``table().select().eq().limit().execute()``); every step is logged in
    ``ops`` so tests can assert that nothing but reads happened, and any
    write method raises ``AttributeError`` loudly.
    """

    def __init__(self, rows=(), fail=False):
        self.rows = [dict(row) for row in rows]
        self.fail = fail
        self.tables = []
        self.ops = []

    def table(self, name):
        self.tables.append(name)
        return _FakeQuery(self, list(self.rows))


class _FakeQuery:
    """One chained read query against an in-memory row list."""

    def __init__(self, client, rows):
        self._client = client
        self._rows = rows

    def select(self, *_columns):
        self._client.ops.append("select")
        return self

    def eq(self, key, value):
        self._client.ops.append("eq")
        self._rows = [row for row in self._rows if row.get(key) == value]
        return self

    def limit(self, count):
        self._client.ops.append("limit")
        self._rows = self._rows[:count]
        return self

    def execute(self):
        self._client.ops.append("execute")
        if self._client.fail:
            raise RuntimeError("store unreachable")
        return SimpleNamespace(data=[dict(row) for row in self._rows])
