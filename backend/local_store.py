"""A local, file-backed stand-in for the Supabase client.

Why this exists: the dashboard's live mode, the desktop agent and the
investigation API all talk to Supabase. At a hackathon venue (or on a
teammate's laptop without the project keys) that makes live mode impossible
to demo. Setting ``MINDKEY_STORE=local`` swaps the Supabase client for this
SQLite-backed client, so the whole pipeline runs offline:

    desktop agent -> POST /typing/session -> ML -> investigation -> dashboard

It implements exactly the subset of the supabase-py query builder that the
backend uses (``table().select().eq().limit().order().execute()``,
``insert()``, ``update().eq()`` and ``delete().eq()``). Every other code path
is unchanged, because routes keep importing ``supabase`` from ``database``.

Storage: one SQLite file, one table per logical table, each row stored as
JSON with a generated ``id`` (uuid4) and ``created_at``. Only the table names
the backend knows are accepted.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

#: Logical tables the backend reads and writes.
KNOWN_TABLES = frozenset(
    {"typing_sessions", "anomaly_results", "baselines", "checkins", "users"}
)


class LocalResult:
    """Mirror of supabase-py's response object (only ``data`` is used)."""

    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, client, table):
        self._client = client
        self._table = table
        self._op = "select"
        self._columns = None
        self._filters = []
        self._limit = None
        self._order = None
        self._payload = None

    # -- builders ------------------------------------------------------------
    def select(self, columns="*"):
        self._op = "select"
        cols = [c.strip() for c in str(columns).split(",") if c.strip()]
        self._columns = None if cols in ([], ["*"]) else cols
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = dict(payload)
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, column, value):
        self._filters.append((column, value))
        return self

    def limit(self, count):
        self._limit = int(count)
        return self

    def order(self, column, desc=False):
        self._order = (column, bool(desc))
        return self

    # -- execution -----------------------------------------------------------
    def _matches(self, row):
        return all(str(row.get(col)) == str(val) for col, val in self._filters)

    def execute(self):
        return self._client._execute(self)


class LocalClient:
    """SQLite-backed client with the supabase-py surface MindKey uses."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock:
            for name in sorted(KNOWN_TABLES):
                self._conn.execute(
                    f'CREATE TABLE IF NOT EXISTS "{name}" '
                    "(id TEXT PRIMARY KEY, seq INTEGER, body TEXT NOT NULL)"
                )
            self._conn.commit()

    def table(self, name):
        if name not in KNOWN_TABLES:
            raise ValueError(f"unknown table: {name!r}")
        return _Query(self, name)

    # -- internals -----------------------------------------------------------
    def _rows(self, table):
        cur = self._conn.execute(f'SELECT body FROM "{table}" ORDER BY seq')
        return [json.loads(body) for (body,) in cur.fetchall()]

    def _write(self, table, row):
        self._conn.execute(
            f'INSERT OR REPLACE INTO "{table}" (id, seq, body) VALUES '
            f'(?, COALESCE((SELECT seq FROM "{table}" WHERE id = ?), '
            f'(SELECT COALESCE(MAX(seq), 0) + 1 FROM "{table}")), ?)',
            (row["id"], row["id"], json.dumps(row, default=str)),
        )

    def _execute(self, q):
        with self._lock:
            if q._op == "insert":
                payload = q._payload if isinstance(q._payload, list) else [q._payload]
                inserted = []
                now = datetime.now(timezone.utc).isoformat()
                for item in payload:
                    row = dict(item)
                    row.setdefault("id", str(uuid.uuid4()))
                    row.setdefault("created_at", now)
                    row["id"] = str(row["id"])
                    self._write(q._table, row)
                    inserted.append(row)
                self._conn.commit()
                return LocalResult(inserted)

            rows = [r for r in self._rows(q._table) if q._matches(r)]

            if q._op == "update":
                updated = []
                for row in rows:
                    row.update(q._payload)
                    self._write(q._table, row)
                    updated.append(row)
                self._conn.commit()
                return LocalResult(updated)

            if q._op == "delete":
                for row in rows:
                    self._conn.execute(f'DELETE FROM "{q._table}" WHERE id = ?', (row["id"],))
                self._conn.commit()
                return LocalResult(rows)

            if q._order is not None:
                column, desc = q._order
                rows.sort(key=lambda r: (r.get(column) is None, str(r.get(column))), reverse=desc)
            if q._limit is not None:
                rows = rows[: q._limit]
            if q._columns is not None:
                rows = [{c: r.get(c) for c in q._columns} for r in rows]
            return LocalResult(rows)


def default_local_path():
    return Path(__file__).resolve().parent / ".local" / "mindkey.db"
