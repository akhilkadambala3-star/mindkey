"""Reproducible offline demo (Phase 3.5).

A small, self-contained driver that runs one deterministic investigation over
synthetic behavioral data and renders the resulting report and investigation
timeline as plain text.

Why this lives here
-------------------
The evaluation catalog (``tests/agent/scenarios.py``) asserts *expected*
behavior; this module is the product-side, runnable demo. It therefore depends
only on the production layers (``investigation`` and ``investigation.agent``)
and never imports anything from ``tests``.

Guarantees
----------
- Fully offline: every dataset lives in memory and the demo never constructs
  the production repository, so no Supabase client, connection or environment
  variable is ever needed or contacted, and no database is read or written.
- Deterministic and reproducible: the reference time (``as_of``) and the trace
  clock are both injectable and default to a fixed constant, so the same
  scenario renders byte-identical evidence, report, trace and timeline.
- Plain text only: no HTML, no dashboard, no UI, no new dependencies.
- Privacy and safety: only structured numeric behavioral features, ids,
  timestamps and counts are used. No typed content, no keystrokes, no raw
  session content, no medical language and no diagnosis.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .engine import ALT_WINDOWS, MAX_ITERATIONS, run_investigation
from .report import GroundedReport, build_grounded_report
from .timeline import render, render_report_lines, render_timeline_for

#: Version of the demo payload shape.
DEMO_SCHEMA_VERSION = "1.0"

#: Fixed demo user id (never a real one).
DEFAULT_USER = "demo-user"

#: Fixed reference time so the demo is reproducible without configuration.
DEFAULT_AS_OF = datetime(2026, 9, 24, 9, 42, tzinfo=timezone.utc)

#: Typical ("baseline") stored feature values.
_BASE_FEATURES = {
    "dwell_mean": 0.12,
    "flight_mean": 0.08,
    "typing_speed": 285.0,
    "correction_rate": 0.06,
    "rhythm_variability": 0.19,
    "pause_count": 4,
}

#: Deviating stored feature values used by the "persistent change" scenario.
_DEVIATING_FEATURES = {
    "dwell_mean": 0.20,
    "flight_mean": 0.14,
    "typing_speed": 200.0,
    "correction_rate": 0.12,
    "rhythm_variability": 0.30,
    "pause_count": 9,
}


class InMemoryRepository:
    """Read-only, in-memory stand-in for the stored-session repository.

    Rows mirror the columns written by the existing pipeline, so the same
    mapping code is exercised without touching a database.
    """

    def __init__(self, sessions=None, anomalies=None):
        self.sessions = [dict(row) for row in (sessions or [])]
        self.anomalies = dict(anomalies or {})

    def list_sessions(self, user_id):
        return [dict(row) for row in self.sessions if row.get("user_id") == user_id]

    def get_anomaly_result(self, session_id):
        row = self.anomalies.get(session_id)
        return dict(row) if row else None


def _session(session_id, as_of, days_ago, **features):
    """One synthetic stored session row, timestamped relative to ``as_of``."""
    start = (as_of - timedelta(days=days_ago)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = start + timedelta(seconds=20)
    row = {
        "id": session_id,
        "user_id": DEFAULT_USER,
        "session_start": start.isoformat(),
        "session_end": end.isoformat(),
    }
    row.update(_BASE_FEATURES)
    row.update(features)
    return row


def _consistent(as_of):
    """Thirty stable sessions: recent patterns match the baseline."""
    sessions = [
        _session(f"S{i:04d}", as_of, i, **_BASE_FEATURES) for i in range(1, 31)
    ]
    return sessions, {}, "S0001"


def _recent_variation(as_of):
    """A single-signal recent shift that does not persist."""
    sessions = [
        _session(f"S{i:04d}", as_of, i, **_BASE_FEATURES) for i in range(7, 36)
    ]
    sessions += [
        _session(f"R{i:04d}", as_of, i, typing_speed=233.0) for i in range(1, 6)
    ]
    return sessions, {}, "R0001"


def _persistent_change(as_of):
    """A multi-signal recent shift plus an anomaly flag on the trigger."""
    sessions = [
        _session(f"S{i:04d}", as_of, i, **_BASE_FEATURES) for i in range(7, 36)
    ]
    sessions += [
        _session(f"R{i:04d}", as_of, i, **_DEVIATING_FEATURES) for i in range(1, 6)
    ]
    anomalies = {"R0001": {"anomaly_score": 0.62, "is_anomaly": True}}
    return sessions, anomalies, "R0001"


def _insufficient_history(as_of):
    """Only three stored sessions: below the model minimum."""
    sessions = [_session(f"S{i:04d}", as_of, i, **_BASE_FEATURES) for i in range(3)]
    return sessions, {}, "S0001"


def _invalid_data(as_of):
    """A dense valid history plus one row that fails validation."""
    sessions = [
        _session(f"S{i:04d}", as_of, i, **_BASE_FEATURES) for i in range(1, 20)
    ]
    sessions.append(_session("BAD1", as_of, 3, typing_speed=0.0))
    return sessions, {}, "S0001"


#: The fixed demo catalog, in a stable order.
DEMO_DATASETS = {
    "consistent": {
        "title": "Consistent with baseline",
        "description": "Thirty stable sessions with no recent deviation.",
        "build": _consistent,
    },
    "recent_variation": {
        "title": "Recent variation",
        "description": "A single-signal recent shift that does not persist.",
        "build": _recent_variation,
    },
    "persistent_change": {
        "title": "Persistent change",
        "description": (
            "A multi-signal recent shift plus an anomaly flag on the trigger."
        ),
        "build": _persistent_change,
    },
    "insufficient_history": {
        "title": "Insufficient history",
        "description": "Only three stored sessions: below the model minimum.",
        "build": _insufficient_history,
    },
    "invalid_data": {
        "title": "Data-quality issue",
        "description": (
            "A dense valid history plus one stored row that fails validation."
        ),
        "build": _invalid_data,
    },
}

#: The demo keys, in catalog order.
DEMO_KEYS = tuple(DEMO_DATASETS)


@dataclass(frozen=True)
class DemoRun:
    """One rendered demo run: structured result plus plain-text renderings."""

    key: str
    title: str
    result: object
    report: GroundedReport
    timeline: list
    report_lines: list


def run_demo(
    dataset="persistent_change",
    *,
    user_id=DEFAULT_USER,
    clock=None,
    as_of=None,
    alt_windows=ALT_WINDOWS,
    max_iterations=MAX_ITERATIONS,
):
    """Run one demo scenario over in-memory data and render it.

    Args:
        dataset: a key of :data:`DEMO_DATASETS`.
        user_id: the demo user whose in-memory history is investigated.
        clock: optional zero-argument clock. Defaults to a frozen clock pinned
            to :data:`DEFAULT_AS_OF`, which makes the trace deterministic.
        as_of: the reference time for windowed analysis. Defaults to
            :data:`DEFAULT_AS_OF`.
        alt_windows: comparison-window pairs used for robustness.
        max_iterations: deterministic collection bound.

    Returns:
        :class:`DemoRun`. Nothing is written anywhere.
    """
    if dataset not in DEMO_DATASETS:
        raise KeyError(f"unknown demo scenario: {dataset!r}")

    as_of = DEFAULT_AS_OF if as_of is None else as_of
    clock = (lambda: DEFAULT_AS_OF) if clock is None else clock

    sessions, anomalies, session_id = DEMO_DATASETS[dataset]["build"](as_of)
    repository = InMemoryRepository(sessions=sessions, anomalies=anomalies)

    result = run_investigation(
        user_id,
        session_id,
        repository=repository,
        clock=clock,
        as_of=as_of,
        alt_windows=alt_windows,
        max_iterations=max_iterations,
    )
    report = build_grounded_report(result, clock=clock)

    return DemoRun(
        key=dataset,
        title=DEMO_DATASETS[dataset]["title"],
        result=result,
        report=report,
        timeline=render_timeline_for(report, result, clock=clock),
        report_lines=render_report_lines(report),
    )


def render_demo(run):
    """Render a :class:`DemoRun` as the full plain-text report plus timeline."""
    return render(run.report, run.result)


def format_run(run, *, timeline=False):
    """Render one demo run for the CLI (plain text)."""
    if timeline:
        return "\n".join(run.timeline)
    return render_demo(run)


def demo_payload(run):
    """The JSON-serializable form of one demo run."""
    return {
        "schema_version": DEMO_SCHEMA_VERSION,
        "key": run.key,
        "title": run.title,
        "stop_reason": run.result.stop_reason,
        "report": run.report.model_dump(mode="json"),
        "timeline": list(run.timeline),
        "report_lines": list(run.report_lines),
    }


def build_parser():
    """Build the demo command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m investigation.agent.demo",
        description=(
            "Run a deterministic, offline MindKey investigation demo and print "
            "a plain-text report and timeline. Not a diagnosis."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=DEMO_KEYS,
        help="scenario to run (repeatable); defaults to every scenario",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the available scenarios and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of plain text",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="print only the investigation timeline",
    )
    return parser


def _enable_utf8_output():
    """Best-effort UTF-8 stdout so the fixed report text always prints.

    The grounded claim templates contain a non-ASCII arrow; consoles that
    default to a legacy code page would otherwise raise on print. Streams that
    cannot be reconfigured (e.g. a test's ``StringIO``) are left alone.
    """
    stream = getattr(sys, "stdout", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8")
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        pass


def main(argv=None):
    """Run the demo CLI. Returns a process exit code."""
    _enable_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for key in DEMO_KEYS:
            entry = DEMO_DATASETS[key]
            print(f"{key}: {entry['title']} - {entry['description']}")
        return 0

    keys = list(args.scenario) if args.scenario else list(DEMO_KEYS)
    runs = [run_demo(key) for key in keys]

    if args.json:
        payload = {
            "schema_version": DEMO_SCHEMA_VERSION,
            "runs": [demo_payload(run) for run in runs],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for position, run in enumerate(runs):
        if position:
            print("")
        print(f"=== {run.key}: {run.title} ===")
        print(format_run(run, timeline=args.timeline))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
