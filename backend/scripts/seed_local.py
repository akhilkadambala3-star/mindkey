"""Seed the offline local store with a demo scenario, dated to end today.

Usage (from backend/):

    MINDKEY_STORE=local python scripts/seed_local.py --scenario persistent_change

Then run the API with the same MINDKEY_STORE=local and open the dashboard in
live mode:

    http://localhost:5500/?api=http://localhost:8000&user=<printed user id>

What it writes (all synthetic, from the engine's demo catalog):
- the scenario's typing sessions, shifted so the latest one is today,
- the ML anomaly results the real Isolation Forest gives them (replayed in
  order, as the typing route would have stored them),
- the scenario's check-ins (same shift),
- the personal baseline, computed by the real /baseline route logic.

It never touches Supabase: the script refuses to run unless the store is
local. ``--reset`` first removes this user's existing rows.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

#: Fixed id for the seeded demo user (a UUID, so it also fits uuid columns).
DEFAULT_LOCAL_USER = "00000000-0000-4000-8000-000000000001"

FEATURES = (
    "dwell_mean",
    "flight_mean",
    "typing_speed",
    "correction_rate",
    "rhythm_variability",
    "pause_count",
)


def _shift_iso(value, delta):
    return (datetime.fromisoformat(value) + delta).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="persistent_change")
    parser.add_argument("--user", default=DEFAULT_LOCAL_USER)
    parser.add_argument("--reset", action="store_true", help="delete this user's rows first")
    args = parser.parse_args(argv)

    os.environ.setdefault("MINDKEY_STORE", "local")
    if os.environ["MINDKEY_STORE"].strip().lower() != "local":
        parser.error("refusing to seed: MINDKEY_STORE must be 'local'")

    from database import supabase as client  # noqa: E402  (after env is set)
    from investigation.agent.demo import DEFAULT_AS_OF, DEMO_DATASETS, demo_checkins
    from routes.baseline import create_or_update_baseline

    if args.scenario not in DEMO_DATASETS:
        parser.error(f"unknown scenario; choose from {', '.join(DEMO_DATASETS)}")

    sessions, anomalies, _ = DEMO_DATASETS[args.scenario]["build"](DEFAULT_AS_OF)
    # Score every session with the real model, as production would have.
    from services.ml_replay import replay_anomalies

    replayed = replay_anomalies(sessions)
    if replayed is not None:
        anomalies = replayed
    checkins = demo_checkins(args.scenario)

    # Shift whole days so the scenario's reference time lands on "now".
    now = datetime.now(timezone.utc)
    delta = timedelta(days=(now.date() - DEFAULT_AS_OF.date()).days)

    if args.reset:
        for table in ("typing_sessions", "anomaly_results", "checkins", "baselines"):
            client.table(table).delete().eq("user_id", args.user).execute()

    id_map = {}
    for row in sorted(sessions, key=lambda r: r["session_start"]):
        record = {k: row[k] for k in FEATURES}
        record.update(
            user_id=args.user,
            session_start=_shift_iso(row["session_start"], delta),
            session_end=_shift_iso(row["session_end"], delta),
        )
        stored = client.table("typing_sessions").insert(record).execute().data[0]
        id_map[row["id"]] = stored["id"]

    for demo_id, result in anomalies.items():
        client.table("anomaly_results").insert(
            {
                "user_id": args.user,
                "session_id": id_map[demo_id],
                "anomaly_score": result["anomaly_score"],
                "is_anomaly": result["is_anomaly"],
            }
        ).execute()

    for c in checkins:
        day = (datetime.fromisoformat(c["date"]) + delta).date().isoformat()
        client.table("checkins").insert(
            {"user_id": args.user, "date": day, "factor": c["factor"], "note": None}
        ).execute()

    try:
        create_or_update_baseline(args.user)
        baseline_note = "baseline stored"
    except Exception as exc:  # fewer than 3 valid sessions is a normal outcome
        baseline_note = f"no baseline ({getattr(exc, 'detail', exc)})"

    print(
        f"Seeded '{args.scenario}' for user {args.user}: {len(sessions)} sessions, "
        f"{len(anomalies)} anomaly result(s), {len(checkins)} check-in(s), {baseline_note}."
    )
    print(f"Dashboard: http://localhost:5500/?api=http://localhost:8000&user={args.user}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
