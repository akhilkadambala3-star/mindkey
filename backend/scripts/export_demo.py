"""Export real engine output for every demo scenario to the static dashboard.

Usage (from backend/):  python scripts/export_demo.py

Writes dashboard/demo/index.json plus one <key>.json per scenario. The static
Vercel deployment then shows genuine investigation traces without a running
backend. Re-run after any engine change so the dashboard never drifts.
"""

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from services import demo  # noqa: E402

OUT = BACKEND.parent / "dashboard" / "demo"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = demo.list_scenarios()
    (OUT / "index.json").write_text(json.dumps(scenarios, indent=1) + "\n")
    for entry in scenarios:
        payload = demo.scenario_payload(entry["key"])
        (OUT / f"{entry['key']}.json").write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
        )
        print(f"wrote {entry['key']}.json  ({payload['report']['conclusion']['status']})")


if __name__ == "__main__":
    main()
