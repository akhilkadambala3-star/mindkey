"""Demo Lab payloads: the real investigation engine on in-memory scenarios.

Everything here is pass-through. Each scenario is built by the frozen
``investigation.agent.demo`` catalog, investigated by the real engine, and
reported verbatim. This module only reshapes the output into one JSON object
the dashboard can render: sessions (for charts), the engine trace, the
evidence registry, the grounded report, and the timeline text.

Nothing is invented here. No thresholds, statuses, or claims are authored,
and the data is always labelled ``data_source: "demo"``.
"""

from investigation.agent.demo import (
    DEFAULT_AS_OF,
    DEMO_DATASETS,
    DEMO_KEYS,
    DEMO_SCHEMA_VERSION,
    demo_checkins,
    run_demo,
)
from investigation.context import label_for_factor

from investigation.bridge import is_valid_stored_session

from .reads import _project_session, _row_order

DEMO_PAYLOAD_VERSION = "1.0"


class UnknownScenario(KeyError):
    """The requested demo scenario key does not exist."""


def list_scenarios():
    """The demo catalog in stable order: key, title, description."""
    return [
        {
            "key": key,
            "title": DEMO_DATASETS[key]["title"],
            "description": DEMO_DATASETS[key]["description"],
        }
        for key in DEMO_KEYS
    ]


def scenario_payload(key):
    """Run one scenario through the real engine and return a JSON-safe dict."""
    if key not in DEMO_DATASETS:
        raise UnknownScenario(key)

    sessions, anomalies, _trigger = DEMO_DATASETS[key]["build"](DEFAULT_AS_OF)
    run = run_demo(key)
    result = run.result.model_dump(mode="json")
    state = result.get("state") or {}
    ordered = sorted(sessions, key=_row_order)

    return {
        "schema_version": DEMO_PAYLOAD_VERSION,
        "demo_schema_version": DEMO_SCHEMA_VERSION,
        "data_source": "demo",
        "engine_output": "real",
        "key": key,
        "title": DEMO_DATASETS[key]["title"],
        "description": DEMO_DATASETS[key]["description"],
        "as_of": DEFAULT_AS_OF.isoformat(),
        "sessions": [
            dict(
                _project_session(row),
                anomaly=anomalies.get(row.get("id")),
                is_valid=bool(is_valid_stored_session(row)),
            )
            for row in ordered
        ],
        "checkins": [
            {"date": c["date"], "factor": c["factor"], "label": label_for_factor(c["factor"])}
            for c in demo_checkins(key)
        ],
        "stop_reason": result.get("stop_reason"),
        "ml_evidence": state.get("ml_evidence"),
        "investigation_plan": state.get("investigation_plan") or [],
        "tools_called": state.get("tools_called") or [],
        "tool_calls": result.get("tool_calls") or [],
        "iterations": result.get("iterations"),
        "final_assessment": result.get("final_assessment") or {},
        "evidence": state.get("evidence") or [],
        "trace": state.get("trace") or [],
        "report": run.report.model_dump(mode="json"),
        "timeline": list(run.timeline),
    }
