"""Demo Lab payloads: the real investigation engine on in-memory scenarios.

Everything here is pass-through. Each scenario is built by the frozen
``investigation.agent.demo`` catalog, investigated by the real engine, and
reported verbatim. This module only reshapes the output into one JSON object
the dashboard can render: sessions (for charts), the engine trace, the
evidence registry, the grounded report, and the timeline text.

Nothing is invented here. No thresholds, statuses, or claims are authored,
and the data is always labelled ``data_source: "demo"``.
"""

from functools import lru_cache

from investigation.agent import (
    build_grounded_report,
    render_timeline_for,
    run_investigation,
)
from investigation.agent.demo import (
    DEFAULT_AS_OF,
    DEFAULT_USER,
    DEMO_DATASETS,
    DEMO_KEYS,
    DEMO_SCHEMA_VERSION,
    InMemoryRepository,
    demo_checkins,
)
from investigation.context import label_for_factor

from investigation.bridge import is_valid_stored_session

from .investigations import engine_details
from .ml_replay import ML_SOURCE, replay_anomalies
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


@lru_cache(maxsize=None)
def _scenario_run(key):
    """Build, score and investigate one scenario (deterministic, cached).

    ML results come from replaying the production Isolation Forest step over
    the scenario's history; the catalog's fixed values are used only when the
    ML runtime is unavailable.
    """
    sessions, catalog_anomalies, trigger = DEMO_DATASETS[key]["build"](DEFAULT_AS_OF)
    replayed = replay_anomalies(sessions)
    anomalies = catalog_anomalies if replayed is None else replayed
    ml_source = "catalog" if replayed is None else ML_SOURCE
    repository = InMemoryRepository(
        sessions=sessions, anomalies=anomalies, checkins=demo_checkins(key)
    )

    def clock():
        return DEFAULT_AS_OF

    result = run_investigation(
        DEFAULT_USER, trigger, repository=repository, clock=clock, as_of=DEFAULT_AS_OF
    )
    report = build_grounded_report(result, clock=clock)
    timeline = render_timeline_for(report, result, clock=clock)
    return sessions, anomalies, ml_source, result, report, timeline


def scenario_payload(key):
    """Run one scenario through the real engine and return a JSON-safe dict."""
    if key not in DEMO_DATASETS:
        raise UnknownScenario(key)

    sessions, anomalies, ml_source, result, report, timeline = _scenario_run(key)
    ordered = sorted(sessions, key=_row_order)

    return {
        "schema_version": DEMO_PAYLOAD_VERSION,
        "demo_schema_version": DEMO_SCHEMA_VERSION,
        "data_source": "demo",
        "engine_output": "real",
        "ml_source": ml_source,
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
        "stop_reason": result.stop_reason,
        **engine_details(result),
        "report": report.model_dump(mode="json"),
        "timeline": list(timeline),
    }
