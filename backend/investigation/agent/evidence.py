"""Evidence registry, deterministic claim templates, and the caching read-only
repository wrapper (Phase 3.1).

Why this module exists
----------------------
Every statement the investigation layer will ever make has to come from a
measured value. This module is what makes that structural rather than a matter
of discipline:

1. :class:`EvidenceItem` is the only shape a fact can take. It carries numbers,
   units, the tool it came from, and a rendered ``statement``.
2. :class:`EvidenceRegistry` is the only producer of items, and **the only way
   to create one is through a named template** in :data:`CLAIM_TEMPLATES`. There
   is no API that accepts free-form text, so a fact cannot be invented at the
   point of writing -- only rendered from values.
3. Templates are pure functions of the item's own fields. Given the same fields
   they produce byte-identical text, which is what lets the (later) evidence
   critic verify each claim against the evidence it cites.

Field conventions (per ``kind``)
--------------------------------
``value`` carries the single most important number for the item:

- ``signal_change`` / ``signal_change_absolute``: the session's current value
- ``window_stat``: the window mean
- ``session_count``: the number of valid sessions
- ``quality`` / ``quality_invalid``: the valid / invalid session count
- ``anomaly``: the stored anomaly score
- ``persistence``: the number of moved signals
- ``persistence_depth``: the number of deviating sessions in the trailing run
- ``persistence_robustness``: the number of conflicting signal directions

``status`` carries a stored label where one exists: the persistence status
(e.g. ``"recent_variation"``), the stored anomaly flag rendered verbatim as
``"true"`` / ``"false"``, or ``"unknown"``.

Privacy
-------
The builders here read only the structured Phase 2 objects they are given (the
six numeric behavioral features, ids, timestamps, counts and statuses). No typed
content is read, and no new data source is touched: this module performs no I/O
and never calls a tool or a repository.
"""

import hashlib
import json
from datetime import datetime
from typing import Callable, Literal

from pydantic import BaseModel, Field

from ..contracts import BehavioralEvidence, Direction
from ..tools import MLEvidenceOutput

#: Version of the *agent* evidence item shape. Deliberately separate from
#: ``contracts.SCHEMA_VERSION`` (which versions the ML <-> Agent evidence
#: package): the two evolve independently.
EVIDENCE_SCHEMA_VERSION = "1.0"

#: Source-tool labels recorded on items, matching the Phase 2 tool names.
SOURCE_ADAPTER = "build_behavioral_evidence"
SOURCE_ML = "get_ml_evidence"

EvidenceKind = Literal[
    "signal_change",
    "signal_change_absolute",
    "signal_unavailable",
    "window_stat",
    "session_count",
    "quality",
    "quality_invalid",
    "anomaly",
    "anomaly_unavailable",
    "persistence",
    "persistence_depth",
    "persistence_robustness",
    "context_absence",
    "context_present",
    "tool_unavailable",
    "cadence",
]


class EvidenceItem(BaseModel):
    """One measured, citable fact about a user's behavioral evidence.

    ``statement`` is always rendered from a template in
    :data:`CLAIM_TEMPLATES`; it is never authored by hand.
    """

    id: str = Field(min_length=1)
    kind: EvidenceKind
    statement: str = Field(min_length=1)
    source_tool: str = Field(min_length=1)

    key: str | None = None
    label: str | None = None
    unit: str | None = None
    direction: Direction | None = None

    value: float | None = None
    baseline_value: float | None = None
    absolute_change: float | None = None
    relative_change: float | None = None

    session_count: int | None = None
    window_days: int | None = None
    threshold: int | None = None
    status: str | None = None
    onset: datetime | None = None

    available: bool = True
    unavailable_reason: str | None = None
    schema_version: str = EVIDENCE_SCHEMA_VERSION


#: Field names a template may reference. Anything else is rejected, which keeps
#: every item structurally complete for the later critic.
_EVIDENCE_FIELDS = frozenset(EvidenceItem.model_fields)


# ---------------------------------------------------------------------------
# Deterministic formatting helpers
# ---------------------------------------------------------------------------


def format_value(value, unit=None, places=2):
    """Format a measurement for a statement. Missing values are explicit."""
    if value is None:
        return "unavailable"
    if unit == "count":
        try:
            return str(int(round(float(value))))
        except (TypeError, ValueError):
            return "unavailable"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "unavailable"


def format_count(value):
    """Format a whole-number count (session counts, moved signals)."""
    if value is None:
        return "unavailable"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "unavailable"


def format_percent(relative_change, places=1):
    """Format a relative change as an unsigned percentage, or ``None``."""
    if relative_change is None:
        return None
    try:
        return f"{abs(float(relative_change)) * 100:.{places}f}%"
    except (TypeError, ValueError):
        return None


def format_date(value):
    """Format a stored timestamp as a date. Missing values are explicit."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    return "an unknown date"


def _quantity(value, unit):
    """Render a value with its unit, e.g. ``"285.00 characters per minute"``."""
    text = format_value(value, unit)
    return f"{text} {unit}".strip() if unit else text


def _label(fields):
    return fields.get("label") or fields.get("key") or "Signal"


def _window_text(fields):
    window = fields.get("window_days")
    return f"{window}-day" if window else "historical"


# ---------------------------------------------------------------------------
# Deterministic claim templates
# ---------------------------------------------------------------------------


def _t_signal_change(fields):
    word = {"increase": "increased", "decrease": "decreased"}.get(
        fields.get("direction"), "changed"
    )
    percent = format_percent(fields.get("relative_change"))
    percent_text = percent if percent is not None else "by an unmeasured amount"
    baseline = format_value(fields.get("baseline_value"), fields.get("unit"))
    return (
        f"{_label(fields)} {word} {percent_text} from the "
        f"{_window_text(fields)} baseline mean ({baseline} \u2192 "
        f"{_quantity(fields.get('value'), fields.get('unit'))})."
    )


def _t_signal_change_absolute(fields):
    unit = fields.get("unit")
    return (
        f"{_label(fields)} changed by {format_value(fields.get('absolute_change'), unit)} "
        f"from the {_window_text(fields)} baseline mean "
        f"({format_value(fields.get('baseline_value'), unit)} \u2192 "
        f"{_quantity(fields.get('value'), unit)})."
    )


def _t_signal_unavailable(fields):
    reason = fields.get("unavailable_reason") or "no value was available"
    return f"{_label(fields)} is unavailable; {reason}."


def _t_window_stat(fields):
    return (
        f"The {_window_text(fields)} {_label(fields)} mean is "
        f"{_quantity(fields.get('value'), fields.get('unit'))} across "
        f"{format_count(fields.get('session_count'))} valid session(s)."
    )


def _t_session_count(fields):
    return (
        f"{format_count(fields.get('session_count'))} valid session(s) are available; "
        f"at least {format_count(fields.get('threshold'))} are required for "
        "model-based evidence."
    )


def _t_quality(fields):
    return (
        f"{format_count(fields.get('value'))} of {format_count(fields.get('session_count'))} "
        "stored session(s) are valid behavioral evidence."
    )


def _t_quality_invalid(fields):
    return (
        f"{format_count(fields.get('value'))} of {format_count(fields.get('session_count'))} "
        "stored session(s) failed validation and were excluded from windowed comparisons."
    )


def _t_anomaly(fields):
    score = format_value(fields.get("value"), places=4)
    status = fields.get("status") or "unknown"
    return (
        f"The stored anomaly score for this session is {score}; the stored flag is "
        f"{status}. An anomaly score is not a diagnosis."
    )


def _t_anomaly_unavailable(fields):
    reason = fields.get("unavailable_reason") or "unavailable"
    return f"No stored ML anomaly result is available for this session ({reason})."


def _t_persistence(fields):
    status = fields.get("status") or "unknown"
    return (
        f'The deterministic persistence assessment is "{status}" from '
        f"{format_count(fields.get('value'))} moved signal(s) over "
        f"{format_count(fields.get('session_count'))} recent valid session(s)."
    )


def _t_persistence_depth(fields):
    if fields.get("available") is False:
        reason = fields.get("unavailable_reason") or "the recent history was too thin"
        return f"Consecutive deviating sessions could not be measured; {reason}."
    count = format_count(fields.get("session_count"))
    window = fields.get("window_days")
    window_text = (
        f"the last {format_count(window)} day(s)" if window else "the recent window"
    )
    return (
        f"{format_count(fields.get('value'))} consecutive valid session(s) of {count} "
        f"in {window_text} deviate from the baseline mean; the run began on "
        f"{format_date(fields.get('onset'))}."
    )


def _t_persistence_robustness(fields):
    status = fields.get("status") or "unknown"
    comparisons = format_count(fields.get("session_count"))
    if status == "agrees":
        return (
            f"The persistence result is consistent with {comparisons} alternative "
            "window comparison(s)."
        )
    if status == "disagrees":
        return (
            "Alternative window comparison(s) disagree with the default recent/baseline "
            f"windows ({format_count(fields.get('value'))} conflicting signal "
            "direction(s)); the persistence claim is downgraded accordingly."
        )
    return (
        f"Window robustness was not assessed ({comparisons} alternative comparison(s) "
        "available); persistence rests on the default windows only."
    )


def _t_context_absence(fields):
    return (
        "No server-side contextual factors (sleep, fatigue, stress) are stored, "
        "so they cannot be confirmed or ruled out."
    )


def _t_context_present(fields):
    source = fields.get("status") or "unknown"
    return (
        f"User-reported contextual factors are available for this session "
        f"(source: {source}); they are self-reported and are not diagnostic."
    )


def _t_tool_unavailable(fields):
    tool = fields.get("source_tool") or "the tool"
    reason = fields.get("unavailable_reason") or "unavailable"
    return f"{tool} was unavailable ({reason}); the related question remains unanswered."


def _t_cadence(fields):
    return (
        f"{format_count(fields.get('session_count'))} valid session(s) were recorded in "
        f"the last {format_count(fields.get('window_days'))} day(s)."
    )


#: The closed set of templates. Every evidence statement comes from one of these.
CLAIM_TEMPLATES: dict[str, Callable[[dict], str]] = {
    "signal_change": _t_signal_change,
    "signal_change_absolute": _t_signal_change_absolute,
    "signal_unavailable": _t_signal_unavailable,
    "window_stat": _t_window_stat,
    "session_count": _t_session_count,
    "quality": _t_quality,
    "quality_invalid": _t_quality_invalid,
    "anomaly": _t_anomaly,
    "anomaly_unavailable": _t_anomaly_unavailable,
    "persistence": _t_persistence,
    "persistence_depth": _t_persistence_depth,
    "persistence_robustness": _t_persistence_robustness,
    "context_absence": _t_context_absence,
    "context_present": _t_context_present,
    "tool_unavailable": _t_tool_unavailable,
    "cadence": _t_cadence,
}


def render_claim(template, **fields):
    """Render one statement from a named template.

    Raises:
        KeyError: the template name is not in :data:`CLAIM_TEMPLATES`.
        ValueError: a field was passed that :class:`EvidenceItem` does not
            define (keeps items structurally complete).
    """
    try:
        renderer = CLAIM_TEMPLATES[template]
    except KeyError as exc:
        raise KeyError(f"unknown claim template: {template!r}") from exc

    unknown = sorted(set(fields) - _EVIDENCE_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown evidence field(s) for template {template!r}: {unknown}"
        )
    return renderer(fields)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class EvidenceRegistry:
    """Append-only collection of :class:`EvidenceItem` objects.

    Items receive sequential ids (``E1``, ``E2``, ...) in registration order, so
    the same inputs always produce the same ids and the same ``digest()``.
    """

    ID_PREFIX = "E"

    def __init__(self):
        self._items: list[EvidenceItem] = []
        self._by_id: dict[str, EvidenceItem] = {}

    # -- creation (template-only) -----------------------------------------

    def register(self, template, *, kind, source_tool, **fields):
        """Render a template and register the resulting item.

        ``id`` and ``statement`` are generated here and can never be supplied
        by a caller, so no caller can hand-write a fact.
        """
        for reserved in ("id", "statement"):
            if reserved in fields:
                raise ValueError(
                    f"{reserved!r} is generated by the registry and cannot be supplied"
                )
        statement = render_claim(template, source_tool=source_tool, **fields)
        item = EvidenceItem(
            id=self._next_id(),
            kind=kind,
            source_tool=source_tool,
            statement=statement,
            **fields,
        )
        self._items.append(item)
        self._by_id[item.id] = item
        return item

    def register_unavailable(self, template, *, kind, source_tool, reason, **fields):
        """Register an item that records *missing* evidence, with the reason."""
        return self.register(
            template,
            kind=kind,
            source_tool=source_tool,
            available=False,
            unavailable_reason=reason,
            **fields,
        )

    def _next_id(self):
        return f"{self.ID_PREFIX}{len(self._items) + 1}"

    # -- reading -----------------------------------------------------------

    def get(self, item_id) -> EvidenceItem | None:
        return self._by_id.get(item_id)

    def all(self) -> list[EvidenceItem]:
        return list(self._items)

    def ids(self) -> list[str]:
        return [item.id for item in self._items]

    def filter(self, *, kind=None, key=None) -> list[EvidenceItem]:
        """Items matching an optional ``kind`` and/or canonical ``key``."""
        return [
            item
            for item in self._items
            if (kind is None or item.kind == kind) and (key is None or item.key == key)
        ]

    def to_dicts(self) -> list[dict]:
        return [item.model_dump(mode="json") for item in self._items]

    def digest(self) -> str:
        """A stable fingerprint of the registry's factual content.

        Excludes nothing that was measured, and includes no timestamps, so two
        runs over the same inputs produce the same digest.
        """
        payload = [
            {
                "id": item.id,
                "kind": item.kind,
                "key": item.key,
                "statement": item.statement,
                "available": item.available,
                "value": item.value,
                "baseline_value": item.baseline_value,
                "relative_change": item.relative_change,
                "session_count": item.session_count,
                "window_days": item.window_days,
                "status": item.status,
            }
            for item in self._items
        ]
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


# ---------------------------------------------------------------------------
# Builders: Phase 2 objects -> evidence items (pure, no I/O)
# ---------------------------------------------------------------------------


def register_from_behavioral_evidence(
    registry: EvidenceRegistry,
    evidence: BehavioralEvidence,
    anomaly: MLEvidenceOutput | None = None,
):
    """Register the evidence items implied by a Phase 2 evidence package.

    This is a pure translation step: it takes already-fetched Phase 2 objects
    and never calls a tool or a repository itself (orchestration lives in the
    graph's ingest node). Missing or unavailable data is registered as explicit
    missing evidence rather than omitted, so the later critic can tell the
    difference between "measured and normal" and "not measured".

    Returns:
        The list of items registered, in registration order.
    """
    items = []
    temporal = evidence.temporal_analysis
    recent_window = temporal.recent.window_days
    baseline_window = temporal.baseline.window_days

    # --- signals ----------------------------------------------------------
    for signal in evidence.signals:
        common = {
            "key": signal.key,
            "label": signal.label,
            "unit": signal.canonical_unit,
        }
        if signal.available and signal.value is not None:
            if signal.relative_change is not None and signal.baseline_value is not None:
                items.append(
                    registry.register(
                        "signal_change",
                        kind="signal_change",
                        source_tool=SOURCE_ADAPTER,
                        direction=signal.direction,
                        value=signal.value,
                        baseline_value=signal.baseline_value,
                        absolute_change=signal.absolute_change,
                        relative_change=signal.relative_change,
                        window_days=baseline_window,
                        **common,
                    )
                )
            elif signal.absolute_change is not None:
                items.append(
                    registry.register(
                        "signal_change_absolute",
                        kind="signal_change_absolute",
                        source_tool=SOURCE_ADAPTER,
                        direction=signal.direction,
                        value=signal.value,
                        baseline_value=signal.baseline_value,
                        absolute_change=signal.absolute_change,
                        window_days=baseline_window,
                        **common,
                    )
                )
            else:
                items.append(
                    registry.register_unavailable(
                        "signal_unavailable",
                        kind="signal_unavailable",
                        source_tool=SOURCE_ADAPTER,
                        reason="no baseline reference was available for this session",
                        **common,
                    )
                )
        else:
            items.append(
                registry.register_unavailable(
                    "signal_unavailable",
                    kind="signal_unavailable",
                    source_tool=SOURCE_ADAPTER,
                    reason=(
                        "no current value or baseline reference was available for "
                        "this session"
                    ),
                    **common,
                )
            )

    # --- completeness of the history -------------------------------------
    quality = evidence.data_quality
    items.append(
        registry.register(
            "quality",
            kind="quality",
            source_tool=SOURCE_ADAPTER,
            value=quality.valid_sessions,
            session_count=quality.total_sessions,
        )
    )
    if quality.invalid_sessions:
        items.append(
            registry.register(
                "quality_invalid",
                kind="quality_invalid",
                source_tool=SOURCE_ADAPTER,
                value=quality.invalid_sessions,
                session_count=quality.total_sessions,
            )
        )
    items.append(
        registry.register(
            "session_count",
            kind="session_count",
            source_tool=SOURCE_ADAPTER,
            session_count=quality.valid_sessions,
            threshold=quality.model_minimum_sessions,
        )
    )
    items.append(
        registry.register(
            "cadence",
            kind="cadence",
            source_tool=SOURCE_ADAPTER,
            session_count=temporal.recent.valid_session_count,
            window_days=recent_window,
        )
    )

    # --- window statistics (recent and baseline) --------------------------
    for stats, window in (
        (temporal.recent_feature_stats, recent_window),
        (temporal.baseline_feature_stats, baseline_window),
    ):
        for stat in stats:
            if not stat.available or stat.mean is None:
                continue
            items.append(
                registry.register(
                    "window_stat",
                    kind="window_stat",
                    source_tool=SOURCE_ADAPTER,
                    key=stat.key,
                    label=stat.label,
                    unit=stat.canonical_unit,
                    value=stat.mean,
                    session_count=stat.sample_count,
                    window_days=window,
                )
            )

    # --- persistence ------------------------------------------------------
    persistence = temporal.persistence
    items.append(
        registry.register(
            "persistence",
            kind="persistence",
            source_tool=SOURCE_ADAPTER,
            status=persistence.status,
            value=float(persistence.signals_moved),
            session_count=persistence.supporting_session_count,
        )
    )

    # --- ML anomaly (numbers only, never interpreted as a condition) ------
    if anomaly is not None and anomaly.available:
        if anomaly.is_anomaly is None:
            flag = "unknown"
        else:
            flag = "true" if anomaly.is_anomaly else "false"
        items.append(
            registry.register(
                "anomaly",
                kind="anomaly",
                source_tool=SOURCE_ML,
                value=anomaly.anomaly_score,
                status=flag,
            )
        )
    else:
        if anomaly is None:
            reason = "the ML evidence tool was not called"
        else:
            reason = anomaly.unavailable_reason or anomaly.status or "unavailable"
        items.append(
            registry.register_unavailable(
                "anomaly_unavailable",
                kind="anomaly_unavailable",
                source_tool=SOURCE_ML,
                reason=reason,
            )
        )

    # --- context (honest about what does not exist) -----------------------
    context = evidence.context
    has_context = bool(
        context
        and context.source
        and context.source != "unavailable"
        and (context.checkins or context.symptoms)
    )
    if has_context:
        items.append(
            registry.register(
                "context_present",
                kind="context_present",
                source_tool=SOURCE_ADAPTER,
                status=context.source,
            )
        )
    else:
        items.append(
            registry.register(
                "context_absence",
                kind="context_absence",
                source_tool=SOURCE_ADAPTER,
            )
        )

    return items


# ---------------------------------------------------------------------------
# Read-through cache over the Phase 2 repository
# ---------------------------------------------------------------------------


class CachingRepository:
    """A memoizing, strictly read-only wrapper around a ``SessionRepository``.

    Why: each Phase 2 windowed tool independently calls ``list_sessions``, so a
    single evidence build currently costs several full reads of the same table.
    This wrapper collapses those into one read per investigation.

    Guarantees:

    - It exposes **only** ``list_sessions`` and ``get_anomaly_result``. There is
      deliberately no ``__getattr__`` pass-through, so write methods on the
      wrapped repository are unreachable through this object.
    - Returned rows are copies: a caller cannot mutate cached or upstream data.
    - Failures are **not** cached, so a later attempt can still succeed.
    - ``stats()`` reports the reads actually performed and the cache hits, which
      the scorecard uses as an efficiency metric.
    """

    def __init__(self, repository):
        self._repository = repository
        self._sessions: dict[str, list[dict]] = {}
        self._anomalies: dict[str, dict | None] = {}
        self._stats = {"list_sessions": 0, "get_anomaly_result": 0, "cache_hits": 0}

    def list_sessions(self, user_id):
        if user_id in self._sessions:
            self._stats["cache_hits"] += 1
            return [dict(row) for row in self._sessions[user_id]]

        rows = self._repository.list_sessions(user_id)
        self._stats["list_sessions"] += 1
        self._sessions[user_id] = [dict(row) for row in (rows or [])]
        return [dict(row) for row in self._sessions[user_id]]

    def get_anomaly_result(self, session_id):
        if session_id in self._anomalies:
            self._stats["cache_hits"] += 1
            row = self._anomalies[session_id]
            return dict(row) if row else None

        row = self._repository.get_anomaly_result(session_id)
        self._stats["get_anomaly_result"] += 1
        self._anomalies[session_id] = dict(row) if row else None
        return dict(row) if row else None

    def stats(self) -> dict:
        """Reads performed upstream, and cache hits served locally."""
        return dict(self._stats)

    def upstream(self):
        """The wrapped repository (for assertions and diagnostics)."""
        return self._repository
