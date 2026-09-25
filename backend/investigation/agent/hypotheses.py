"""Hypothesis catalog and evaluation (Phase 3.2).

Behavioral hypotheses, never medical ones
----------------------------------------
The four hypotheses are *behavioral investigation* hypotheses. They describe
what could account for a measured deviation: ordinary variation, circumstances
the person reported, a sustained change, or a data problem. None of them names,
implies or tests for a medical condition, and the catalog is fixed here so a
caller cannot introduce one: :func:`evaluate_hypotheses` rejects any supplied
hypothesis whose text is not one of the four catalog statements.

Evidence linking
----------------
``Hypothesis.supporting_evidence`` / ``contradicting_evidence`` hold **bare
evidence ids** (``"E7"``) that resolve through the
:class:`~investigation.agent.evidence.EvidenceRegistry`. No statement text is
invented here: a rule fires only when it has at least one real id to cite, so a
``"supported"`` hypothesis always cites measurable evidence.

A rule is never allowed to contradict Phase 2's verdict in the alarming
direction. ``H3`` ("Persistent behavioral change") can only be *supported* when
``PersistenceFinding.eligible_for_persistence_claim`` is true, which already
requires Phase 2 to have said ``"persistent_change"``, at least the model
minimum of valid sessions, a sustained per-session depth, and window robustness
that did not disagree.

Missing evidence
----------------
Missing evidence is tracked as canonical question **keys**. :data:`QUESTION_TEXT`
gives each key its user-facing sentence and :data:`QUESTION_TOOLS` says which
Phase 2 tools could answer it -- the empty tuple means nothing can answer it yet
(no contextual store and no capture metadata exist), which is exactly the
signal the investigation router needs to stop collecting.

No I/O: this module reads only the registry and already-fetched Phase 2 objects.
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..contracts import BehavioralEvidence
from ..state import Hypothesis, HypothesisStatus
from .evidence import EvidenceRegistry
from .persistence import PersistenceFinding
from .trace import InvestigationTrace

HypothesisId = Literal["H1", "H2", "H3", "H4"]

#: Tools allowed to answer an open question. Only Phase 2 tool names appear.
_KNOWN_TOOLS = (
    "get_ml_evidence",
    "get_recent_sessions",
    "get_historical_baseline",
    "calculate_behavioral_drift",
    "compare_time_windows",
)


class HypothesisSpec(BaseModel):
    """One fixed behavioral hypothesis."""

    id: HypothesisId
    statement: str = Field(min_length=1)
    scope: str = Field(min_length=1)


HYPOTHESIS_CATALOG: tuple[HypothesisSpec, ...] = (
    HypothesisSpec(
        id="H1",
        statement="Temporary behavioral variation",
        scope=(
            "The deviation is isolated or recent and may resolve without "
            "indicating a sustained change."
        ),
    ),
    HypothesisSpec(
        id="H2",
        statement="Contextual disruption (sleep, fatigue, stress or workload)",
        scope=(
            "The deviation coincides with user-reported circumstances rather "
            "than the behavior itself."
        ),
    ),
    HypothesisSpec(
        id="H3",
        statement="Persistent behavioral change",
        scope=(
            "The deviation is sustained across sessions and survives comparison "
            "against alternative windows."
        ),
    ),
    HypothesisSpec(
        id="H4",
        statement="Data-quality or capture artifact",
        scope=(
            "The deviation is attributable to missing, invalid or inconsistent "
            "behavioral data rather than to behavior."
        ),
    ),
)

#: Canonical missing-evidence keys, in fixed reporting order.
QUESTION_TEXT: dict[str, str] = {
    "recent_session_depth": (
        "how many of the most recent valid sessions deviate from the baseline"
    ),
    "additional_sessions": (
        "additional valid sessions to test whether the deviation is sustained"
    ),
    "window_robustness": (
        "whether alternative comparison windows agree with the default windows"
    ),
    "longer_horizon": "whether the change is visible over a longer history",
    "context_factors": (
        "user-reported contextual factors (sleep, fatigue, stress) for the "
        "affected period"
    ),
    "capture_change": (
        "whether the capture setup (keyboard, device, environment) changed"
    ),
}

#: Which Phase 2 tools could answer each question. Empty = cannot be answered yet.
QUESTION_TOOLS: dict[str, tuple[str, ...]] = {
    "recent_session_depth": ("get_recent_sessions",),
    "additional_sessions": ("get_recent_sessions",),
    "window_robustness": ("calculate_behavioral_drift", "compare_time_windows"),
    "longer_horizon": ("compare_time_windows",),
    "context_factors": (),
    "capture_change": (),
}

QUESTION_KEYS: tuple[str, ...] = tuple(QUESTION_TEXT)

#: Questions no available data source can answer in this phase.
UNANSWERABLE_QUESTIONS: tuple[str, ...] = tuple(
    key for key in QUESTION_KEYS if not QUESTION_TOOLS[key]
)


class HypothesisReasoning(BaseModel):
    """Why one hypothesis ended up with its status (machine-readable)."""

    id: HypothesisId
    status: HypothesisStatus
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class HypothesisEvaluation(BaseModel):
    """The evaluated hypotheses plus the questions still open."""

    hypotheses: list[Hypothesis]
    reasoning: list[HypothesisReasoning]
    missing_evidence: list[str] = Field(default_factory=list)


def question_text(key, default="an unregistered question"):
    """The user-facing sentence for a question key."""
    return QUESTION_TEXT.get(key, default)


def is_answerable(key):
    """True when at least one tool could answer this question."""
    return bool(QUESTION_TOOLS.get(key))


def answerable_questions(keys):
    """Registered, answerable keys present in ``keys``, in canonical order."""
    present = set(keys or ())
    return [key for key in QUESTION_KEYS if key in present and is_answerable(key)]


def unanswerable_questions(keys):
    """Registered, unanswerable keys present in ``keys``, in canonical order."""
    present = set(keys or ())
    return [key for key in QUESTION_KEYS if key in present and not is_answerable(key)]


def create_hypotheses():
    """Create the four catalog hypotheses, all ``uncertain``.

    The investigation always starts from the least alarming position: no
    hypothesis is pre-supported, and no hypothesis is omitted.
    """
    return [Hypothesis(hypothesis=spec.statement) for spec in HYPOTHESIS_CATALOG]


def _ids(registry, kind, key=None):
    """Evidence ids of a kind (optionally for one canonical feature key)."""
    return [item.id for item in registry.filter(kind=kind, key=key)]


def _merge(*groups):
    """Concatenate id groups, preserving first-seen order and dropping repeats."""
    merged = []
    for group in groups:
        for item_id in group:
            if item_id not in merged:
                merged.append(item_id)
    return merged


def _status(supporting, contradicting) -> HypothesisStatus:
    if supporting and not contradicting:
        return "supported"
    if contradicting:
        return "weakened"
    return "uncertain"


def _evaluate_h4(registry, evidence, finding, ids):
    """Data-quality / capture artifact."""
    quality = evidence.data_quality
    unavailable = _merge(ids["signal_unavailable"], ids["anomaly_unavailable"], ids["tool_unavailable"])

    supporting = _merge(
        [] if quality.meets_model_minimum else ids["session_count"],
        ids["quality_invalid"] if quality.invalid_sessions else [],
        ids["quality"] if quality.sessions_without_timestamp else [],
        unavailable,
        ids["persistence_robustness"] if finding.robustness == "disagrees" else [],
    )

    clean = (
        quality.meets_model_minimum
        and quality.invalid_sessions == 0
        and quality.sessions_without_timestamp == 0
        and not unavailable
        and finding.robustness == "agrees"
    )
    contradicting = (
        _merge(ids["session_count"], ids["quality"], ids["persistence_robustness"])
        if clean
        else []
    )

    missing = ["capture_change"] if unavailable else []
    return supporting, contradicting, missing


def evaluate_hypotheses(
    registry: EvidenceRegistry,
    evidence: BehavioralEvidence,
    finding: PersistenceFinding,
    hypotheses=None,
    *,
    trace: InvestigationTrace | None = None,
) -> HypothesisEvaluation:
    """Evaluate the four catalog hypotheses against the measured evidence.

    Args:
        registry: the evidence registry, whose ids are cited.
        evidence: the Phase 2 package (data quality and context).
        finding: the :class:`PersistenceFinding` measured for this investigation.
        hypotheses: an optional existing hypothesis list (e.g. from
            ``create_hypotheses``). Its statements must match the catalog
            exactly, so no foreign or alarming hypothesis can be introduced.
        trace: optional investigation trace.

    Returns:
        HypothesisEvaluation. Every cited id is a real registry id, and
        supporting and contradicting evidence never overlap for one hypothesis.
    """
    if hypotheses is not None:
        if len(hypotheses) != len(HYPOTHESIS_CATALOG):
            raise ValueError(
                f"expected {len(HYPOTHESIS_CATALOG)} hypotheses, got {len(hypotheses)}"
            )
        for hypothesis, spec in zip(hypotheses, HYPOTHESIS_CATALOG):
            if hypothesis.hypothesis != spec.statement:
                raise ValueError(
                    "hypothesis text must match the catalog statement "
                    f"{spec.statement!r}"
                )

    ids = {
        "persistence": _ids(registry, "persistence"),
        "persistence_depth": _ids(registry, "persistence_depth"),
        "persistence_robustness": _ids(registry, "persistence_robustness"),
        "session_count": _ids(registry, "session_count"),
        "quality": _ids(registry, "quality"),
        "quality_invalid": _ids(registry, "quality_invalid"),
        "signal_unavailable": _ids(registry, "signal_unavailable"),
        "anomaly_unavailable": _ids(registry, "anomaly_unavailable"),
        "tool_unavailable": _ids(registry, "tool_unavailable"),
        "context_present": _ids(registry, "context_present"),
    }

    eligible = finding.eligible_for_persistence_claim

    # H1 -- temporary behavioral variation.
    h1_support = (
        [] if eligible else _merge(ids["persistence"], ids["persistence_depth"])
    )
    h1_contradict = (
        _merge(ids["persistence"], ids["persistence_depth"], ids["persistence_robustness"])
        if eligible
        else []
    )
    h1_missing = ["recent_session_depth"] if finding.depth is None else []

    # H2 -- contextual disruption. Absent context is never refutation, so this
    # hypothesis can only be supported by real user-reported context rows.
    h2_support = list(ids["context_present"])
    h2_contradict = []
    h2_missing = ["context_factors"]

    # H3 -- persistent behavioral change.
    h3_support = (
        _merge(
            ids["persistence"],
            ids["persistence_depth"],
            ids["persistence_robustness"],
            ids["session_count"],
        )
        if eligible
        else []
    )
    if eligible:
        h3_contradict = []
    elif not evidence.data_quality.meets_model_minimum:
        h3_contradict = _merge(ids["session_count"])
    elif finding.phase2_status != "persistent_change":
        h3_contradict = _merge(ids["persistence"], ids["persistence_depth"])
    elif finding.depth is None or finding.depth < finding.minimum_recent_sessions:
        h3_contradict = _merge(ids["persistence_depth"])
    elif finding.robustness == "disagrees":
        h3_contradict = _merge(ids["persistence_robustness"])
    else:
        h3_contradict = []
    h3_missing = []
    if finding.depth is None:
        h3_missing.append("additional_sessions")
    if finding.robustness == "not_assessed":
        h3_missing.append("window_robustness")

    h4_support, h4_contradict, h4_missing = _evaluate_h4(
        registry, evidence, finding, ids
    )

    rules = (
        ("H1", h1_support, h1_contradict, h1_missing),
        ("H2", h2_support, h2_contradict, h2_missing),
        ("H3", h3_support, h3_contradict, h3_missing),
        ("H4", h4_support, h4_contradict, h4_missing),
    )

    evaluated = []
    reasoning = []
    open_questions = []
    for spec, (hypothesis_id, supporting, contradicting, missing) in zip(
        HYPOTHESIS_CATALOG, rules
    ):
        status = _status(supporting, contradicting)
        evaluated.append(
            Hypothesis(
                hypothesis=spec.statement,
                supporting_evidence=list(supporting),
                contradicting_evidence=list(contradicting),
                missing_evidence=list(missing),
                status=status,
            )
        )
        reasoning.append(
            HypothesisReasoning(
                id=hypothesis_id,
                status=status,
                supporting_evidence=list(supporting),
                contradicting_evidence=list(contradicting),
                missing_evidence=list(missing),
            )
        )
        for key in missing:
            if key not in open_questions:
                open_questions.append(key)

        if trace is not None:
            trace.record_hypothesis_update(
                "evaluate_hypotheses",
                evidence_ids=_merge(supporting, contradicting),
                detail=f"{hypothesis_id}:{status}",
            )

    # Report open questions in canonical question order, not discovery order.
    missing_evidence = [key for key in QUESTION_KEYS if key in set(open_questions)]

    return HypothesisEvaluation(
        hypotheses=evaluated,
        reasoning=reasoning,
        missing_evidence=missing_evidence,
    )
