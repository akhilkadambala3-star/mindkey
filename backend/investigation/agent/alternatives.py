"""Alternative explanation scan (Phase 3.2).

Purpose
-------
Before the investigation concludes anything about a sustained change, it must
actively look for the non-alarming explanations. This module walks a fixed
candidate list and reports, for each candidate, whether the *measured* evidence
speaks for it, against it, or cannot speak to it at all.

Three rules make this honest rather than decorative:

1. **Nothing is invented.** A candidate is only ``supported`` or ``weakened``
   when the evidence registry already holds the facts that decide it.
2. **Absence is not refutation.** Where no data source exists (no contextual
   store, no capture metadata), the candidate is ``unavailable`` -- never
   ``weakened``.
3. **Every verdict is citable.** ``supported`` / ``weakened`` findings always
   carry at least one evidence id; ``partially_evaluated`` marks a proxy that
   decides the candidate only weakly (currently the session-cadence proxy for
   workload/schedule).

No I/O: only the registry and already-fetched Phase 2 objects are read.
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..contracts import BehavioralEvidence
from .evidence import EvidenceRegistry
from .persistence import PersistenceFinding
from .trace import InvestigationTrace

#: Recorded as the ``source_tool`` of the evidence this module registers.
SOURCE_ALTERNATIVES = "assess_alternatives"

AlternativeStatus = Literal["supported", "weakened", "partially_evaluated", "unavailable"]

#: Fixed candidate order. Labels are presentation text, not clinical language.
ALTERNATIVE_CATALOG: tuple[tuple[str, str], ...] = (
    ("temporary_variation", "Temporary variation"),
    ("data_quality_artifact", "Data-quality artifact"),
    ("insufficient_data", "Insufficient data"),
    ("technical_failure", "Technical or tool failure"),
    ("unusual_workload_or_schedule", "Unusual workload or schedule (sampling cadence)"),
    ("keyboard_or_environment_change", "Keyboard or environment change"),
    ("poor_sleep", "Poor sleep"),
    ("fatigue", "Fatigue"),
    ("stress", "Stress"),
    ("illness_or_mood", "Illness or mood change"),
    ("distraction", "Distraction"),
)

#: candidate -> label, and the canonical reporting order.
LABELS: dict[str, str] = dict(ALTERNATIVE_CATALOG)
_CANDIDATE_ORDER: dict[str, int] = {
    candidate: index for index, (candidate, _) in enumerate(ALTERNATIVE_CATALOG)
}

#: The contextual candidates: user-reported circumstances for which no
#: server-side store exists, so they are always reported as unavailable.
CONTEXT_CANDIDATES = ("poor_sleep", "fatigue", "stress", "illness_or_mood", "distraction")


class AlternativeFinding(BaseModel):
    """One candidate explanation and what the evidence says about it."""

    candidate: str = Field(min_length=1)
    label: str = Field(min_length=1)
    status: AlternativeStatus
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


def _ids(registry, kind):
    return [item.id for item in registry.filter(kind=kind)]


def _finding(candidate, status, reason, evidence_ids=()):
    return AlternativeFinding(
        candidate=candidate,
        label=LABELS[candidate],
        status=status,
        reason=reason,
        evidence_ids=list(evidence_ids),
    )


def _rate(count, window_days):
    if count is None or not window_days:
        return None
    return count / window_days


def _assess_workload(registry, evidence):
    """Session-cadence proxy for workload/schedule disruption.

    Registers the baseline-window cadence item (Phase 3.1 registers the recent
    one) so the comparison is citable from both windows. This is a proxy: it can
    show that the recent window holds fewer observations than the baseline
    window, not why.
    """
    recent = evidence.temporal_analysis.recent
    baseline = evidence.temporal_analysis.baseline

    cadence_item = registry.register(
        "cadence",
        kind="cadence",
        source_tool=SOURCE_ALTERNATIVES,
        session_count=baseline.valid_session_count,
        window_days=baseline.window_days,
    )
    earlier_ids = [
        item.id for item in registry.filter(kind="cadence") if item.id != cadence_item.id
    ]
    evidence_ids = earlier_ids + [cadence_item.id]

    recent_rate = _rate(recent.valid_session_count, recent.window_days)
    baseline_rate = _rate(baseline.valid_session_count, baseline.window_days)

    if not baseline_rate:
        return _finding(
            "unusual_workload_or_schedule",
            "unavailable",
            "sampling_cadence_not_comparable",
            evidence_ids,
        )

    if recent_rate is not None and recent_rate < 0.5 * baseline_rate:
        return _finding(
            "unusual_workload_or_schedule",
            "partially_evaluated",
            "recent_sampling_sparser_than_baseline",
            evidence_ids,
        )

    return _finding(
        "unusual_workload_or_schedule",
        "weakened",
        "recent_sampling_comparable_to_baseline",
        evidence_ids,
    )


def assess_alternatives(
    registry: EvidenceRegistry,
    evidence: BehavioralEvidence,
    finding: PersistenceFinding,
    *,
    trace: InvestigationTrace | None = None,
) -> list[AlternativeFinding]:
    """Scan every catalog candidate against the measured evidence.

    Returns:
        Findings in :data:`ALTERNATIVE_CATALOG` order.
    """
    if not isinstance(registry, EvidenceRegistry):
        raise TypeError("registry must be an EvidenceRegistry")

    quality = evidence.data_quality
    persistence_ids = _ids(registry, "persistence")
    depth_ids = _ids(registry, "persistence_depth")
    quality_ids = _ids(registry, "quality")
    invalid_ids = _ids(registry, "quality_invalid")
    session_ids = _ids(registry, "session_count")
    unavailable_ids = (
        _ids(registry, "anomaly_unavailable")
        + _ids(registry, "tool_unavailable")
        + _ids(registry, "signal_unavailable")
    )
    context_ids = _ids(registry, "context_absence") + _ids(registry, "context_present")

    findings = []

    # -- temporary variation ----------------------------------------------
    if finding.eligible_for_persistence_claim:
        findings.append(
            _finding(
                "temporary_variation",
                "weakened",
                "eligible_for_persistence_claim",
                persistence_ids + depth_ids,
            )
        )
    else:
        findings.append(
            _finding(
                "temporary_variation",
                "supported",
                finding.downgrade_reason or "not_eligible_for_persistence_claim",
                persistence_ids + depth_ids,
            )
        )

    # -- data-quality artifact ---------------------------------------------
    if quality.invalid_sessions:
        findings.append(
            _finding(
                "data_quality_artifact",
                "supported",
                "invalid_sessions_present",
                invalid_ids or quality_ids,
            )
        )
    else:
        findings.append(
            _finding("data_quality_artifact", "weakened", "no_invalid_sessions", quality_ids)
        )

    # -- insufficient data -------------------------------------------------
    if not quality.meets_model_minimum:
        findings.append(
            _finding("insufficient_data", "supported", "below_model_minimum", session_ids)
        )
    else:
        findings.append(
            _finding("insufficient_data", "weakened", "meets_model_minimum", session_ids)
        )

    # -- technical failure -------------------------------------------------
    # When every required source returned data, that is itself citable: the
    # items those sources produced are the evidence against a tool failure.
    available_source_ids = (
        _ids(registry, "anomaly")
        + _ids(registry, "quality")
        + _ids(registry, "session_count")
    )
    if unavailable_ids:
        findings.append(
            _finding(
                "technical_failure",
                "supported",
                "evidence_source_unavailable",
                unavailable_ids,
            )
        )
    else:
        findings.append(
            _finding(
                "technical_failure",
                "weakened",
                "all_required_sources_available",
                available_source_ids,
            )
        )

    # -- workload / schedule (cadence proxy) -------------------------------
    findings.append(_assess_workload(registry, evidence))

    # -- keyboard / environment --------------------------------------------
    findings.append(
        _finding(
            "keyboard_or_environment_change",
            "unavailable",
            "no_capture_metadata_stored",
            invalid_ids,
        )
    )

    # -- contextual candidates (no store exists) ---------------------------
    for candidate in CONTEXT_CANDIDATES:
        findings.append(
            _finding(candidate, "unavailable", "no_contextual_store", context_ids)
        )

    findings.sort(key=lambda item: _CANDIDATE_ORDER[item.candidate])

    if trace is not None:
        for item in findings:
            trace.record_state_change(
                "assess_alternatives", detail=f"{item.candidate}:{item.status}"
            )

    return findings
