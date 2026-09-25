"""Grounded assessment / report assembly (Phase 3.4).

Turns an ``InvestigationResult`` plus its :class:`~investigation.agent.critic.Critique`
into a structured, JSON-serializable ``GroundedReport``.

Grounding rules
---------------
The only sources of the report's factual text are:

1. ``EvidenceItem.statement`` -- rendered by Phase 3.1's fixed claim templates,
   never authored here;
2. the fixed ``HYPOTHESIS_CATALOG`` / ``ALTERNATIVE_CATALOG`` statements;
3. fixed report templates (the conclusion, summary and limitation lines).

There is deliberately no API that accepts free-form text, so the report cannot
invent a claim. Every claim carries the evidence ids it rests on and the trace
sequence numbers of the events that recorded that evidence, and any claim the
critic rejected is moved to ``rejected_claims`` rather than asserted.

This module performs no I/O: it has no repository, API or database access, uses
no language model, and adds no dependencies beyond Pydantic and the standard
library.
"""

from typing import Literal

from pydantic import BaseModel, Field

from .critic import RejectedClaim, critique, evidence_index, trace_digest
from .engine import InvestigationResult
from .hypotheses import HYPOTHESIS_CATALOG
from .trace import InvestigationTrace

#: Version of the report shape.
REPORT_SCHEMA_VERSION = "1.0"

#: Fixed, non-diagnostic disclaimer carried by every report.
REPORT_DISCLAIMER = (
    "This report describes measured behavioral signals only; it establishes no "
    "cause and makes no medical claim."
)

#: Fixed conclusion statements, keyed by status.
CONCLUSION_TEXT = {
    "grounded": (
        "The available evidence supports a sustained behavioral change relative "
        "to the personal baseline; no cause is established."
    ),
    "preliminary": (
        "The available evidence shows a behavioral deviation, but it does not "
        "support a sustained-change conclusion."
    ),
    "no_deviation": (
        "Recent interaction patterns are consistent with the personal baseline."
    ),
    "inconclusive": (
        "The available evidence is insufficient to reach a behavioral "
        "conclusion about the triggering session."
    ),
}

#: Fixed report-authored limitations.
REPORT_LIMITATIONS = (
    "This report is produced by a deterministic, rule-based critic; it uses no language model.",
    "No mechanism, condition or cause is identified.",
)

#: Stop reasons that make the investigation inconclusive regardless of content.
INSUFFICIENT_STOP_REASONS = frozenset({"evidence_insufficient", "data_unavailable"})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Conclusion(BaseModel):
    """The report's single, fixed-template conclusion."""

    status: Literal["grounded", "preliminary", "no_deviation", "inconclusive"]
    basis: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class ReportClaim(BaseModel):
    """One grounded statement in the report, linked to evidence and the trace."""

    id: str = Field(min_length=1)
    kind: Literal["observation", "hypothesis", "alternative"]
    statement: str = Field(min_length=1)
    status: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    trace_seqs: list[int] = Field(default_factory=list)


class UncertaintySummary(BaseModel):
    """The investigation's residual uncertainty, as fixed reason codes."""

    level: Literal["low", "moderate", "high"]
    reasons: list[str] = Field(default_factory=list)


class ReportLink(BaseModel):
    """How the report ties back to the evidence and the investigation trace."""

    evidence_digest: str
    engine_trace_digest: str
    critic_trace_digest: str
    trace_event_count: int
    stop_event_seq: int | None = None


class GroundedReport(BaseModel):
    """The complete, grounded, non-diagnostic assessment."""

    schema_version: str = REPORT_SCHEMA_VERSION
    user_id: str
    session_id: str
    conclusion: Conclusion
    summary: list[str] = Field(default_factory=list)
    observations: list[ReportClaim] = Field(default_factory=list)
    hypothesis_assessment: list[ReportClaim] = Field(default_factory=list)
    alternatives: list[ReportClaim] = Field(default_factory=list)
    uncertainty: UncertaintySummary
    limitations: list[str] = Field(default_factory=list)
    next_action: str = ""
    rejected_claims: list[RejectedClaim] = Field(default_factory=list)
    engine_assessment: dict = Field(default_factory=dict)
    link: ReportLink
    critic_events: list[dict] = Field(default_factory=list)
    disclaimer: str = REPORT_DISCLAIMER


# ---------------------------------------------------------------------------
# Trace/evidence link helpers
# ---------------------------------------------------------------------------


def _seqs_for(events, evidence_ids):
    wanted = set(evidence_ids or ())
    if not wanted:
        return []
    seqs = []
    for event in events:
        if wanted & set(event.get("evidence_ids") or ()):
            seq = event.get("seq")
            if seq is not None:
                seqs.append(seq)
    return sorted(seqs)


def _stop_seq(events):
    for event in events:
        if event.get("event_type") == "stop":
            return event.get("seq")
    return None


def _ids_of_kind(index, *kinds):
    return [item.id for item in index.values() if item.kind in kinds]


def _suffix(ids):
    return f" [{', '.join(ids)}]" if ids else ""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _observations(result, index, rejected):
    claims = []
    for position, item in enumerate(index.values(), start=1):
        if item.id in rejected:
            continue
        claims.append(
            ReportClaim(
                id=f"O{position}",
                kind="observation",
                statement=item.statement,
                status=item.status,
                evidence_ids=[item.id],
                trace_seqs=_seqs_for(result.state.trace, [item.id]),
            )
        )
    return claims


def _hypothesis_claims(result, rejected):
    claims = []
    for hypothesis, spec in zip(result.hypotheses or [], HYPOTHESIS_CATALOG):
        if spec.id in rejected:
            continue
        claims.append(
            ReportClaim(
                id=spec.id,
                kind="hypothesis",
                statement=spec.statement,
                status=hypothesis.status,
                evidence_ids=list(hypothesis.supporting_evidence)
                + list(hypothesis.contradicting_evidence),
                trace_seqs=_seqs_for(
                    result.state.trace,
                    list(hypothesis.supporting_evidence)
                    + list(hypothesis.contradicting_evidence),
                ),
            )
        )
    return claims


def _alternative_claims(result, rejected):
    claims = []
    for alternative in result.alternatives or []:
        if alternative.candidate in rejected:
            continue
        claims.append(
            ReportClaim(
                id=alternative.candidate,
                kind="alternative",
                statement=alternative.label,
                status=alternative.status,
                evidence_ids=list(alternative.evidence_ids),
                trace_seqs=_seqs_for(result.state.trace, alternative.evidence_ids),
            )
        )
    return claims


def _uncertainty(result, verdict):
    assessment = result.final_assessment or {}
    quality = assessment.get("data_quality")
    persistence = assessment.get("persistence") or {}
    robustness = persistence.get("robustness")

    reasons = []
    high = moderate = False

    def add(reason):
        if reason not in reasons:
            reasons.append(reason)

    if result.stop_reason == "data_unavailable":
        add("evidence_unavailable")
        high = True

    if (
        isinstance(quality, dict)
        and not quality.get("meets_model_minimum")
    ) or result.stop_reason == "evidence_insufficient":
        add("insufficient_history")
        high = True

    if not persistence.get("eligible"):
        add("persistence_not_established")
        moderate = True

    if robustness == "disagrees":
        add("window_robustness_disagrees")
        moderate = True
    elif robustness == "not_assessed":
        add("window_robustness_not_assessed")
        moderate = True

    for entry in assessment.get("open_questions") or []:
        if isinstance(entry, dict) and not entry.get("answerable", False):
            add(f"unanswered:{entry.get('key')}")
            moderate = True

    if not verdict.is_sound:
        add("critic_rejections_present")
        moderate = True

    add("cause_not_established")

    if high:
        level = "high"
    elif moderate:
        level = "moderate"
    else:
        level = "low"
    return UncertaintySummary(level=level, reasons=reasons)


def _limitations(result, verdict):
    limitations = list(result.limitations or [])
    for rejection in verdict.rejections:
        limitations.append(
            f"A claim about {rejection.subject!r} was rejected by the critic "
            f"({rejection.reason}); it is not presented as a finding."
        )
    limitations.extend(REPORT_LIMITATIONS)
    return limitations


def _conclusion(result, verdict, index):
    assessment = result.final_assessment or {}
    quality = assessment.get("data_quality")
    persistence = assessment.get("persistence") or {}
    robustness = persistence.get("robustness")

    persistence_ids = _ids_of_kind(index, "persistence")

    if result.stop_reason == "data_unavailable":
        status, basis = "inconclusive", "data_unavailable"
    elif (isinstance(quality, dict) and not quality.get("meets_model_minimum")) or (
        result.stop_reason == "evidence_insufficient"
    ):
        status, basis = "inconclusive", "insufficient_history"
    elif not verdict.is_sound:
        status, basis = "inconclusive", "critic_rejections_present"
    elif persistence.get("status") == "stable":
        status, basis = "no_deviation", "stable_baseline"
    elif (
        persistence.get("eligible")
        and robustness == "agrees"
        and result.stop_reason == "evidence_sufficient"
    ):
        status, basis = "grounded", "grounded_persistent_change"
    else:
        status, basis = "preliminary", "deviation_not_persistent"

    return Conclusion(
        status=status,
        basis=basis,
        statement=CONCLUSION_TEXT[status],
        evidence_ids=persistence_ids,
    )


def _summary(result, verdict, index):
    assessment = result.final_assessment or {}
    quality = assessment.get("data_quality") or {}
    persistence = assessment.get("persistence") or {}

    quality_ids = _ids_of_kind(index, "quality", "session_count")
    persistence_ids = _ids_of_kind(index, "persistence", "persistence_robustness")

    supported = [
        spec.statement
        for spec, hypothesis in zip(HYPOTHESIS_CATALOG, result.hypotheses or [])
        if hypothesis.status == "supported"
    ]

    lines = []
    if quality:
        lines.append(
            f"Data quality: {quality.get('valid_sessions')} of "
            f"{quality.get('total_sessions')} stored session(s) are valid "
            f"behavioral evidence.{_suffix(quality_ids)}"
        )
    lines.append(
        f"Deterministic persistence assessment: "
        f"'{persistence.get('status')}'; {persistence.get('signals_moved')} "
        f"moved signal(s), robustness '{persistence.get('robustness')}'."
        f"{_suffix(persistence_ids)}"
    )
    lines.append(
        "Supported behavioral hypotheses: "
        + (", ".join(supported) if supported else "none")
        + "."
    )
    lines.append(f"Claims rejected by the critic: {len(verdict.rejections)}.")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_grounded_report(
    result: InvestigationResult,
    *,
    trace: InvestigationTrace | None = None,
    clock=None,
) -> GroundedReport:
    """Build a grounded report from one investigation result.

    Args:
        result: the engine's structured outcome.
        trace: optional trace to record critic events into. When omitted, a
            trace is created and **seeded with the engine's events** so the
            critic continues the same sequence.
        clock: optional zero-argument clock for the critic's trace events.

    Returns:
        GroundedReport. The result is never mutated.
    """
    engine_events = list(result.state.trace or [])

    if trace is None:
        trace = InvestigationTrace.from_events(engine_events, clock=clock)

    max_engine_seq = max(
        (event.get("seq") or 0 for event in engine_events), default=0
    )
    verdict = critique(result, trace=trace)
    all_events = trace.to_dicts()
    critic_events = [
        event for event in all_events if (event.get("seq") or 0) > max_engine_seq
    ]

    index = evidence_index(result.state.evidence)
    rejected = verdict.rejected_subjects

    link = ReportLink(
        evidence_digest=result.evidence_digest,
        engine_trace_digest=trace_digest(engine_events),
        critic_trace_digest=trace_digest(critic_events),
        trace_event_count=len(engine_events) + len(critic_events),
        stop_event_seq=_stop_seq(engine_events),
    )

    uncertainty = _uncertainty(result, verdict)

    return GroundedReport(
        user_id=result.user_id,
        session_id=result.session_id,
        conclusion=_conclusion(result, verdict, index),
        summary=_summary(result, verdict, index),
        observations=_observations(result, index, rejected),
        hypothesis_assessment=_hypothesis_claims(result, rejected),
        alternatives=_alternative_claims(result, rejected),
        uncertainty=uncertainty,
        limitations=_limitations(result, verdict),
        next_action=result.state.next_action or "",
        rejected_claims=verdict.rejections,
        engine_assessment=result.final_assessment or {},
        link=link,
        critic_events=critic_events,
        disclaimer=REPORT_DISCLAIMER,
    )
