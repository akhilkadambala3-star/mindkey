"""Deterministic critic (Phase 3.4).

The critic is the investigation's skeptic. It does not reason about behavior:
it re-checks that every claim the pipeline produced is **grounded in the
evidence it cites**, that the structures do not contradict each other, and that
nothing in the output uses clinical language. Claims that fail are *rejected* --
removed from the accepted set, recorded with a fixed reason code, and emitted as
``critic_rejection`` trace events.

What it verifies (fixed order, fixed reason codes)
--------------------------------------------------
1. every cited evidence id resolves in the registry
2. supported claims carry supporting evidence; weakened claims carry
   contradicting evidence; uncertain claims carry neither
3. supporting and contradicting evidence never overlap
4. a persistent-change claim (H3) never cites *unavailable* evidence as
   support -- while conservative (H1, ``temporary_variation``) and data-quality
   (H4, ``technical_failure``) claims legitimately may: Phase 3.2 supports H1
   with an *uncomputable* persistence depth, which is a data-limitation fact,
   not a measured change
5. a persistent-change claim (H3) is never supported without Phase 3.2
   eligibility
6. H1 and H3 are never both supported
7. hypotheses and alternatives come from the fixed catalogs
8. decided alternatives always cite evidence
9. the engine's own ``final_assessment`` agrees with the result it summarises
10. the two stored evidence digests agree
11. no asserted text contains a denylisted clinical term

Privacy: the critic reads only structured result fields (ids, counts, statuses,
labels, system-authored statements). It performs no I/O, calls no tool or
repository, and has no free-text input or output API.
"""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .alternatives import ALTERNATIVE_CATALOG
from .engine import InvestigationResult
from .evidence import EvidenceItem
from .hypotheses import HYPOTHESIS_CATALOG
from .trace import InvestigationTrace, TraceEvent

#: Version of the critique shape.
CRITIC_SCHEMA_VERSION = "1.0"

#: The closed set of rejection / contradiction reasons. Pydantic rejects others.
CriticReason = Literal[
    "evidence_id_unresolved",
    "support_without_evidence",
    "support_contradiction_overlap",
    "unavailable_evidence_asserted",
    "persistent_change_without_eligibility",
    "mutually_exclusive_hypotheses",
    "foreign_hypothesis",
    "foreign_alternative",
    "alternative_without_evidence",
    "alternative_status_inconsistent",
    "hypothesis_alternative_conflict",
    "assessment_mismatch",
    "evidence_digest_mismatch",
    "clinical_language",
]

#: Vocabulary that must never appear in asserted claim text.
CLINICAL_TERMS = (
    "parkinson",
    "dementia",
    "alzheimer",
    "depress",
    "bipolar",
    "diagnos",
    "disease",
    "disorder",
    "decline",
    "impair",
    "patient",
    "clinical",
)

#: Fixed Phase 2 safety *negations*. They are the only clinical-looking text the
#: pipeline is allowed to produce, so they are stripped before scanning -- the
#: same convention the existing hypothesis tests use.
SAFETY_EXEMPTIONS = (
    "an anomaly score is not a diagnosis.",
    "it is not a medical diagnosis and does not establish a cause.",
    "it is not a clinical finding.",
)

#: Hypotheses that assert a *sustained/persistent change*. Only these may never
#: rest on unavailable evidence. H1 (temporary variation) and the data-quality
#: hypotheses legitimately cite unavailable evidence -- e.g. Phase 3.2 supports
#: H1 with an uncomputable persistence depth, which is a data-limitation fact
#: rather than a measured change.
PERSISTENCE_CLAIM_HYPOTHESES = frozenset({"H3"})

#: Alternative statuses that are only meaningful when backed by evidence.
DECIDED_ALTERNATIVE_STATUSES = frozenset(
    {"supported", "weakened", "partially_evaluated"}
)

_ALT_STATUSES = frozenset(
    {"supported", "weakened", "partially_evaluated", "unavailable"}
)

_ALT_LABELS = dict(ALTERNATIVE_CATALOG)

_CATALOG_STATEMENTS = tuple(spec.statement for spec in HYPOTHESIS_CATALOG)
_CATALOG_IDS = tuple(spec.id for spec in HYPOTHESIS_CATALOG)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Contradiction(BaseModel):
    """Two parts of the investigation output that cannot both hold."""

    code: CriticReason
    subject: str = Field(min_length=1)
    detail: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class RejectedClaim(BaseModel):
    """A claim the critic refused to accept, with its fixed reason."""

    reason: CriticReason
    subject: str = Field(min_length=1)
    statement: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """The critic's verdict on one investigation."""

    schema_version: str = CRITIC_SCHEMA_VERSION
    is_sound: bool
    rejections: list[RejectedClaim] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    accepted_evidence_ids: list[str] = Field(default_factory=list)
    accepted_hypothesis_ids: list[str] = Field(default_factory=list)
    accepted_alternative_candidates: list[str] = Field(default_factory=list)
    denied_claim_count: int = 0

    @property
    def rejected_subjects(self) -> set:
        return {rejection.subject for rejection in self.rejections}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def trace_digest(events) -> str:
    """A timestamp-independent digest of a trace.

    Timestamps are excluded so the digest is stable across runs that use the
    wall clock; everything that describes *what happened* is included.
    """
    payload = []
    for event in events or ():
        if isinstance(event, TraceEvent):
            data = event.model_dump(mode="json")
        else:
            data = dict(event)
        payload.append(
            {
                "seq": data.get("seq"),
                "node": data.get("node"),
                "event_type": data.get("event_type"),
                "tool": data.get("tool"),
                "args_fingerprint": data.get("args_fingerprint"),
                "evidence_ids": list(data.get("evidence_ids") or []),
                "detail": data.get("detail"),
            }
        )
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def evidence_index(evidence_dicts) -> dict:
    """Reconstruct ``{id: EvidenceItem}`` from ``AgentState.evidence``.

    Records that do not validate are skipped rather than trusted; any claim
    citing one then fails the ``evidence_id_unresolved`` check.
    """
    index = {}
    for payload in evidence_dicts or ():
        try:
            item = EvidenceItem(**payload)
        except (ValidationError, TypeError):
            continue
        index[item.id] = item
    return index


def _strip_exemptions(text) -> str:
    cleaned = str(text or "").lower()
    for phrase in SAFETY_EXEMPTIONS:
        cleaned = cleaned.replace(phrase, "")
    return cleaned


def _persistence(result) -> dict:
    assessment = result.final_assessment or {}
    persistence = assessment.get("persistence")
    return persistence if isinstance(persistence, dict) else {}


def _meets_minimum(result):
    assessment = result.final_assessment or {}
    quality = assessment.get("data_quality")
    if not isinstance(quality, dict):
        return None
    return quality.get("meets_model_minimum")


# ---------------------------------------------------------------------------
# The critique
# ---------------------------------------------------------------------------


def critique(result: InvestigationResult, *, trace: InvestigationTrace | None = None) -> Critique:
    """Validate one ``InvestigationResult`` against its own evidence.

    Args:
        result: the engine's structured outcome.
        trace: optional trace to record ``critic_rejection`` events into. It
            should have been seeded with the engine's events (see
            ``InvestigationTrace.from_events``) so the sequence continues.

    Returns:
        Critique. The result is never mutated.
    """
    index = evidence_index(result.state.evidence)
    rejections: list[RejectedClaim] = []
    contradictions: list[Contradiction] = []

    def reject(reason, subject, statement="", ids=()):
        rejections.append(
            RejectedClaim(
                reason=reason,
                subject=str(subject),
                statement=str(statement or ""),
                evidence_ids=[str(i) for i in ids],
            )
        )

    def contradict(code, subject, detail="", ids=()):
        contradictions.append(
            Contradiction(
                code=code,
                subject=str(subject),
                detail=str(detail or ""),
                evidence_ids=[str(i) for i in ids],
            )
        )

    # -- hypotheses --------------------------------------------------------
    hypotheses = list(result.hypotheses or [])
    if len(hypotheses) != len(_CATALOG_STATEMENTS):
        reject(
            "foreign_hypothesis",
            "hypotheses",
            f"expected {len(_CATALOG_STATEMENTS)} hypotheses, found {len(hypotheses)}",
        )

    hyp_pairs = []
    for hypothesis, spec in zip(hypotheses, HYPOTHESIS_CATALOG):
        ids = list(hypothesis.supporting_evidence) + list(hypothesis.contradicting_evidence)
        if hypothesis.hypothesis != spec.statement:
            reject("foreign_hypothesis", spec.id, hypothesis.hypothesis, ids)
            continue
        hyp_pairs.append((spec.id, hypothesis))

    hyp_status = {}
    subject_ids = {}
    for hid, hypothesis in hyp_pairs:
        supporting = list(hypothesis.supporting_evidence)
        contradicting = list(hypothesis.contradicting_evidence)
        all_ids = supporting + contradicting
        subject_ids[hid] = all_ids

        unresolved = [item_id for item_id in all_ids if item_id not in index]
        if unresolved:
            reject("evidence_id_unresolved", hid, hypothesis.hypothesis, unresolved)

        if hypothesis.status == "supported" and not supporting:
            reject("support_without_evidence", hid, hypothesis.hypothesis, all_ids)
        if hypothesis.status == "weakened" and not contradicting:
            reject("support_without_evidence", hid, hypothesis.hypothesis, all_ids)
        if hypothesis.status == "uncertain" and (supporting or contradicting):
            reject("support_without_evidence", hid, hypothesis.hypothesis, all_ids)

        overlap = sorted(set(supporting) & set(contradicting))
        if overlap:
            reject("support_contradiction_overlap", hid, hypothesis.hypothesis, overlap)

        if hid in PERSISTENCE_CLAIM_HYPOTHESES:
            unavailable = [
                item_id
                for item_id in supporting
                if item_id in index and not index[item_id].available
            ]
            if unavailable:
                reject(
                    "unavailable_evidence_asserted",
                    hid,
                    hypothesis.hypothesis,
                    unavailable,
                )

        hyp_status[hid] = hypothesis.status

    if hyp_status.get("H3") == "supported":
        persistence = _persistence(result)
        if not persistence.get("eligible"):
            h3 = next((h for hid, h in hyp_pairs if hid == "H3"), None)
            reject(
                "persistent_change_without_eligibility",
                "H3",
                "" if h3 is None else h3.hypothesis,
                [] if h3 is None else h3.supporting_evidence,
            )

    if hyp_status.get("H1") == "supported" and hyp_status.get("H3") == "supported":
        contradict(
            "mutually_exclusive_hypotheses",
            "H1/H3",
            "temporary variation and persistent change are both supported",
        )

    # -- alternatives ------------------------------------------------------
    alternatives = list(result.alternatives or [])
    alt_status = {}
    for alternative in alternatives:
        candidate = alternative.candidate
        if candidate not in _ALT_LABELS:
            reject("foreign_alternative", candidate, alternative.label, alternative.evidence_ids)
            continue

        ids = list(alternative.evidence_ids)
        subject_ids[candidate] = ids

        if alternative.status not in _ALT_STATUSES:
            reject(
                "alternative_status_inconsistent",
                candidate,
                alternative.label,
                ids,
            )
            continue

        unresolved = [item_id for item_id in ids if item_id not in index]
        if unresolved:
            reject("evidence_id_unresolved", candidate, alternative.label, unresolved)

        if alternative.status in DECIDED_ALTERNATIVE_STATUSES and not ids:
            reject("alternative_without_evidence", candidate, alternative.label, ids)

        alt_status[candidate] = alternative.status

    # -- cross-structure consistency --------------------------------------
    temporary = alt_status.get("temporary_variation")
    h1 = hyp_status.get("H1")
    if temporary is not None and h1 is not None:
        if (temporary == "supported") != (h1 == "supported"):
            contradict(
                "hypothesis_alternative_conflict",
                "temporary_variation",
                "the temporary-variation alternative disagrees with H1",
                subject_ids.get("temporary_variation", []),
            )

    insufficient = alt_status.get("insufficient_data")
    meets = _meets_minimum(result)
    if insufficient is not None and meets is not None:
        if (insufficient == "supported") != (not meets):
            contradict(
                "hypothesis_alternative_conflict",
                "insufficient_data",
                "the insufficient-data alternative disagrees with the data quality",
                subject_ids.get("insufficient_data", []),
            )

    # -- the engine's own summary -----------------------------------------
    assessment = result.final_assessment or {}
    if assessment.get("stop_reason") != result.stop_reason:
        contradict(
            "assessment_mismatch",
            "stop_reason",
            f"assessment={assessment.get('stop_reason')!r} result={result.stop_reason!r}",
        )

    result_statuses = {
        hid: status for hid, status in zip(_CATALOG_IDS, (h.status for h in hypotheses))
    }
    assessment_statuses = {
        entry.get("id"): entry.get("status")
        for entry in (assessment.get("hypotheses") or [])
        if isinstance(entry, dict)
    }
    if assessment_statuses and assessment_statuses != result_statuses:
        contradict(
            "assessment_mismatch",
            "hypotheses",
            "final_assessment hypothesis statuses disagree with the result",
        )

    supported_ids = sorted(
        hid for hid, status in result_statuses.items() if status == "supported"
    )
    if list(assessment.get("supported_hypotheses") or []) != supported_ids:
        contradict(
            "assessment_mismatch",
            "supported_hypotheses",
            "final_assessment supported hypotheses disagree with the result",
        )

    assessment_open = [
        entry.get("key")
        for entry in (assessment.get("open_questions") or [])
        if isinstance(entry, dict)
    ]
    if list(result.missing_evidence or []) != assessment_open:
        contradict(
            "assessment_mismatch",
            "missing_evidence",
            "final_assessment open questions disagree with the result",
        )

    if assessment.get("evidence_digest") != result.evidence_digest:
        contradict(
            "evidence_digest_mismatch",
            "evidence_digest",
            "the assessment digest differs from the result digest",
        )

    # -- clinical language -------------------------------------------------
    surfaces = []
    for hid, hypothesis in hyp_pairs:
        surfaces.append((hid, hypothesis.hypothesis, subject_ids.get(hid, [])))
    for alternative in alternatives:
        surfaces.append(
            (
                alternative.candidate,
                f"{alternative.label} {alternative.reason}",
                subject_ids.get(alternative.candidate, []),
            )
        )
    for item_id, item in index.items():
        surfaces.append((item_id, item.statement, [item_id]))
    surfaces.append(("limitations", " ".join(result.limitations or []), []))
    surfaces.append(("next_action", (result.state.next_action or ""), []))
    surfaces.append(("final_assessment", json.dumps(assessment, default=str), []))

    for subject, text, ids in surfaces:
        cleaned = _strip_exemptions(text)
        hits = [term for term in CLINICAL_TERMS if term in cleaned]
        if hits:
            reject("clinical_language", subject, text, ids)

    rejected = {rejection.subject for rejection in rejections}
    accepted_hypotheses = [hid for hid in _CATALOG_IDS if hid not in rejected]
    accepted_alternatives = [
        candidate for candidate, _ in ALTERNATIVE_CATALOG if candidate not in rejected
    ]
    accepted_evidence = [item_id for item_id in index if item_id not in rejected]

    verdict = Critique(
        is_sound=not rejections and not contradictions,
        rejections=rejections,
        contradictions=contradictions,
        accepted_evidence_ids=accepted_evidence,
        accepted_hypothesis_ids=accepted_hypotheses,
        accepted_alternative_candidates=accepted_alternatives,
        denied_claim_count=len(rejections),
    )

    if trace is not None:
        for rejection in verdict.rejections:
            trace.record_critic_rejection(
                "critic",
                evidence_ids=rejection.evidence_ids,
                detail=f"{rejection.reason}:{rejection.subject}",
            )
        for contradiction in verdict.contradictions:
            trace.record_state_change(
                "critic",
                detail=f"{contradiction.code}:{contradiction.subject}",
            )
        trace.record_decision(
            "critic",
            "critique_sound" if verdict.is_sound else "critique_rejections",
            detail=(
                f"is_sound={verdict.is_sound} "
                f"rejections={len(verdict.rejections)} "
                f"contradictions={len(verdict.contradictions)}"
            ),
        )

    return verdict
