"""Per-session persistence analysis (Phase 3.2).

Relationship to Phase 2
-----------------------
``BehavioralEvidence.temporal_analysis.persistence`` (Phase 2's
``PersistenceAssessment``) is the **authoritative** persistence verdict. This
module never recomputes, overrides or replaces it. It adds only:

- **depth**: how many of the most recent valid sessions deviate from the
  baseline mean, walking backwards to form a trailing run,
- **onset**: when that run began,
- **robustness**: whether alternative comparison windows agree with the default
  recent/baseline windows,
- **eligibility**: a single conservative conjunction that can only *narrow*
  Phase 2's verdict, never widen it.

Because eligibility requires ``phase2_status == "persistent_change"``, this
module can refuse a persistence claim that Phase 2 would allow, and can never
allow one that Phase 2 refuses. Phase 2's own thresholds are imported, not
re-declared: ``MOVED_RELATIVE_THRESHOLD`` (0.10) and
``MIN_RECENT_SESSIONS_FOR_PERSISTENCE`` (5) from ``investigation.adapter``.

The reference baseline
----------------------
Deviations are measured against ``BehavioralEvidence.signals[*].baseline_value``,
which Phase 2 populates from ``calculate_behavioral_drift``'s per-feature
baseline mean -- the ``baseline_window_days`` (30) *immediately preceding* the
recent window (roughly days 7-37 ago), **not** the last 30 days. No new baseline
is computed here.

No I/O
------
This module performs no data access of any kind: it consumes objects the caller
already fetched. It has no ``repository`` parameter, by design.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from ..adapter import (
    DEFAULT_RECENT_WINDOW_DAYS,
    MIN_RECENT_SESSIONS_FOR_PERSISTENCE,
    MOVED_RELATIVE_THRESHOLD,
)
from ..bridge import canonical_feature_value
from ..contracts import BehavioralEvidence, FeatureStats
from ..tools import BehavioralDriftOutput, RecentSessionsOutput
from ..units import CANONICAL_FEATURE_KEYS
from .evidence import EvidenceRegistry
from .trace import InvestigationTrace

#: Recorded as the ``source_tool`` of the evidence this module registers.
SOURCE_PERSISTENCE = "assess_persistence"

#: Depth classification. Deliberately distinct from Phase 2's
#: ``PersistenceStatus`` vocabulary so the two can never be confused.
PersistenceClassification = Literal[
    "insufficient_data",
    "no_current_deviation",
    "single_session",
    "recent_run",
    "sustained",
]

RobustnessStatus = Literal["not_assessed", "agrees", "disagrees"]

DowngradeReason = Literal[
    "insufficient_history",
    "phase2_status_not_persistent",
    "persistence_depth_insufficient",
    "window_robustness_disagrees",
]

class PersistenceFinding(BaseModel):
    """What this module measured, plus the Phase 2 verdict it is constrained by."""

    # Phase 2's verdict, carried verbatim. Never recomputed here.
    phase2_status: str
    phase2_signals_moved: int
    phase2_supporting_sessions: int

    eligible_for_persistence_claim: bool
    downgrade_reason: DowngradeReason | None = None

    classification: PersistenceClassification
    depth: int | None = None
    onset: datetime | None = None
    valid_recent_sessions: int = 0
    deviating_feature_keys: list[str] = Field(default_factory=list)
    window_days: int | None = None
    minimum_sessions: int = 0
    minimum_recent_sessions: int = MIN_RECENT_SESSIONS_FOR_PERSISTENCE

    robustness: RobustnessStatus = "not_assessed"
    robustness_comparisons: int = 0
    robustness_conflicts: int = 0

    evidence_ids: list[str] = Field(default_factory=list)


def _aware(value):
    """Return an aware datetime (assumes UTC when naive), or ``None``."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def baseline_reference(evidence: BehavioralEvidence) -> dict[str, float]:
    """The per-feature baseline means the depth walk compares against.

    Zero and missing baselines are dropped rather than divided by (this mirrors
    Phase 2's ``_relative_change`` returning ``None`` for a zero baseline).
    """
    reference = {}
    for signal in evidence.signals:
        if signal.baseline_value is None or signal.baseline_value == 0:
            continue
        reference[signal.key] = float(signal.baseline_value)
    return reference


def deviating_keys(session, reference) -> list[str]:
    """Canonical feature keys in which ``session`` deviates from the baseline.

    A feature deviates when the relative difference from its baseline mean
    reaches ``MOVED_RELATIVE_THRESHOLD``.
    """
    keys = []
    for key in CANONICAL_FEATURE_KEYS:
        if key not in reference:
            continue
        mean = reference[key]
        if mean is None or mean == 0:
            continue
        value = canonical_feature_value(session, key)
        if value is None or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        if abs(value - mean) / abs(mean) >= MOVED_RELATIVE_THRESHOLD:
            keys.append(key)
    return keys


def _valid_recent_sessions(recent_sessions, as_of, window_days):
    """Valid sessions, optionally restricted to ``]as_of - window_days, as_of]``.

    Sessions without a usable timestamp are excluded from the windowed walk
    rather than assumed into it (mirroring Phase 2's ``_within``).
    """
    if recent_sessions is None:
        return []
    sessions = [s for s in getattr(recent_sessions, "sessions", None) or [] if s.is_valid]

    if as_of is not None and window_days:
        reference = _aware(as_of)
        cutoff = reference - timedelta(days=window_days)
        kept = []
        for session in sessions:
            timestamp = _aware(session.session_start)
            if timestamp is None:
                continue
            if cutoff <= timestamp <= reference:
                kept.append(session)
        sessions = kept

    floor = datetime.min.replace(tzinfo=timezone.utc)
    sessions.sort(key=lambda s: _aware(s.session_start) or floor)
    return sessions


def _trailing_run(valid_sessions, reference):
    """Depth, onset and deviating keys of the current trailing run.

    Walks from the newest session backwards, counting *deviating* sessions and
    tolerating one isolated non-deviating session inside the run (a fresh
    deviation resets that allowance). The run ends at the first non-deviating
    session when no allowance is left, or when the newest session does not
    deviate at all.
    """
    depth = 0
    onset = None
    gap_used = False
    seen = set()

    for session in reversed(valid_sessions):
        keys = deviating_keys(session, reference)
        if keys:
            depth += 1
            gap_used = False
            onset = _aware(session.session_start)
            seen.update(keys)
        elif depth == 0 or gap_used:
            break
        else:
            gap_used = True

    ordered = [key for key in CANONICAL_FEATURE_KEYS if key in seen]
    return depth, onset, ordered


def _classify(depth) -> PersistenceClassification:
    if depth is None:
        return "insufficient_data"
    if depth == 0:
        return "no_current_deviation"
    if depth == 1:
        return "single_session"
    if depth < MIN_RECENT_SESSIONS_FOR_PERSISTENCE:
        return "recent_run"
    return "sustained"


def moved_signature_from_stats(
    recent_stats: Sequence[FeatureStats], baseline_stats: Sequence[FeatureStats]
) -> dict[str, int]:
    """``key -> +1/-1`` for features whose window means moved materially."""
    baseline_by_key = {stat.key: stat for stat in baseline_stats}
    signature = {}
    for stat in recent_stats:
        base = baseline_by_key.get(stat.key)
        if base is None or not stat.available or not base.available:
            continue
        if stat.mean is None or base.mean in (None, 0):
            continue
        relative = (stat.mean - base.mean) / base.mean
        if abs(relative) < MOVED_RELATIVE_THRESHOLD:
            continue
        signature[stat.key] = 1 if relative > 0 else -1
    return signature


def moved_signature_from_drift(drift: BehavioralDriftOutput) -> dict[str, int]:
    """``key -> +1/-1`` for features that moved materially in a drift result."""
    signature = {}
    for feature in getattr(drift, "features", None) or []:
        if not feature.available or feature.relative_change is None:
            continue
        if abs(feature.relative_change) < MOVED_RELATIVE_THRESHOLD:
            continue
        if feature.relative_change > 0:
            signature[feature.key] = 1
        elif feature.relative_change < 0:
            signature[feature.key] = -1
    return signature


def assess_robustness(default_signature, alt_drifts):
    """Compare alternative window results with the default window signature.

    Returns ``(status, comparisons, conflicts)``. A window disagrees when it
    conflicts on more than half of the signals the two windows share; windows
    with nothing comparable are ignored. No comparable window at all yields
    ``"not_assessed"`` (which does not block a claim -- only disagreement does).
    """
    comparisons = 0
    conflicts = 0
    disagreeing = False

    for drift in alt_drifts or ():
        if drift is None:
            continue
        alternative = moved_signature_from_drift(drift)
        comparable = [key for key in alternative if key in default_signature]
        if not comparable:
            continue
        comparisons += 1
        bad = sum(1 for key in comparable if alternative[key] != default_signature[key])
        conflicts += bad
        if bad / len(comparable) > 0.5:
            disagreeing = True

    if comparisons == 0:
        return "not_assessed", comparisons, conflicts
    return ("disagrees" if disagreeing else "agrees"), comparisons, conflicts


def assess_persistence(
    registry: EvidenceRegistry,
    evidence: BehavioralEvidence,
    recent_sessions: RecentSessionsOutput | None,
    *,
    alt_drifts=(),
    as_of: datetime | None = None,
    window_days: int = DEFAULT_RECENT_WINDOW_DAYS,
    trace: InvestigationTrace | None = None,
) -> PersistenceFinding:
    """Measure persistence depth, onset and robustness, and gate the claim.

    Args:
        registry: the evidence registry to cite into.
        evidence: the Phase 2 package (its ``persistence`` verdict is
            authoritative and only ever narrowed).
        recent_sessions: ``get_recent_sessions`` output. ``None`` is treated as
            "no usable recent sessions".
        alt_drifts: additional ``calculate_behavioral_drift`` results used to
            test whether the result survives different window sizes.
        as_of: when given, sessions older than ``window_days`` before it are
            excluded (keeps this function independent of the wall clock).
        window_days: the recent window size used for the depth walk.
        trace: optional investigation trace.

    Returns:
        PersistenceFinding. Phase 2's verdict is carried verbatim; eligibility
        can only be more conservative than it.
    """
    if not isinstance(registry, EvidenceRegistry):
        raise TypeError("registry must be an EvidenceRegistry")

    quality = evidence.data_quality
    phase2 = evidence.temporal_analysis.persistence

    reference = baseline_reference(evidence)
    valid = _valid_recent_sessions(recent_sessions, as_of, window_days)

    if len(valid) < MIN_RECENT_SESSIONS_FOR_PERSISTENCE:
        depth, onset, deviating = None, None, []
    else:
        depth, onset, deviating = _trailing_run(valid, reference)

    classification = _classify(depth)

    default_signature = moved_signature_from_stats(
        evidence.temporal_analysis.recent_feature_stats,
        evidence.temporal_analysis.baseline_feature_stats,
    )
    robustness, comparisons, conflicts = assess_robustness(default_signature, alt_drifts)

    # -- evidence ----------------------------------------------------------
    if depth is None:
        depth_item = registry.register_unavailable(
            "persistence_depth",
            kind="persistence_depth",
            source_tool=SOURCE_PERSISTENCE,
            reason=(
                f"fewer than {MIN_RECENT_SESSIONS_FOR_PERSISTENCE} valid sessions "
                f"are available in the recent window ({len(valid)} present)"
            ),
            session_count=len(valid),
            window_days=window_days,
            threshold=MIN_RECENT_SESSIONS_FOR_PERSISTENCE,
            status=classification,
        )
    else:
        depth_item = registry.register(
            "persistence_depth",
            kind="persistence_depth",
            source_tool=SOURCE_PERSISTENCE,
            value=float(depth),
            session_count=len(valid),
            window_days=window_days,
            threshold=MIN_RECENT_SESSIONS_FOR_PERSISTENCE,
            status=classification,
            onset=onset,
        )

    robustness_item = registry.register(
        "persistence_robustness",
        kind="persistence_robustness",
        source_tool=SOURCE_PERSISTENCE,
        status=robustness,
        value=float(conflicts),
        session_count=comparisons,
    )

    # -- eligibility: only ever narrower than Phase 2 ----------------------
    # Precedence is deliberate and fixed: a closed quality gate is reported
    # before a non-persistent Phase 2 verdict, which is reported before thin
    # depth, which is reported before window disagreement.
    failed = None
    if not quality.meets_model_minimum:
        failed = "insufficient_history"
    elif phase2.status != "persistent_change":
        failed = "phase2_status_not_persistent"
    elif depth is None or depth < MIN_RECENT_SESSIONS_FOR_PERSISTENCE:
        failed = "persistence_depth_insufficient"
    elif robustness == "disagrees":
        failed = "window_robustness_disagrees"

    finding = PersistenceFinding(
        phase2_status=phase2.status,
        phase2_signals_moved=phase2.signals_moved,
        phase2_supporting_sessions=phase2.supporting_session_count,
        eligible_for_persistence_claim=failed is None,
        downgrade_reason=failed,
        classification=classification,
        depth=depth,
        onset=onset,
        valid_recent_sessions=len(valid),
        deviating_feature_keys=deviating,
        window_days=window_days,
        minimum_sessions=quality.model_minimum_sessions,
        robustness=robustness,
        robustness_comparisons=comparisons,
        robustness_conflicts=conflicts,
        evidence_ids=[depth_item.id, robustness_item.id],
    )

    if trace is not None:
        trace.record_evidence(
            "assess_persistence",
            finding.evidence_ids,
            detail=f"{classification}/{robustness}",
        )

    return finding
