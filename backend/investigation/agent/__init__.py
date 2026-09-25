"""MindKey investigation agent foundations (Phase 3.1).

This subpackage sits on top of the deterministic Phase 2 evidence layer. Phase
3.1 provides only the foundations:

- ``trace``: the ordered investigation trace (``TraceEvent``,
  ``InvestigationTrace``).
- ``evidence``: the ``EvidenceItem`` registry, the closed set of deterministic
  claim templates, the pure builders that translate a Phase 2
  ``BehavioralEvidence`` package into citable evidence items, and
  ``CachingRepository`` (a read-only, memoizing repository wrapper).
- ``persistence``: per-session persistence depth, onset and window robustness,
  plus the eligibility gate (which can only narrow Phase 2's verdict).
- ``hypotheses``: the fixed H1-H4 behavioral hypothesis catalog, its
  evidence-linked evaluation, and the missing-evidence question registry.
- ``alternatives``: the deterministic alternative-explanation scan.
- ``engine`` (Phase 3.3): the bounded, deterministic investigation state
  machine that orchestrates the modules above and returns an
  ``InvestigationResult``.

Explicit non-goals for this phase (by design):

- No LLM and no LangGraph. No new dependencies at all: only Pydantic and the
  standard library are used.
- The engine is orchestration only: it adds no reasoning rules, no thresholds,
  and no claims.
- No database writes, no schema changes, no API routes, and no dashboard work.
- No typed content is read or processed: only structured numeric behavioral
  features, ids, timestamps, counts and statuses.
- No medical diagnosis and no causal claims. An anomaly score is a number.

Layering rule: this subpackage may import from ``investigation`` (the frozen
Phase 2 layer). Nothing in ``investigation`` may import from
``investigation.agent`` -- that is why ``AgentState`` carries the evidence and
trace as plain dicts and validates them here.
"""

from .alternatives import (
    ALTERNATIVE_CATALOG,
    CONTEXT_CANDIDATES,
    SOURCE_ALTERNATIVES,
    AlternativeFinding,
    AlternativeStatus,
    assess_alternatives,
)
from .engine import (
    ALT_WINDOWS,
    DEFAULT_RECENT_LIMIT,
    ENGINE_SCHEMA_VERSION,
    MAX_ITERATIONS,
    MAX_TOOL_CALLS,
    NODE_NAMES,
    CollectionRequest,
    InvestigationResult,
    StopReason,
    run_investigation,
    select_next_collection,
)
from .evidence import (
    CLAIM_TEMPLATES,
    EVIDENCE_SCHEMA_VERSION,
    SOURCE_ADAPTER,
    SOURCE_ML,
    CachingRepository,
    EvidenceItem,
    EvidenceKind,
    EvidenceRegistry,
    format_count,
    format_percent,
    format_value,
    register_from_behavioral_evidence,
    render_claim,
)
from .hypotheses import (
    HYPOTHESIS_CATALOG,
    QUESTION_KEYS,
    QUESTION_TEXT,
    QUESTION_TOOLS,
    UNANSWERABLE_QUESTIONS,
    HypothesisEvaluation,
    HypothesisId,
    HypothesisReasoning,
    HypothesisSpec,
    answerable_questions,
    create_hypotheses,
    evaluate_hypotheses,
    is_answerable,
    question_text,
    unanswerable_questions,
)
from .persistence import (
    SOURCE_PERSISTENCE,
    PersistenceClassification,
    PersistenceFinding,
    RobustnessStatus,
    assess_persistence,
    assess_robustness,
    baseline_reference,
    deviating_keys,
    moved_signature_from_drift,
    moved_signature_from_stats,
)
from .trace import (
    EVENT_LABELS,
    FINGERPRINT_LENGTH,
    TraceEvent,
    TraceEventType,
    InvestigationTrace,
    args_fingerprint,
)

__all__ = [
    "ALT_WINDOWS",
    "ALTERNATIVE_CATALOG",
    "CLAIM_TEMPLATES",
    "CONTEXT_CANDIDATES",
    "CachingRepository",
    "CollectionRequest",
    "DEFAULT_RECENT_LIMIT",
    "ENGINE_SCHEMA_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "EVENT_LABELS",
    "EvidenceItem",
    "EvidenceKind",
    "EvidenceRegistry",
    "FINGERPRINT_LENGTH",
    "HYPOTHESIS_CATALOG",
    "HypothesisEvaluation",
    "HypothesisId",
    "HypothesisReasoning",
    "HypothesisSpec",
    "InvestigationResult",
    "InvestigationTrace",
    "MAX_ITERATIONS",
    "MAX_TOOL_CALLS",
    "NODE_NAMES",
    "PersistenceClassification",
    "PersistenceFinding",
    "QUESTION_KEYS",
    "QUESTION_TEXT",
    "QUESTION_TOOLS",
    "RobustnessStatus",
    "SOURCE_ADAPTER",
    "SOURCE_ALTERNATIVES",
    "SOURCE_ML",
    "SOURCE_PERSISTENCE",
    "StopReason",
    "TraceEvent",
    "TraceEventType",
    "UNANSWERABLE_QUESTIONS",
    "AlternativeFinding",
    "AlternativeStatus",
    "answerable_questions",
    "args_fingerprint",
    "assess_alternatives",
    "assess_persistence",
    "assess_robustness",
    "baseline_reference",
    "create_hypotheses",
    "deviating_keys",
    "evaluate_hypotheses",
    "format_count",
    "format_percent",
    "format_value",
    "is_answerable",
    "moved_signature_from_drift",
    "moved_signature_from_stats",
    "question_text",
    "register_from_behavioral_evidence",
    "render_claim",
    "run_investigation",
    "select_next_collection",
    "unanswerable_questions",
]
