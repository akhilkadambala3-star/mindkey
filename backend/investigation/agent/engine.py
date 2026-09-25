"""Deterministic investigation engine (Phase 3.3).

This module is the orchestration layer. It is **not** a reasoning layer: every
rule, threshold, hypothesis, alternative and claim already lives in Phase 2
(``investigation``) and Phase 3.1/3.2 (``investigation.agent``). The engine only
decides *what to do next* -- which node runs, which tool (if any) is justified,
whether to keep collecting, and why the investigation stopped -- and it does so
with fixed, inspectable rules.

Design
------
A bounded state machine over nine nodes::

    INGEST_SIGNAL -> CHECK_DATA_QUALITY -> INITIALIZE_HYPOTHESES
      -> [ INVESTIGATE -> COLLECT_REQUIRED_EVIDENCE -> ASSESS_PERSISTENCE
           -> UPDATE_HYPOTHESES -> CHECK_STOP_CONDITIONS ]* -> FINALIZE

Each node is a small function ``node(ctx) -> next_node_name``. The only I/O is
the read-only ``CachingRepository`` passed in; the engine never writes.

Documented deviation from the conceptual master ordering
--------------------------------------------------------
The master flow lists ``ASSESS_PERSISTENCE`` before ``INITIALIZE_HYPOTHESES``.
The engine initialises the (pure, order-independent) hypothesis catalog first
and assesses persistence *after* the required window-robustness evidence has
been collected. This is deliberate and approved: Phase 3.2's safety rule
"robustness disagreement blocks the persistence claim" can only act if
persistence is assessed against the collected alternative windows.

Single persistence assessment
-----------------------------
Phase 3.2's ``assess_persistence`` is called **exactly once**, on the main
``EvidenceRegistry``, so no ``persistence_depth`` / ``persistence_robustness``
item is ever duplicated. Phase 2's persistence status is never recomputed,
replaced or overridden.

No LLM, no LangGraph, no new dependencies. This module imports only the standard
library, Pydantic, and the frozen Phase 2/3.1/3.2 modules.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from ..contracts import BehavioralEvidence
from ..repository import RepositoryError, default_repository
from ..state import AgentState, Hypothesis
from ..tools import (
    BehavioralDriftOutput,
    MLEvidenceOutput,
    RecentSessionsOutput,
    calculate_behavioral_drift,
    compare_time_windows,
    get_historical_baseline,
    get_ml_evidence,
    get_recent_sessions,
)
from ..adapter import DEFAULT_RECENT_WINDOW_DAYS, build_behavioral_evidence
from .alternatives import AlternativeFinding, assess_alternatives
from .evidence import (
    CachingRepository,
    EvidenceRegistry,
    register_from_behavioral_evidence,
)
from .hypotheses import (
    QUESTION_KEYS,
    QUESTION_TOOLS,
    HypothesisEvaluation,
    create_hypotheses,
    evaluate_hypotheses,
    is_answerable,
    question_text,
)
from .persistence import PersistenceFinding, assess_persistence
from .trace import InvestigationTrace, args_fingerprint

#: Version of the engine result shape (independent of the evidence schema).
ENGINE_SCHEMA_VERSION = "1.0"

# -- Fixed, documented constants (approved for Phase 3.3) --------------------

#: Maximum number of collection passes through the loop.
MAX_ITERATIONS = 4

#: Maximum number of tool calls in one investigation.
MAX_TOOL_CALLS = 8

#: Alternative comparison-window pairs used to test persistence robustness
#: (Phase 2's default is recent=7 vs baseline=30).
ALT_WINDOWS: tuple[tuple[int, int], ...] = ((14, 14),)

#: Sessions fetched for the persistence depth walk.
DEFAULT_RECENT_LIMIT = 500

#: Windows used for the ``longer_horizon`` question, if it is ever asked.
LONGER_HORIZON_WINDOWS = (30, 60)

# -- Node names (closed set) -------------------------------------------------

NODE_INGEST_SIGNAL = "ingest_signal"
NODE_CHECK_DATA_QUALITY = "check_data_quality"
NODE_INITIALIZE_HYPOTHESES = "initialize_hypotheses"
NODE_INVESTIGATE = "investigate"
NODE_COLLECT_REQUIRED_EVIDENCE = "collect_required_evidence"
NODE_ASSESS_PERSISTENCE = "assess_persistence"
NODE_UPDATE_HYPOTHESES = "update_hypotheses"
NODE_CHECK_STOP_CONDITIONS = "check_stop_conditions"
NODE_FINALIZE = "finalize"

NODE_NAMES: tuple[str, ...] = (
    NODE_INGEST_SIGNAL,
    NODE_CHECK_DATA_QUALITY,
    NODE_INITIALIZE_HYPOTHESES,
    NODE_INVESTIGATE,
    NODE_COLLECT_REQUIRED_EVIDENCE,
    NODE_ASSESS_PERSISTENCE,
    NODE_UPDATE_HYPOTHESES,
    NODE_CHECK_STOP_CONDITIONS,
    NODE_FINALIZE,
)

_DONE = "__done__"

#: The bootstrap evidence every adequate investigation needs.
PLAN_RECENT_SESSIONS = "recent_sessions"
PLAN_WINDOW_ROBUSTNESS = "window_robustness"

#: Fixed stop reasons. Pydantic rejects anything else.
StopReason = Literal[
    "evidence_sufficient",
    "evidence_insufficient",
    "no_further_evidence",
    "iteration_limit",
    "tool_budget_exhausted",
    "data_unavailable",
    "only_unanswerable_questions_remain",
]

#: Fixed, system-authored limitation added by the engine itself.
ENGINE_LIMITATION = (
    "This investigation is a deterministic, rule-based orchestration of the "
    "measured evidence. It uses no language model and establishes no cause."
)

DISCLAIMER = (
    "This assessment describes measured behavioral signals only; it "
    "establishes no cause and makes no medical claim."
)


# ---------------------------------------------------------------------------
# Routing inputs / outputs
# ---------------------------------------------------------------------------


@dataclass
class CollectionRequest:
    """One justified, not-yet-performed tool call."""

    step: str
    tool: str
    args: dict
    question: str | None = None


def _fingerprint(request: CollectionRequest) -> str:
    return args_fingerprint(request.tool, **request.args)


def _plan_requests(step, *, user_id, recent_limit, alt_windows):
    """Every candidate request for one bootstrap plan step, in fixed order."""
    if step == PLAN_RECENT_SESSIONS:
        return [
            CollectionRequest(
                step,
                "get_recent_sessions",
                {"user_id": user_id, "limit": recent_limit, "window_days": None},
            )
        ]
    if step == PLAN_WINDOW_ROBUSTNESS:
        return [
            CollectionRequest(
                step,
                "calculate_behavioral_drift",
                {
                    "user_id": user_id,
                    "recent_window_days": recent,
                    "baseline_window_days": baseline,
                },
            )
            for recent, baseline in alt_windows
        ]
    return []


def _question_requests(key, *, user_id, recent_limit, alt_windows):
    """Every candidate request for one open missing-evidence question."""
    requests = []
    for tool in QUESTION_TOOLS.get(key, ()):
        if tool == "get_recent_sessions":
            requests.append(
                CollectionRequest(
                    key,
                    tool,
                    {"user_id": user_id, "limit": recent_limit, "window_days": None},
                    question=key,
                )
            )
        elif tool == "calculate_behavioral_drift":
            requests.extend(
                CollectionRequest(
                    key,
                    tool,
                    {
                        "user_id": user_id,
                        "recent_window_days": recent,
                        "baseline_window_days": baseline,
                    },
                    question=key,
                )
                for recent, baseline in alt_windows
            )
        elif tool == "compare_time_windows":
            requests.append(
                CollectionRequest(
                    key,
                    tool,
                    {
                        "user_id": user_id,
                        "window_a_days": LONGER_HORIZON_WINDOWS[0],
                        "window_b_days": LONGER_HORIZON_WINDOWS[1],
                    },
                    question=key,
                )
            )
        elif tool == "get_historical_baseline":
            requests.append(
                CollectionRequest(
                    key,
                    tool,
                    {"user_id": user_id, "window_days": LONGER_HORIZON_WINDOWS[1]},
                    question=key,
                )
            )
    return requests


def select_next_collection(
    *,
    user_id,
    plan=(),
    collected_plan=(),
    open_questions=(),
    called_fingerprints=(),
    alt_windows=ALT_WINDOWS,
    recent_limit=DEFAULT_RECENT_LIMIT,
    budget_remaining=MAX_TOOL_CALLS,
):
    """Pick the next justified tool call, or ``None``.

    Pure and deterministic. Bootstrap plan steps are served first, in order,
    then open questions in the canonical :data:`QUESTION_KEYS` order. A request
    whose argument fingerprint was already called is skipped, which is what
    stops the engine from re-reading the same data. Questions whose only tools
    have all been called are therefore dead ends and yield ``None``.
    """
    if budget_remaining <= 0:
        return None

    collected = set(collected_plan or ())
    called = set(called_fingerprints or ())

    for step in plan or ():
        if step in collected:
            continue
        for request in _plan_requests(
            step, user_id=user_id, recent_limit=recent_limit, alt_windows=alt_windows
        ):
            if _fingerprint(request) in called:
                continue
            return request

    present = set(open_questions or ())
    for key in QUESTION_KEYS:
        if key not in present:
            continue
        for request in _question_requests(
            key, user_id=user_id, recent_limit=recent_limit, alt_windows=alt_windows
        ):
            if _fingerprint(request) in called:
                continue
            return request

    return None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class InvestigationResult(BaseModel):
    """The complete, structured outcome of one investigation.

    Mirrored fields (``hypotheses``, ``missing_evidence``, ``alternatives``,
    ``stop_reason``, ``final_assessment``) are populated from the same
    authoritative ``state``, so the two can never disagree.
    """

    schema_version: str = ENGINE_SCHEMA_VERSION
    user_id: str
    session_id: str
    stop_reason: StopReason
    iterations: int
    tool_calls: list[dict] = Field(default_factory=list)
    evidence_digest: str
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    alternatives: list[AlternativeFinding] = Field(default_factory=list)
    final_assessment: dict = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    state: AgentState


# ---------------------------------------------------------------------------
# Internal run context
# ---------------------------------------------------------------------------


@dataclass
class _Context:
    user_id: str
    session_id: str
    caching: object
    trace: InvestigationTrace
    state: AgentState
    registry: EvidenceRegistry
    as_of: datetime
    max_iterations: int
    max_tool_calls: int
    alt_windows: tuple
    blocking_reason: str | None = None

    evidence: BehavioralEvidence | None = None
    anomaly: MLEvidenceOutput | None = None
    recent_sessions: RecentSessionsOutput | None = None
    alt_drifts: list[BehavioralDriftOutput] = field(default_factory=list)
    finding: PersistenceFinding | None = None
    evaluation: HypothesisEvaluation | None = None
    alternatives: list[AlternativeFinding] = field(default_factory=list)

    plan: list[str] = field(default_factory=list)
    collected_plan: set = field(default_factory=set)
    pending: CollectionRequest | None = None
    called: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    nodes_visited: list[str] = field(default_factory=list)
    no_more_evidence: bool = False
    iterations: int = 0
    stop_reason: str | None = None


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _unavailable_reason(result):
    reason = getattr(result, "unavailable_reason", None)
    if reason:
        return str(reason)
    return "unavailable"


def _register_compare_windows(ctx, result):
    """Register the recent-window means of a ``compare_time_windows`` result."""
    ids = []
    window = result.window_a
    if window is None:
        return ids
    for stat in window.stats:
        if not stat.available or stat.mean is None:
            continue
        item = ctx.registry.register(
            "window_stat",
            kind="window_stat",
            source_tool="compare_time_windows",
            key=stat.key,
            label=stat.label,
            unit=stat.canonical_unit,
            value=stat.mean,
            session_count=stat.sample_count,
            window_days=window.window_days,
        )
        ids.append(item.id)
    return ids


def _run_tool(ctx, request):
    """Execute one tool call, register its evidence, and record the trace.

    Returns ``(result, evidence_ids)``. A result is never invented: an
    unavailable or failing tool is recorded as explicit missing evidence.
    """
    tool = request.tool
    args = dict(request.args)
    fingerprint = _fingerprint(request)

    if fingerprint in ctx.called:
        ctx.trace.record_decision(
            NODE_COLLECT_REQUIRED_EVIDENCE,
            "duplicate_suppressed",
            detail=f"duplicate_suppressed:{tool}",
        )
        return None, []

    ctx.trace.record_tool_call(NODE_COLLECT_REQUIRED_EVIDENCE, tool, args=args)

    result = None
    try:
        if tool == "get_recent_sessions":
            result = get_recent_sessions(
                args["user_id"],
                limit=args["limit"],
                window_days=args.get("window_days"),
                repository=ctx.caching,
            )
        elif tool == "calculate_behavioral_drift":
            result = calculate_behavioral_drift(
                args["user_id"],
                recent_window_days=args["recent_window_days"],
                baseline_window_days=args["baseline_window_days"],
                repository=ctx.caching,
            )
        elif tool == "compare_time_windows":
            result = compare_time_windows(
                args["user_id"],
                window_a_days=args["window_a_days"],
                window_b_days=args["window_b_days"],
                repository=ctx.caching,
            )
        elif tool == "get_historical_baseline":
            result = get_historical_baseline(
                args["user_id"],
                window_days=args["window_days"],
                repository=ctx.caching,
            )
    except RepositoryError:
        result = None
    except Exception:  # defensive: a tool must never abort the investigation
        result = None

    ctx.called[fingerprint] = tool

    available = bool(getattr(result, "available", False))
    evidence_ids = []

    if available:
        if tool == "get_recent_sessions":
            ctx.recent_sessions = result
        elif tool == "calculate_behavioral_drift":
            ctx.alt_drifts.append(result)
        elif tool == "compare_time_windows":
            evidence_ids = _register_compare_windows(ctx, result)
        # get_historical_baseline needs no citable item here: Phase 2 already
        # derives the baseline that the evidence layer cites.
    else:
        item = ctx.registry.register_unavailable(
            "tool_unavailable",
            kind="tool_unavailable",
            source_tool=tool,
            reason=_unavailable_reason(result),
        )
        evidence_ids = [item.id]

    ctx.trace.record_tool_result(
        NODE_COLLECT_REQUIRED_EVIDENCE,
        tool,
        evidence_ids=evidence_ids,
        detail="available" if available else "unavailable",
    )
    ctx.tool_calls.append(
        {
            "tool": tool,
            "fingerprint": fingerprint,
            "available": available,
            "evidence_ids": evidence_ids,
            "step": request.step,
        }
    )
    ctx.state.tools_called.append(tool)
    return result, evidence_ids


def _budget_remaining(ctx):
    return ctx.max_tool_calls - len(ctx.tool_calls)


def _plan_complete(ctx):
    """True when no uncollected, still-satisfiable bootstrap step remains."""
    for step in ctx.plan:
        if step in ctx.collected_plan:
            continue
        requests = _plan_requests(
            step,
            user_id=ctx.user_id,
            recent_limit=DEFAULT_RECENT_LIMIT,
            alt_windows=ctx.alt_windows,
        )
        outstanding = [
            request for request in requests if _fingerprint(request) not in ctx.called
        ]
        if outstanding:
            return False
    return True


def _select(ctx):
    return select_next_collection(
        user_id=ctx.user_id,
        plan=ctx.plan,
        collected_plan=ctx.collected_plan,
        open_questions=ctx.evaluation.missing_evidence if ctx.evaluation else (),
        called_fingerprints=ctx.called,
        alt_windows=ctx.alt_windows,
        recent_limit=DEFAULT_RECENT_LIMIT,
        budget_remaining=_budget_remaining(ctx),
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _ingest_signal(ctx):
    ctx.trace.record_enter(NODE_INGEST_SIGNAL)
    ctx.state.trigger = {
        "source": "typing_session",
        "session_id": ctx.session_id,
    }

    if ctx.caching is not None:
        anomaly = get_ml_evidence(
            ctx.user_id, ctx.session_id, repository=ctx.caching
        )
        ctx.anomaly = anomaly
        ctx.state.ml_evidence = anomaly.model_dump(mode="json")

        fingerprint = args_fingerprint(
            "build_behavioral_evidence",
            user_id=ctx.user_id,
            session_id=ctx.session_id,
        )
        ctx.trace.record_tool_call(
            NODE_INGEST_SIGNAL,
            "build_behavioral_evidence",
            args={"user_id": ctx.user_id, "session_id": ctx.session_id},
        )
        try:
            evidence = build_behavioral_evidence(
                ctx.user_id, ctx.session_id, repository=ctx.caching
            )
        except (RepositoryError, ValueError):
            evidence = None
        ctx.evidence = evidence
        ctx.called[fingerprint] = "build_behavioral_evidence"

        evidence_ids = []
        if evidence is not None:
            items = register_from_behavioral_evidence(ctx.registry, evidence, anomaly)
            evidence_ids = [item.id for item in items]
            ctx.state.evidence = ctx.registry.to_dicts()

        ctx.trace.record_tool_result(
            NODE_INGEST_SIGNAL,
            "build_behavioral_evidence",
            evidence_ids=evidence_ids,
            detail="available" if evidence is not None else "unavailable",
        )
    else:
        ctx.trace.record_decision(
            NODE_INGEST_SIGNAL, "data_unavailable", detail="data_unavailable:no_repository"
        )

    ctx.trace.record_exit(NODE_INGEST_SIGNAL)
    return NODE_CHECK_DATA_QUALITY


def _check_data_quality(ctx):
    ctx.trace.record_enter(NODE_CHECK_DATA_QUALITY)

    if ctx.evidence is None:
        if ctx.blocking_reason is None:
            ctx.blocking_reason = "data_unavailable"
        ctx.trace.record_decision(
            NODE_CHECK_DATA_QUALITY, "data_unavailable", detail="data_unavailable:no_evidence"
        )
        ctx.trace.record_exit(NODE_CHECK_DATA_QUALITY)
        return NODE_INITIALIZE_HYPOTHESES

    quality = ctx.evidence.data_quality
    plan = [PLAN_RECENT_SESSIONS]
    if quality.meets_model_minimum:
        plan.append(PLAN_WINDOW_ROBUSTNESS)
    ctx.plan = plan
    ctx.state.investigation_plan = list(plan)

    ctx.trace.record_decision(
        NODE_CHECK_DATA_QUALITY,
        "plan_required_evidence",
        detail="plan_required_evidence:" + ("+".join(plan) if plan else "none"),
    )
    ctx.trace.record_exit(NODE_CHECK_DATA_QUALITY)
    return NODE_INITIALIZE_HYPOTHESES


def _initialize_hypotheses(ctx):
    ctx.trace.record_enter(NODE_INITIALIZE_HYPOTHESES)
    ctx.state.hypotheses = create_hypotheses()
    ctx.trace.record_exit(NODE_INITIALIZE_HYPOTHESES)
    return NODE_INVESTIGATE


def _investigate(ctx):
    ctx.trace.record_enter(NODE_INVESTIGATE)
    ctx.state.iteration = ctx.iterations

    request = _select(ctx)
    if request is None:
        ctx.no_more_evidence = True
        ctx.trace.record_decision(NODE_INVESTIGATE, "no_collectible_evidence")
        ctx.trace.record_exit(NODE_INVESTIGATE)
        return NODE_CHECK_STOP_CONDITIONS

    ctx.no_more_evidence = False
    ctx.pending = request
    ctx.trace.record_decision(
        NODE_INVESTIGATE,
        f"collect:{request.tool}",
        detail=f"collect:{request.tool} step={request.step}",
    )
    ctx.trace.record_exit(NODE_INVESTIGATE)
    return NODE_COLLECT_REQUIRED_EVIDENCE


def _collect_required_evidence(ctx):
    ctx.trace.record_enter(NODE_COLLECT_REQUIRED_EVIDENCE)
    request = ctx.pending
    ctx.pending = None
    if request is None:
        ctx.trace.record_exit(NODE_COLLECT_REQUIRED_EVIDENCE)
        return NODE_ASSESS_PERSISTENCE

    _run_tool(ctx, request)

    if request.step in ctx.plan:
        ctx.collected_plan.add(request.step)

    # An iteration is one executed collection, so a decision-only pass does not
    # consume the bounded iteration budget.
    ctx.iterations += 1
    ctx.state.iteration = ctx.iterations

    ctx.trace.record_exit(NODE_COLLECT_REQUIRED_EVIDENCE)
    return NODE_ASSESS_PERSISTENCE


def _assess_persistence(ctx):
    ctx.trace.record_enter(NODE_ASSESS_PERSISTENCE)

    if ctx.evidence is None:
        ctx.trace.record_exit(NODE_ASSESS_PERSISTENCE)
        return NODE_CHECK_STOP_CONDITIONS

    if ctx.finding is not None:
        ctx.trace.record_decision(
            NODE_ASSESS_PERSISTENCE, "persistence_already_assessed"
        )
        ctx.trace.record_exit(NODE_ASSESS_PERSISTENCE)
        return NODE_UPDATE_HYPOTHESES

    if not _plan_complete(ctx):
        ctx.trace.record_decision(
            NODE_ASSESS_PERSISTENCE, "awaiting_required_evidence"
        )
        ctx.trace.record_exit(NODE_ASSESS_PERSISTENCE)
        return NODE_CHECK_STOP_CONDITIONS

    # Single assessment, on the main registry. Phase 3.2 records its own
    # evidence_added event, so the ids appear in the trace exactly once.
    ctx.finding = assess_persistence(
        ctx.registry,
        ctx.evidence,
        ctx.recent_sessions,
        alt_drifts=ctx.alt_drifts,
        as_of=ctx.as_of,
        window_days=DEFAULT_RECENT_WINDOW_DAYS,
        trace=ctx.trace,
    )
    ctx.trace.record_decision(
        NODE_ASSESS_PERSISTENCE,
        "persistence_assessed",
        detail=(
            f"persistence_assessed {ctx.finding.classification}/"
            f"{ctx.finding.robustness}"
        ),
    )
    ctx.trace.record_exit(NODE_ASSESS_PERSISTENCE)
    return NODE_UPDATE_HYPOTHESES


def _update_hypotheses(ctx):
    ctx.trace.record_enter(NODE_UPDATE_HYPOTHESES)

    if ctx.evidence is None or ctx.finding is None:
        ctx.trace.record_decision(NODE_UPDATE_HYPOTHESES, "awaiting_persistence")
        ctx.trace.record_exit(NODE_UPDATE_HYPOTHESES)
        return NODE_CHECK_STOP_CONDITIONS

    if ctx.evaluation is not None:
        ctx.trace.record_decision(
            NODE_UPDATE_HYPOTHESES, "hypotheses_already_updated"
        )
        ctx.trace.record_exit(NODE_UPDATE_HYPOTHESES)
        return NODE_CHECK_STOP_CONDITIONS

    evaluation = evaluate_hypotheses(
        ctx.registry, ctx.evidence, ctx.finding, create_hypotheses(), trace=ctx.trace
    )
    ctx.evaluation = evaluation
    ctx.state.hypotheses = evaluation.hypotheses
    ctx.state.missing_evidence = evaluation.missing_evidence

    ctx.alternatives = assess_alternatives(
        ctx.registry, ctx.evidence, ctx.finding, trace=ctx.trace
    )

    ctx.trace.record_exit(NODE_UPDATE_HYPOTHESES)
    return NODE_CHECK_STOP_CONDITIONS


def _soft_stop_reason(ctx):
    finding = ctx.finding
    if finding is not None and finding.eligible_for_persistence_claim:
        return "evidence_sufficient"
    open_questions = list(ctx.evaluation.missing_evidence) if ctx.evaluation else []
    if open_questions and all(not is_answerable(key) for key in open_questions):
        return "only_unanswerable_questions_remain"
    return "no_further_evidence"


def _check_stop_conditions(ctx):
    ctx.trace.record_enter(NODE_CHECK_STOP_CONDITIONS)
    ctx.state.iteration = ctx.iterations

    reason = None

    if ctx.evidence is None:
        reason = ctx.blocking_reason or "data_unavailable"
    else:
        quality = ctx.evidence.data_quality
        insufficient = (
            quality.valid_sessions == 0 or not quality.meets_model_minimum
        )
        if _plan_complete(ctx) and insufficient:
            # Evidence insufficiency takes precedence over budget stops, and is
            # never converted into a positive conclusion.
            reason = "evidence_insufficient"
        elif ctx.iterations >= ctx.max_iterations:
            reason = "iteration_limit"
        elif _budget_remaining(ctx) <= 0:
            reason = "tool_budget_exhausted"
        elif ctx.no_more_evidence:
            reason = _soft_stop_reason(ctx)

    if reason is None:
        # Required evidence is still being collected and the budget allows it.
        ctx.trace.record_exit(NODE_CHECK_STOP_CONDITIONS)
        return NODE_INVESTIGATE

    ctx.stop_reason = reason
    ctx.state.stop_reason = reason
    ctx.trace.record_stop(NODE_CHECK_STOP_CONDITIONS, reason)
    ctx.trace.record_exit(NODE_CHECK_STOP_CONDITIONS)
    return NODE_FINALIZE


# ---------------------------------------------------------------------------
# Assessment / finalization helpers
# ---------------------------------------------------------------------------


def _persistence_summary(ctx):
    finding = ctx.finding
    status = "unknown"
    if ctx.evidence is not None:
        status = ctx.evidence.temporal_analysis.persistence.status
    if finding is None:
        return {
            "status": status,
            "eligible": False,
            "downgrade_reason": None,
            "classification": "insufficient_data",
            "depth": None,
            "onset": None,
            "robustness": "not_assessed",
            "signals_moved": 0,
        }
    return {
        "status": finding.phase2_status,
        "eligible": finding.eligible_for_persistence_claim,
        "downgrade_reason": finding.downgrade_reason,
        "classification": finding.classification,
        "depth": finding.depth,
        "onset": None if finding.onset is None else finding.onset.isoformat(),
        "robustness": finding.robustness,
        "signals_moved": finding.phase2_signals_moved,
    }


def _hypothesis_summary(ctx):
    if ctx.evaluation is None:
        return []
    return [
        {"id": reasoning.id, "status": reasoning.status}
        for reasoning in ctx.evaluation.reasoning
    ]


def _build_final_assessment(ctx):
    quality = None
    if ctx.evidence is not None:
        data_quality = ctx.evidence.data_quality
        quality = {
            "valid_sessions": data_quality.valid_sessions,
            "total_sessions": data_quality.total_sessions,
            "meets_model_minimum": data_quality.meets_model_minimum,
            "issues": list(data_quality.issues),
        }

    hypotheses = _hypothesis_summary(ctx)
    open_questions = []
    if ctx.evaluation is not None:
        open_questions = [
            {
                "key": key,
                "question": question_text(key),
                "answerable": is_answerable(key),
            }
            for key in ctx.evaluation.missing_evidence
        ]

    return {
        "trigger_session_id": ctx.session_id,
        "stop_reason": ctx.stop_reason,
        "data_quality": quality,
        "persistence": _persistence_summary(ctx),
        "hypotheses": hypotheses,
        "supported_hypotheses": [
            item["id"] for item in hypotheses if item["status"] == "supported"
        ],
        "open_questions": open_questions,
        "iterations": ctx.iterations,
        "tool_call_count": len(ctx.tool_calls),
        "evidence_digest": ctx.registry.digest(),
        "disclaimer": DISCLAIMER,
    }


def _build_limitations(ctx):
    limitations = []
    if ctx.evidence is not None:
        limitations.extend(ctx.evidence.limitations)
    else:
        limitations.append(
            "No behavioral evidence could be built, so no signals were analysed."
        )
    limitations.append(ENGINE_LIMITATION)
    if ctx.evaluation is not None and ctx.evaluation.missing_evidence:
        unanswerable = [
            key
            for key in ctx.evaluation.missing_evidence
            if not is_answerable(key)
        ]
        if unanswerable:
            limitations.append(
                "Some questions could not be answered from the available data "
                "sources (" + ", ".join(unanswerable) + ")."
            )
    return limitations


def _next_action(ctx):
    reason = ctx.stop_reason
    if reason == "evidence_insufficient":
        return (
            "Collect more valid sessions so the available history reaches the "
            "model minimum before persistence can be assessed."
        )
    if reason == "data_unavailable":
        return "Restore read access to the stored sessions and investigate again."
    if reason == "iteration_limit" or reason == "tool_budget_exhausted":
        return "Re-run the investigation with a larger collection budget."
    if reason == "only_unanswerable_questions_remain":
        return (
            "No available data source can answer the remaining questions; "
            "user-reported context would be required."
        )
    return "No further evidence can be collected from the available data sources."


def _derive_context(ctx):
    return {
        "engine_schema_version": ENGINE_SCHEMA_VERSION,
        "stop_reason": ctx.stop_reason,
        "iterations": ctx.iterations,
        "nodes_visited": list(ctx.nodes_visited),
        "plan": list(ctx.plan),
        "collected_plan": [
            step for step in ctx.plan if step in ctx.collected_plan
        ],
        "tool_calls": [dict(call) for call in ctx.tool_calls],
        "evidence_digest": ctx.registry.digest(),
        "persistence": _persistence_summary(ctx),
        "hypotheses": _hypothesis_summary(ctx),
        "alternatives": [
            item.model_dump(mode="json") for item in ctx.alternatives
        ],
    }


def _finalize(ctx):
    ctx.trace.record_enter(NODE_FINALIZE)

    state = ctx.state
    if ctx.stop_reason is None:
        ctx.stop_reason = "no_further_evidence"
    state.stop_reason = ctx.stop_reason

    if ctx.evaluation is not None:
        state.hypotheses = ctx.evaluation.hypotheses
        state.missing_evidence = ctx.evaluation.missing_evidence
    elif not state.hypotheses:
        state.hypotheses = create_hypotheses()

    state.investigation_plan = list(ctx.plan)
    state.tool_results = [dict(call) for call in ctx.tool_calls]
    state.final_assessment = _build_final_assessment(ctx)
    state.limitations = _build_limitations(ctx)
    state.next_action = _next_action(ctx)

    state.context.setdefault("derived", {})
    state.context["derived"] = _derive_context(ctx)

    state.evidence = ctx.registry.to_dicts()
    ctx.trace.record_exit(NODE_FINALIZE)
    state.trace = ctx.trace.to_dicts()

    return _DONE


_NODES = {
    NODE_INGEST_SIGNAL: _ingest_signal,
    NODE_CHECK_DATA_QUALITY: _check_data_quality,
    NODE_INITIALIZE_HYPOTHESES: _initialize_hypotheses,
    NODE_INVESTIGATE: _investigate,
    NODE_COLLECT_REQUIRED_EVIDENCE: _collect_required_evidence,
    NODE_ASSESS_PERSISTENCE: _assess_persistence,
    NODE_UPDATE_HYPOTHESES: _update_hypotheses,
    NODE_CHECK_STOP_CONDITIONS: _check_stop_conditions,
    NODE_FINALIZE: _finalize,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_result(ctx):
    state = ctx.state
    return InvestigationResult(
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        stop_reason=state.stop_reason,
        iterations=ctx.iterations,
        tool_calls=[dict(call) for call in ctx.tool_calls],
        evidence_digest=ctx.registry.digest(),
        hypotheses=state.hypotheses,
        missing_evidence=state.missing_evidence,
        alternatives=ctx.alternatives,
        final_assessment=state.final_assessment or {},
        limitations=state.limitations,
        state=state,
    )


def run_investigation(
    user_id,
    session_id,
    *,
    repository=None,
    trace=None,
    clock=None,
    as_of=None,
    max_iterations=MAX_ITERATIONS,
    max_tool_calls=MAX_TOOL_CALLS,
    alt_windows=ALT_WINDOWS,
):
    """Run one deterministic investigation and return its result.

    Args:
        user_id: the user whose read-only history is investigated.
        session_id: the triggering session.
        repository: a read-only ``SessionRepository``. When omitted the
            production (Supabase) repository is used; a failure to construct it
            stops the investigation with ``data_unavailable``.
        trace: an optional ``InvestigationTrace`` (e.g. with a frozen clock).
        clock: optional zero-argument clock used to build a trace.
        as_of: the reference time for windowed analysis (injectable for tests).
        max_iterations, max_tool_calls: deterministic collection bounds.
        alt_windows: comparison-window pairs used for robustness.

    Returns:
        InvestigationResult. Missing or unavailable evidence is represented
        explicitly; nothing is fabricated.
    """
    if not user_id:
        raise ValueError("user_id is required")
    if not session_id:
        raise ValueError("session_id is required")

    active_trace = trace if trace is not None else InvestigationTrace(clock=clock)
    if as_of is None:
        as_of = datetime.now(timezone.utc)

    caching = None
    blocking_reason = None
    if repository is None:
        try:
            caching = CachingRepository(default_repository())
        except RepositoryError:
            caching = None
            blocking_reason = "data_unavailable"
    else:
        caching = CachingRepository(repository)

    ctx = _Context(
        user_id=user_id,
        session_id=str(session_id),
        caching=caching,
        trace=active_trace,
        state=AgentState(),
        registry=EvidenceRegistry(),
        as_of=as_of,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        alt_windows=tuple(alt_windows),
        blocking_reason=blocking_reason,
    )

    step_budget = max_iterations * 4 + 16
    node = NODE_INGEST_SIGNAL
    steps = 0
    while node != _DONE:
        steps += 1
        if steps > step_budget:
            raise RuntimeError("investigation state machine exceeded its step budget")
        ctx.nodes_visited.append(node)
        node = _NODES[node](ctx)

    return _build_result(ctx)
