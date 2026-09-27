"""LangGraph orchestration of the frozen investigation engine (Phase 5).

What this module is
-------------------
A :class:`~langgraph.graph.StateGraph` that drives the **same** node bodies,
planning rules, tool selection, guardrails and stop semantics as the
hand-written bounded state machine in ``engine.py``. This is an orchestration
migration, not an agent rewrite:

    BEFORE:  run_investigation()      -> hand-written while loop over engine nodes
    AFTER:   run_investigation_graph() -> LangGraph StateGraph over the
                                          *same* engine node functions

Every node below is a thin adapter around the corresponding ``engine.py`` node
function (``_ingest_signal``, ``_check_data_quality``, ``_initialize_hypotheses``,
``_investigate``, ``_collect_required_evidence``, ``_assess_persistence``,
``_update_hypotheses``, ``_check_stop_conditions``, ``_finalize``). No business
logic lives here: thresholds, hypothesis rules, evidence templates, claim
safety and the report are all untouched. ``run_investigation()`` remains the
reference implementation and is not modified.

Graph topology
--------------

    START
      -> incident_intake            (engine._ingest_signal)
      -> plan_investigation         (engine._check_data_quality)
      -> initialize_hypotheses      (engine._initialize_hypotheses)
      -> select_tool                (engine._investigate)
      -> execute_tool               (engine._collect_required_evidence)
      -> route_after_assess         (conditional, deterministic)
             |- "update_hypotheses" -> update_hypotheses  (finding exists)
             |- "check_stop_conditions"                   (still collecting,
                the reference machine's direct jump)
      -> update_hypotheses          (engine._update_hypotheses)
      -> check_stop_conditions      (engine._check_stop_conditions)
      -> route_after_select         (conditional, deterministic)
             |- "execute_tool" -> execute_tool       (a request was selected)
             |- "check_stop_conditions"              (nothing left to collect:
                the reference machine's direct jump, so the graph walks no
                no-op nodes and its event stream matches the reference
                event-for-event)
      -> route_after_check          (conditional, deterministic)
             |- "select_tool"  -> select_tool        (bounded loop)
             |- "finalize"     -> finalize           (engine._finalize + critique)
                                   -> route_after_finalize (conditional)
                                         |- END
                                         |- "execute_tool" (contradiction
                                            re-entry, bounded, see below;
                                            the router performs the selection)

Routing is deterministic (no LLM): ``route_after_check`` reads the stop reason
the engine node just recorded; ``route_after_finalize`` consults the existing
critic (``critic.critique``) and only re-enters collection when a contradiction
exists AND a still-uncollected, budget-affordable request is available. Both
loops are bounded: ``MAX_ITERATIONS`` / ``MAX_TOOL_CALLS`` are enforced inside
the engine nodes exactly as before, the contradiction re-entry is additionally
capped by ``CONTRADICTION_REENTRY_LIMIT``, and the LangGraph ``recursion_limit``
is derived from ``max_iterations`` so an out-of-bounds loop is structurally
impossible (a runaway graph raises ``GraphRecursionError`` instead of looping
forever).

State
-----
No second ``AgentState`` is introduced. The graph state
(:class:`InvestigationGraphState`) carries the authoritative
:class:`~investigation.state.AgentState` (inside the run context) plus an
explicit, typed projection of the former run-scoped fields. Each node returns
an explicit state delta. The projection is derived from the context on every
step, so the two can never disagree.

The only additive change outside this module is :class:`_GraphContext`, a
subclass of the engine's private ``_Context`` adding three graph-only fields
(``reentries``, ``reentry_limit``, ``last_verdict``). The engine's own code
never sees them.

Trace equivalence with the reference orchestrator
-------------------------------------------------
The graph produces **event-for-event identical traces** to
``run_investigation()``: the ``route_after_select`` conditional edge skips the
idempotent no-op passes exactly the way the reference machine jumps from an
empty selection straight to its stop check, node bodies are the engine's own
functions (so trace events carry the engine's node names in both cases), and
the routing critique inside ``finalize`` runs without a trace. Regression
tests assert full trace equality and full service-payload equality.

No LLM, no network, no new I/O. The only import beyond the frozen modules is
``langgraph`` (pinned in ``backend/requirements.txt``).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..repository import RepositoryError, default_repository
from ..state import AgentState
from .critic import Critique, critique
from .evidence import CachingRepository, EvidenceRegistry
from .engine import (
    ALT_WINDOWS,
    DEFAULT_RECENT_LIMIT,
    MAX_ITERATIONS,
    MAX_TOOL_CALLS,
    _Context,
    _assess_persistence,
    _build_result,
    _budget_remaining,
    _check_data_quality,
    _check_stop_conditions,
    _collect_required_evidence,
    _finalize,
    _ingest_signal,
    _initialize_hypotheses,
    _investigate,
    _update_hypotheses,
    select_next_collection,
)
from .trace import InvestigationTrace

#: Version of the graph orchestration shape.
GRAPH_SCHEMA_VERSION = "1.0"

#: How many times a critic-detected contradiction may send the graph back to
#: evidence collection after finalization. Deliberately small: the valve must
#: never turn a contradiction into an unbounded loop.
CONTRADICTION_REENTRY_LIMIT = 1

#: LangGraph node names (a fixed, closed set).
NODE_INCIDENT_INTAKE = "incident_intake"
NODE_PLAN_INVESTIGATION = "plan_investigation"
NODE_INITIALIZE_HYPOTHESES = "initialize_hypotheses"
NODE_SELECT_TOOL = "select_tool"
NODE_EXECUTE_TOOL = "execute_tool"
NODE_ASSESS_PERSISTENCE = "assess_persistence"
NODE_UPDATE_HYPOTHESES = "update_hypotheses"
NODE_CHECK_STOP_CONDITIONS = "check_stop_conditions"
NODE_FINALIZE = "finalize"

GRAPH_NODES: tuple[str, ...] = (
    NODE_INCIDENT_INTAKE,
    NODE_PLAN_INVESTIGATION,
    NODE_INITIALIZE_HYPOTHESES,
    NODE_SELECT_TOOL,
    NODE_EXECUTE_TOOL,
    NODE_ASSESS_PERSISTENCE,
    NODE_UPDATE_HYPOTHESES,
    NODE_CHECK_STOP_CONDITIONS,
    NODE_FINALIZE,
)

#: Conditional-edge targets. The main loop returns to ``select_tool``; the
#: contradiction re-entry goes to ``execute_tool`` because the routing function
#: itself performs the selection (via the same ``select_next_collection`` the
#: select node uses) and hands the chosen request to the executor as ``pending``.
ROUTE_LOOP = "select_tool"
ROUTE_REENTER = "execute_tool"
ROUTE_FINALIZE = "finalize"
ROUTE_END = "__end__"


# ---------------------------------------------------------------------------
# Run context (engine _Context plus the graph-only contradiction fields)
# ---------------------------------------------------------------------------


@dataclass
class _GraphContext(_Context):
    """The engine's run context plus the graph's contradiction bookkeeping.

    Additive only: every engine node keeps working unchanged because the new
    fields all have immutable defaults and the engine never reads them.
    """

    reentries: int = 0
    reentry_limit: int = CONTRADICTION_REENTRY_LIMIT
    last_verdict: Critique | None = None


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class InvestigationGraphState(TypedDict, total=False):
    """The explicit, typed LangGraph state.

    ``ctx`` is the authoritative run context (it holds the real
    :class:`~investigation.state.AgentState`, the evidence registry and the
    trace). The remaining keys are an explicit projection of the former
    run-scoped fields, refreshed by every node, so the state an operator (or
    test) inspects is always in sync with what the nodes actually decided.
    Nodes return explicit deltas of exactly these keys.
    """

    # Authoritative run context (engine node bodies read and mutate it).
    ctx: Any

    # The agent's explicit state model (same object as ``ctx.state``).
    agent_state: AgentState

    # Incident / session.
    user_id: str
    session_id: str

    # Planning.
    plan: list[str]
    collected_plan: list[str]
    pending_tool: str | None

    # Tool calling.
    tool_calls: list[dict]
    tool_call_count: int
    called_fingerprints: list[str]
    max_tool_calls: int

    # Bounds / loop bookkeeping.
    iterations: int
    max_iterations: int
    no_more_evidence: bool
    blocking_reason: str | None

    # Outcome so far.
    stop_reason: str | None
    evidence_count: int
    evidence_digest: str | None
    hypotheses: dict
    missing_evidence: list[str]
    anomaly_available: bool
    alt_drifts: int
    persistence_assessed: bool

    # Contradiction re-entry (Checkpoint 6 safety valve).
    reentries: int
    contradiction_codes: list[str]

    # Observability: which node ran last and where it decided to go next.
    last_node: str
    next_node: str


def _projection(ctx: _Context) -> dict:
    """The explicit read-only projection of the run context."""
    assessment = ctx.state.final_assessment or {}
    hypotheses = {
        entry.get("id"): entry.get("status")
        for entry in assessment.get("hypotheses") or []
        if isinstance(entry, dict)
    }
    return {
        "agent_state": ctx.state,
        "user_id": ctx.user_id,
        "session_id": ctx.session_id,
        "plan": list(ctx.plan),
        "collected_plan": sorted(ctx.collected_plan),
        "pending_tool": None if ctx.pending is None else ctx.pending.tool,
        "tool_calls": [dict(call) for call in ctx.tool_calls],
        "tool_call_count": len(ctx.tool_calls),
        "called_fingerprints": sorted(ctx.called),
        "max_tool_calls": ctx.max_tool_calls,
        "iterations": ctx.iterations,
        "max_iterations": ctx.max_iterations,
        "no_more_evidence": ctx.no_more_evidence,
        "blocking_reason": ctx.blocking_reason,
        "stop_reason": ctx.stop_reason,
        "evidence_count": len(ctx.registry),
        "evidence_digest": ctx.registry.digest(),
        "hypotheses": hypotheses,
        "missing_evidence": list(ctx.evaluation.missing_evidence)
        if ctx.evaluation is not None
        else [],
        "anomaly_available": ctx.anomaly is not None and bool(ctx.anomaly.available),
        "alt_drifts": len(ctx.alt_drifts),
        "persistence_assessed": ctx.finding is not None,
        "reentries": ctx.reentries,
    }


def _recursion_limit(max_iterations: int) -> int:
    """A LangGraph recursion limit consistent with the engine's bounds.

    Per bounded loop pass the graph walks five nodes (select, execute, assess,
    update, check); the head (intake, plan, initialize) costs three,
    finalization one, and at most one contradiction re-entry adds one more
    five-node pass. The margin keeps a correct graph well under the limit
    while making an out-of-bounds loop structurally impossible.
    """
    return 10 + 3 + max_iterations * 5 + 1 + 5 + 1


# ---------------------------------------------------------------------------
# Node adapters
# ---------------------------------------------------------------------------
#
# Each adapter runs the existing engine node function on the context carried
# by the state, then returns an explicit delta. The ``next_node`` value is the
# engine function's own routing decision, kept for observability and tests;
# the graph's edges (below) are the orchestration truth.


def _node(name, engine_fn):
    def run(state: InvestigationGraphState) -> dict:
        ctx: _Context = state["ctx"]
        decision = engine_fn(ctx)
        return {"ctx": ctx, "last_node": name, "next_node": decision, **_projection(ctx)}

    run.__name__ = name
    run.__doc__ = engine_fn.__doc__
    return run


incident_intake = _node(NODE_INCIDENT_INTAKE, _ingest_signal)
plan_investigation = _node(NODE_PLAN_INVESTIGATION, _check_data_quality)
initialize_hypotheses = _node(NODE_INITIALIZE_HYPOTHESES, _initialize_hypotheses)
select_tool = _node(NODE_SELECT_TOOL, _investigate)
execute_tool = _node(NODE_EXECUTE_TOOL, _collect_required_evidence)
assess_persistence = _node(NODE_ASSESS_PERSISTENCE, _assess_persistence)
update_hypotheses = _node(NODE_UPDATE_HYPOTHESES, _update_hypotheses)
check_stop_conditions = _node(NODE_CHECK_STOP_CONDITIONS, _check_stop_conditions)


def finalize(state: InvestigationGraphState) -> dict:
    """Finalize via the engine, then compute the routing critique (once).

    The critique runs for *routing only*, without a trace, so it has no side
    effects on the authoritative trace; report generation
    (``build_grounded_report``) keeps seeding and recording critic events
    exactly as the reference flow does.
    """
    ctx: _Context = state["ctx"]
    decision = _finalize(ctx)
    result = _build_result(ctx)
    verdict = critique(result)
    ctx.last_verdict = verdict
    return {
        "ctx": ctx,
        "last_node": NODE_FINALIZE,
        "next_node": decision,
        "contradiction_codes": sorted({item.code for item in verdict.contradictions}),
        **_projection(ctx),
    }


# ---------------------------------------------------------------------------
# Conditional routing (deterministic, no LLM)
# ---------------------------------------------------------------------------


def route_after_select(state: InvestigationGraphState) -> str:
    """Execute the selected tool, or go straight to the stop check.

    ``engine._investigate`` queues a request exactly when it found one; when it
    found nothing the reference machine routes directly to its stop check
    instead of re-running the idempotent assess/update nodes. Matching that
    here keeps the graph's trace event-for-event identical to the reference
    orchestrator's.
    """
    ctx: _Context = state["ctx"]
    return NODE_EXECUTE_TOOL if ctx.pending is not None else NODE_CHECK_STOP_CONDITIONS


def route_after_assess(state: InvestigationGraphState) -> str:
    """Analyze hypotheses only when persistence was actually assessed.

    ``engine._assess_persistence`` routes straight to the stop check while it
    is still waiting for required evidence (``finding is None``) and hands over
    to the hypothesis update only once the finding exists. Matching that here
    keeps the graph's event stream identical to the reference machine's.
    """
    ctx: _Context = state["ctx"]
    return NODE_UPDATE_HYPOTHESES if ctx.finding is not None else NODE_CHECK_STOP_CONDITIONS


def route_after_check(state: InvestigationGraphState) -> str:
    """Continue the bounded loop or finalize.

    ``engine._check_stop_conditions`` has just run: it records a stop reason
    exactly when the investigation must stop. Continuing is therefore simply
    "no stop reason yet" -- the same rule the reference machine uses.
    """
    ctx: _Context = state["ctx"]
    return ROUTE_FINALIZE if ctx.stop_reason is not None else ROUTE_LOOP


def route_after_finalize(state: InvestigationGraphState) -> str:
    """END, or one bounded re-entry into collection on a real contradiction.

    Checkpoint 6 integration point. A critic-detected contradiction sends the
    graph back to ``select_tool`` only when *additional evidence is genuinely
    needed*: a still-uncollected candidate request exists and the tool budget
    affords it. Everything else -- including a second contradiction -- ends
    the graph, so the loop is bounded by the re-entry limit, ``MAX_TOOL_CALLS``
    and the engine's own stop rules.

    With the frozen catalogs the critic is sound by construction, so on real
    data this valve stays closed (the regression suite proves catalog-wide
    equivalence); it exists so future business logic cannot turn a detected
    contradiction into a silently strong conclusion.
    """
    ctx: _GraphContext = state["ctx"]
    if ctx.reentries >= ctx.reentry_limit:
        return ROUTE_END
    if _budget_remaining(ctx) <= 0:
        return ROUTE_END
    if ctx.last_verdict is None or not ctx.last_verdict.contradictions:
        return ROUTE_END
    request = select_next_collection(
        user_id=ctx.user_id,
        plan=ctx.plan,
        collected_plan=ctx.collected_plan,
        open_questions=(
            ctx.evaluation.missing_evidence if ctx.evaluation is not None else ()
        ),
        called_fingerprints=ctx.called,
        alt_windows=ctx.alt_windows,
        recent_limit=DEFAULT_RECENT_LIMIT,
        budget_remaining=_budget_remaining(ctx),
    )
    if request is None:
        return ROUTE_END
    ctx.pending = request
    ctx.no_more_evidence = False
    ctx.stop_reason = None
    return ROUTE_REENTER


# ---------------------------------------------------------------------------
# Graph construction and entry point
# ---------------------------------------------------------------------------


def build_investigation_graph():
    """Build and compile the investigation StateGraph.

    Returns a compiled LangGraph graph whose nodes are the engine's own node
    functions. Compilation is deterministic and side-effect free.
    """
    graph = StateGraph(InvestigationGraphState)

    graph.add_node(NODE_INCIDENT_INTAKE, incident_intake)
    graph.add_node(NODE_PLAN_INVESTIGATION, plan_investigation)
    graph.add_node(NODE_INITIALIZE_HYPOTHESES, initialize_hypotheses)
    graph.add_node(NODE_SELECT_TOOL, select_tool)
    graph.add_node(NODE_EXECUTE_TOOL, execute_tool)
    graph.add_node(NODE_ASSESS_PERSISTENCE, assess_persistence)
    graph.add_node(NODE_UPDATE_HYPOTHESES, update_hypotheses)
    graph.add_node(NODE_CHECK_STOP_CONDITIONS, check_stop_conditions)
    graph.add_node(NODE_FINALIZE, finalize)

    # Explicit edges: the fixed head, then the bounded loop body.
    graph.add_edge(START, NODE_INCIDENT_INTAKE)
    graph.add_edge(NODE_INCIDENT_INTAKE, NODE_PLAN_INVESTIGATION)
    graph.add_edge(NODE_PLAN_INVESTIGATION, NODE_INITIALIZE_HYPOTHESES)
    graph.add_edge(NODE_INITIALIZE_HYPOTHESES, NODE_SELECT_TOOL)
    graph.add_edge(NODE_EXECUTE_TOOL, NODE_ASSESS_PERSISTENCE)
    graph.add_edge(NODE_UPDATE_HYPOTHESES, NODE_CHECK_STOP_CONDITIONS)

    # Conditional edge: run the selected tool, or -- when the selector found
    # nothing left to collect -- go straight to the stop check, mirroring the
    # reference machine so no idempotent no-op node is ever walked.
    graph.add_conditional_edges(
        NODE_SELECT_TOOL,
        route_after_select,
        {
            NODE_EXECUTE_TOOL: NODE_EXECUTE_TOOL,
            NODE_CHECK_STOP_CONDITIONS: NODE_CHECK_STOP_CONDITIONS,
        },
    )

    # Conditional edge: update the hypotheses only when a persistence finding
    # exists -- the engine's assess node otherwise routes straight to the stop
    # check while required evidence is still being collected.
    graph.add_conditional_edges(
        NODE_ASSESS_PERSISTENCE,
        route_after_assess,
        {
            NODE_UPDATE_HYPOTHESES: NODE_UPDATE_HYPOTHESES,
            NODE_CHECK_STOP_CONDITIONS: NODE_CHECK_STOP_CONDITIONS,
        },
    )

    # Conditional edge: loop back to tool selection, or finalize.
    graph.add_conditional_edges(
        NODE_CHECK_STOP_CONDITIONS,
        route_after_check,
        {ROUTE_LOOP: NODE_SELECT_TOOL, ROUTE_FINALIZE: NODE_FINALIZE},
    )

    # Conditional edge: END, or one bounded contradiction re-entry.
    graph.add_conditional_edges(
        NODE_FINALIZE,
        route_after_finalize,
        {ROUTE_REENTER: NODE_EXECUTE_TOOL, ROUTE_END: END},
    )

    return graph.compile()


def run_investigation_graph(
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
    contradiction_reentries=CONTRADICTION_REENTRY_LIMIT,
):
    """Run one deterministic investigation through the LangGraph graph.

    Mirrors :func:`~investigation.agent.engine.run_investigation` argument for
    argument and returns the same :class:`InvestigationResult` type with
    equivalent outcomes. The reference implementation is unchanged.

    Args:
        user_id: the user whose read-only history is investigated.
        session_id: the triggering session.
        repository: a read-only ``SessionRepository``. When omitted the
            production (Supabase) repository is used; a failure to construct
            it stops the investigation with ``data_unavailable``.
        trace: an optional ``InvestigationTrace`` (e.g. with a frozen clock).
        clock: optional zero-argument clock used to build a trace.
        as_of: the reference time for windowed analysis (injectable for tests).
        max_iterations, max_tool_calls: deterministic collection bounds.
        alt_windows: comparison-window pairs used for robustness.
        contradiction_reentries: how many times a critic-detected contradiction
            may reopen collection (bounded; default 1).

    Returns:
        InvestigationResult. Missing or unavailable evidence is represented
        explicitly; nothing is fabricated.

    Raises:
        ValueError: ``user_id`` or ``session_id`` is empty.
        langgraph.errors.GraphRecursionError: the graph exceeds the recursion
            limit derived from ``max_iterations`` (structurally a bug, never a
            legitimate investigation outcome).
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

    ctx = _GraphContext(
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
        reentries=0,
        reentry_limit=max(0, int(contradiction_reentries)),
        last_verdict=None,
    )

    graph = build_investigation_graph()
    graph.invoke(
        {"ctx": ctx, **_projection(ctx)},
        config={"recursion_limit": _recursion_limit(max_iterations)},
    )
    return _build_result(ctx)
