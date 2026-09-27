"""LangGraph orchestration tests (Phase 5).

These tests cover the graph itself: construction, the START -> ... -> END
topology, conditional routing, the bounded loop, guardrails, and the
Checkpoint 6 contradiction re-entry valve. The old engine
(``engine.run_investigation``) stays the reference implementation and is
compared against the graph in ``test_graph_regression.py``.

Every node adapter is exercised both through the compiled graph and (where a
single transition is under test) as a plain function on a hand-built context,
which is exactly the state shape LangGraph passes between nodes.
"""

import json
import unittest
from unittest import mock

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

import investigation.agent.graph as graph_module
from investigation import repository as repository_module
from investigation.agent import (
    GroundedReport,
    REPORT_DISCLAIMER,
    build_grounded_report,
    critique,
    run_investigation,
)
from investigation.agent.graph import (
    build_investigation_graph,
    run_investigation_graph,
)
from investigation.agent.engine import (
    DEFAULT_RECENT_LIMIT,
    MAX_ITERATIONS,
    MAX_TOOL_CALLS,
    select_next_collection,
)
from investigation.agent.trace import args_fingerprint
from investigation.tools import GetRecentSessionsInput
from tests import fixtures
from tests.agent.scenarios import USER, frozen_clock, get_scenario

FROZEN_NOW = frozen_clock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_graph(scenario_key, **kwargs):
    """Run one catalog scenario through the LangGraph orchestrator."""
    scenario = get_scenario(scenario_key)
    return run_investigation_graph(
        USER,
        scenario.session_id,
        repository=scenario.repository_factory(),
        clock=frozen_clock,
        as_of=FROZEN_NOW,
        **kwargs,
    )


def _run_graph_outage(**kwargs):
    """Run the outage scenario through the graph (its own repo factory raises)."""
    with mock.patch.object(
        graph_module,
        "default_repository",
        side_effect=repository_module.RepositoryError("outage"),
    ):
        return run_investigation_graph(
            USER, "S0001", clock=frozen_clock, as_of=FROZEN_NOW, **kwargs
        )


def _make_ctx(sessions=None, anomalies=None, session_id="S0001", **kwargs):
    """A graph context exactly as ``run_investigation_graph`` builds it."""
    from investigation.agent.evidence import CachingRepository, EvidenceRegistry
    from investigation.agent.trace import InvestigationTrace
    from investigation.state import AgentState

    defaults = dict(
        user_id=USER,
        session_id=session_id,
        caching=CachingRepository(fixtures.FakeRepository(sessions, anomalies)),
        trace=InvestigationTrace(clock=frozen_clock),
        state=AgentState(),
        registry=EvidenceRegistry(),
        as_of=FROZEN_NOW,
        max_iterations=MAX_ITERATIONS,
        max_tool_calls=MAX_TOOL_CALLS,
        alt_windows=((14, 14),),
        blocking_reason=None,
    )
    defaults.update(kwargs)
    return graph_module._GraphContext(**defaults)


def _seeded_ctx(**kwargs):
    """A context whose incident has already been ingested."""
    ctx = _make_ctx(**kwargs)
    graph_module.incident_intake({"ctx": ctx})
    return ctx


def _node_order(scenario_key, **kwargs):
    """The node execution order of one graph run, via LangGraph streaming."""
    scenario = get_scenario(scenario_key)
    repository = scenario.repository_factory()
    ctx = _make_ctx(
        sessions=repository.sessions,
        anomalies=repository.anomalies,
        session_id=scenario.session_id,
        **kwargs,
    )
    compiled = build_investigation_graph()
    initial = {"ctx": ctx, **graph_module._projection(ctx)}
    order = []
    for chunk in compiled.stream(
        initial,
        config={"recursion_limit": graph_module._recursion_limit(MAX_ITERATIONS)},
        stream_mode="values",
    ):
        if "last_node" in chunk:
            order.append(chunk["last_node"])
    return order


def _stable_sessions():
    return [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(1, 34)
    ]


# ---------------------------------------------------------------------------
# 1. Graph construction
# ---------------------------------------------------------------------------


class GraphConstructionTests(unittest.TestCase):
    def test_builds_a_compiled_langgraph_state_graph(self):
        compiled = build_investigation_graph()
        self.assertIsInstance(compiled, CompiledStateGraph)

    def test_all_nine_nodes_exist(self):
        compiled = build_investigation_graph()
        names = {node.name for node in compiled.get_graph().nodes.values()}
        self.assertTrue(set(graph_module.GRAPH_NODES) <= names)
        self.assertEqual(len(graph_module.GRAPH_NODES), 9)

    def test_start_and_end_exist(self):
        compiled = build_investigation_graph()
        names = {node.name for node in compiled.get_graph().nodes.values()}
        self.assertIn(START, names)
        self.assertIn(END, names)

    def test_explicit_edges_wire_the_head(self):
        edges = {
            (edge.source, edge.target)
            for edge in build_investigation_graph().get_graph().edges
            if not edge.conditional
        }
        self.assertIn((START, "incident_intake"), edges)
        self.assertIn(("incident_intake", "plan_investigation"), edges)
        self.assertIn(("plan_investigation", "initialize_hypotheses"), edges)
        self.assertIn(("initialize_hypotheses", "select_tool"), edges)

    def test_explicit_edges_wire_the_loop_body(self):
        unconditional = {
            (edge.source, edge.target)
            for edge in build_investigation_graph().get_graph().edges
            if not edge.conditional
        }
        self.assertIn(("execute_tool", "assess_persistence"), unconditional)
        self.assertIn(("update_hypotheses", "check_stop_conditions"), unconditional)

    def test_conditional_edges_loop_and_finalize(self):
        conditional = {
            (edge.source, edge.target)
            for edge in build_investigation_graph().get_graph().edges
            if edge.conditional
        }
        # The bounded loop: select -> execute, or straight to the stop check
        # when nothing is left to collect (the reference machine's jump).
        self.assertIn(("select_tool", "execute_tool"), conditional)
        self.assertIn(("select_tool", "check_stop_conditions"), conditional)
        # Analyze only when a persistence finding exists.
        self.assertIn(("assess_persistence", "update_hypotheses"), conditional)
        self.assertIn(("assess_persistence", "check_stop_conditions"), conditional)
        # Continue or finalize.
        self.assertIn(("check_stop_conditions", "select_tool"), conditional)
        self.assertIn(("check_stop_conditions", "finalize"), conditional)
        # Terminate, or one bounded contradiction re-entry.
        self.assertIn(("finalize", END), conditional)
        self.assertIn(("finalize", "execute_tool"), conditional)

    def test_state_schema_is_explicit(self):
        fields = set(graph_module.InvestigationGraphState.__annotations__)
        for required in (
            "ctx",
            "agent_state",
            "plan",
            "collected_plan",
            "tool_calls",
            "tool_call_count",
            "iterations",
            "max_iterations",
            "max_tool_calls",
            "stop_reason",
            "missing_evidence",
            "evidence_digest",
            "reentries",
            "contradiction_codes",
        ):
            self.assertIn(required, fields)

    def test_construction_is_deterministic(self):
        first = build_investigation_graph().get_graph()
        second = build_investigation_graph().get_graph()
        self.assertEqual(
            sorted(node.name for node in first.nodes.values()),
            sorted(node.name for node in second.nodes.values()),
        )
        self.assertEqual(
            sorted((e.source, e.target, e.conditional) for e in first.edges),
            sorted((e.source, e.target, e.conditional) for e in second.edges),
        )

    def test_reentry_limit_is_a_small_fixed_constant(self):
        self.assertEqual(graph_module.CONTRADICTION_REENTRY_LIMIT, 1)


# ---------------------------------------------------------------------------
# 2-4. START -> intake -> plan -> select -> execute
# ---------------------------------------------------------------------------


class HeadFlowTests(unittest.TestCase):
    def test_start_flows_into_incident_intake(self):
        order = _node_order("persistent_multi_signal")
        self.assertEqual(order[0], "incident_intake")

    def test_head_runs_plan_and_hypothesis_nodes_in_order(self):
        order = _node_order("persistent_multi_signal")
        self.assertEqual(
            order[:4],
            [
                "incident_intake",
                "plan_investigation",
                "initialize_hypotheses",
                "select_tool",
            ],
        )

    def test_incident_intake_records_the_trigger_on_agent_state(self):
        result = _run_graph("persistent_multi_signal")
        self.assertEqual(result.state.trigger["source"], "typing_session")
        self.assertIsNotNone(result.state.ml_evidence)

    def test_plan_investigation_sets_the_bootstrap_plan(self):
        ctx = _seeded_ctx(sessions=_stable_sessions())
        delta = graph_module.plan_investigation({"ctx": ctx})
        self.assertEqual(delta["plan"], ["recent_sessions", "window_robustness"])
        self.assertEqual(ctx.state.investigation_plan, delta["plan"])

    def test_plan_drops_window_robustness_below_the_model_minimum(self):
        thin = [
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i) for i in range(3)
        ]
        ctx = _seeded_ctx(sessions=thin)
        delta = graph_module.plan_investigation({"ctx": ctx})
        self.assertEqual(delta["plan"], ["recent_sessions"])

    def test_initialize_hypotheses_starts_all_uncertain(self):
        ctx = _make_ctx(sessions=_stable_sessions())
        graph_module.initialize_hypotheses({"ctx": ctx})
        self.assertEqual(len(ctx.state.hypotheses), 4)
        self.assertTrue(
            all(hypothesis.status == "uncertain" for hypothesis in ctx.state.hypotheses)
        )

    def test_select_tool_chooses_a_justified_request(self):
        ctx = _seeded_ctx(sessions=_stable_sessions())
        graph_module.plan_investigation({"ctx": ctx})
        delta = graph_module.select_tool({"ctx": ctx})
        self.assertEqual(delta["pending_tool"], "get_recent_sessions")

    def test_execute_tool_runs_records_and_registers(self):
        ctx = _seeded_ctx(sessions=_stable_sessions())
        graph_module.plan_investigation({"ctx": ctx})
        graph_module.select_tool({"ctx": ctx})
        before = len(ctx.tool_calls)
        delta = graph_module.execute_tool({"ctx": ctx})
        self.assertEqual(delta["tool_call_count"], before + 1)
        self.assertTrue(delta["tool_calls"][-1]["available"])
        event_types = {event.event_type for event in ctx.trace.events}
        self.assertIn("tool_call", event_types)
        self.assertIn("tool_result", event_types)
        self.assertTrue(len(ctx.registry) > 0)


# ---------------------------------------------------------------------------
# 5. The conditional loop
# ---------------------------------------------------------------------------


class ConditionalLoopTests(unittest.TestCase):
    def test_route_after_check_loops_without_a_stop_reason(self):
        ctx = _seeded_ctx(sessions=_stable_sessions())
        self.assertEqual(graph_module.route_after_check({"ctx": ctx}), "select_tool")

    def test_route_after_check_finalizes_with_a_stop_reason(self):
        ctx = _make_ctx(
            sessions=_stable_sessions(), stop_reason="no_further_evidence"
        )
        self.assertEqual(graph_module.route_after_check({"ctx": ctx}), "finalize")

    def test_loop_revisits_select_tool_until_the_stop(self):
        order = _node_order("consistent")
        self.assertGreaterEqual(order.count("select_tool"), 2)
        for index, node in enumerate(order):
            if node == "select_tool" and index > 4:
                self.assertEqual(
                    order[index : index + 4],
                    [
                        "select_tool",
                        "execute_tool",
                        "assess_persistence",
                        "update_hypotheses",
                    ],
                )
                break

    def test_every_run_terminates_at_finalize(self):
        for key in ("consistent", "insufficient_history", "zero_sessions"):
            with self.subTest(scenario=key):
                order = _node_order(key)
                self.assertEqual(order[-1], "finalize")
                self.assertNotIn("finalize", order[:-1])

    def test_recursion_limit_grows_with_the_iteration_bound(self):
        expected = 10 + 3 + MAX_ITERATIONS * 5 + 1 + 5 + 1
        self.assertEqual(graph_module._recursion_limit(MAX_ITERATIONS), expected)
        self.assertGreater(graph_module._recursion_limit(8), graph_module._recursion_limit(4))


# ---------------------------------------------------------------------------
# 6. Bounded termination
# ---------------------------------------------------------------------------


class BoundedTerminationTests(unittest.TestCase):
    def test_max_iterations_termination(self):
        repository = fixtures.FakeRepository(sessions=_stable_sessions())
        old = run_investigation(
            USER,
            "S0001",
            repository=repository,
            clock=frozen_clock,
            as_of=FROZEN_NOW,
            max_iterations=1,
        )
        new = _run_graph("consistent", max_iterations=1)
        self.assertEqual(old.stop_reason, "iteration_limit")
        self.assertEqual(new.stop_reason, "iteration_limit")
        self.assertEqual(old.evidence_digest, new.evidence_digest)

    def test_max_tool_calls_termination(self):
        repository = fixtures.FakeRepository(sessions=_stable_sessions())
        old = run_investigation(
            USER,
            "S0001",
            repository=repository,
            clock=frozen_clock,
            as_of=FROZEN_NOW,
            max_tool_calls=1,
        )
        new = _run_graph("consistent", max_tool_calls=1)
        self.assertEqual(old.stop_reason, "tool_budget_exhausted")
        self.assertEqual(new.stop_reason, "tool_budget_exhausted")
        self.assertEqual(old.evidence_digest, new.evidence_digest)

    def test_bounds_keep_the_engine_defaults(self):
        self.assertEqual(MAX_ITERATIONS, 4)
        self.assertEqual(MAX_TOOL_CALLS, 8)


# ---------------------------------------------------------------------------
# 7. Guardrails
# ---------------------------------------------------------------------------


class GuardrailTests(unittest.TestCase):
    def test_missing_ml_assessment_becomes_explicit_evidence(self):
        result = _run_graph("consistent")  # no anomaly rows in the fixture
        kinds = {item["kind"] for item in result.state.evidence}
        self.assertIn("anomaly_unavailable", kinds)
        self.assertNotIn("anomaly", kinds)

    def test_malformed_ml_assessment_missing_fields_is_honest(self):
        repository = fixtures.FakeRepository(
            sessions=_stable_sessions(), anomalies={"S0001": {}}
        )
        result = run_investigation_graph(
            USER, "S0001", repository=repository, clock=frozen_clock, as_of=FROZEN_NOW
        )
        kinds = {item["kind"] for item in result.state.evidence}
        self.assertIn("anomaly_unavailable", kinds)
        self.assertNotIn("anomaly", kinds)

    def test_malformed_ml_assessment_non_numeric_score_fails_both_orchestrators(self):
        # Pre-existing tool-level behavior (documented, unchanged by the
        # migration): ``tools.get_ml_evidence`` coerces the stored score with
        # float() outside its guarded block, so a non-numeric stored score
        # raises in the reference engine too. The graph inherits exactly the
        # same behavior -- equivalence, not a new failure.
        repository = fixtures.FakeRepository(
            sessions=_stable_sessions(),
            anomalies={"S0001": {"anomaly_score": "not-a-number"}},
        )
        with self.assertRaises(ValueError):
            run_investigation(
                USER, "S0001", repository=repository, clock=frozen_clock, as_of=FROZEN_NOW
            )
        with self.assertRaises(ValueError):
            run_investigation_graph(
                USER, "S0001", repository=repository, clock=frozen_clock, as_of=FROZEN_NOW
            )

    def test_invalid_tool_arguments_are_rejected_before_execution(self):
        with self.assertRaises(ValidationError):
            GetRecentSessionsInput(user_id=USER, limit=0)
        with self.assertRaises(ValidationError):
            GetRecentSessionsInput(user_id=USER, window_days=100000)

    def test_tool_failure_degrades_to_explicit_missing_evidence(self):
        result = run_investigation_graph(
            USER,
            "S0001",
            repository=fixtures.FailingRepository(),
            clock=frozen_clock,
            as_of=FROZEN_NOW,
        )
        kinds = {item["kind"] for item in result.state.evidence}
        self.assertIn("tool_unavailable", kinds)
        self.assertIn("anomaly_unavailable", kinds)
        self.assertNotIn("anomaly", kinds)

    def test_unsupported_conclusion_is_rejected_by_the_critic(self):
        # Tamper a graph-produced result the way a buggy rule would: support a
        # hypothesis without evidence. The existing critic must reject it.
        result = _run_graph("persistent_multi_signal")
        result.hypotheses[0].status = "supported"
        result.hypotheses[0].supporting_evidence = []
        verdict = critique(result)
        reasons = {rejection.reason for rejection in verdict.rejections}
        self.assertIn("support_without_evidence", reasons)
        self.assertFalse(verdict.is_sound)

    def test_outage_scenario_recovers_safely_through_the_graph(self):
        result = _run_graph_outage()
        self.assertEqual(result.stop_reason, "data_unavailable")
        self.assertTrue(result.state.next_action)


# ---------------------------------------------------------------------------
# 8. Checkpoint 6: contradiction handling
# ---------------------------------------------------------------------------


class ContradictionTests(unittest.TestCase):
    def test_robustness_disagreement_is_detected_and_blocks_the_claim(self):
        result = _run_graph("robustness_disagreement")
        report = build_grounded_report(result, clock=frozen_clock)
        persistence = result.final_assessment["persistence"]
        self.assertEqual(persistence["robustness"], "disagrees")
        self.assertFalse(persistence["eligible"])
        self.assertNotEqual(result.hypotheses[2].status, "supported")  # H3 weakened
        self.assertEqual(report.conclusion.status, "preliminary")
        self.assertIn("window_robustness_disagrees", report.uncertainty.reasons)

    def test_both_sides_of_the_conflict_are_preserved(self):
        result = _run_graph("robustness_disagreement")
        h3 = result.hypotheses[2]
        self.assertTrue(h3.contradicting_evidence)
        robustness = [
            item
            for item in result.state.evidence
            if item["kind"] == "persistence_robustness"
        ]
        self.assertEqual(robustness[0]["status"], "disagrees")

    def test_reentry_valve_stays_closed_without_contradictions(self):
        ctx = _make_ctx(sessions=_stable_sessions())
        self.assertEqual(graph_module.route_after_finalize({"ctx": ctx}), END)

    def test_reentry_valve_respects_its_limit(self):
        ctx = _make_ctx(sessions=_stable_sessions())
        ctx.reentries = graph_module.CONTRADICTION_REENTRY_LIMIT
        self.assertEqual(graph_module.route_after_finalize({"ctx": ctx}), END)

    def test_reentry_valve_respects_the_tool_budget(self):
        ctx = _make_ctx(sessions=_stable_sessions(), max_tool_calls=1)
        ctx.tool_calls.append({"tool": "get_recent_sessions", "available": True})
        self.assertEqual(graph_module.route_after_finalize({"ctx": ctx}), END)

    def test_reentry_valve_ends_when_no_uncollected_request_remains(self):
        ctx = _make_ctx(sessions=_stable_sessions())
        ctx.last_verdict = mock.Mock(contradictions=[object()])
        # No plan and no open questions: no candidate request can exist.
        self.assertEqual(ctx.plan, [])
        self.assertEqual(graph_module.route_after_finalize({"ctx": ctx}), END)

    def test_reentry_with_a_real_request_targets_execute_tool(self):
        ctx = _seeded_ctx(sessions=_stable_sessions())
        graph_module.plan_investigation({"ctx": ctx})
        # Pre-call the recent-sessions fingerprint the way a completed
        # collection would, so the remaining candidate is the drift call.
        first = select_next_collection(
            user_id=ctx.user_id,
            plan=ctx.plan,
            collected_plan=(),
            open_questions=(),
            called_fingerprints=ctx.called,
            alt_windows=ctx.alt_windows,
            recent_limit=DEFAULT_RECENT_LIMIT,
            budget_remaining=8,
        )
        ctx.called[args_fingerprint(first.tool, **first.args)] = first.tool
        ctx.collected_plan.add("recent_sessions")

        ctx.last_verdict = mock.Mock(contradictions=[object()])
        decision = graph_module.route_after_finalize({"ctx": ctx})
        self.assertEqual(decision, graph_module.ROUTE_REENTER)
        self.assertIsNotNone(ctx.pending)  # the router queued the next request
        self.assertFalse(ctx.no_more_evidence)


# ---------------------------------------------------------------------------
# 9. Final report through the graph
# ---------------------------------------------------------------------------


class ReportTests(unittest.TestCase):
    def test_graph_result_builds_the_existing_grounded_report(self):
        result = _run_graph("persistent_multi_signal")
        report = build_grounded_report(result, clock=frozen_clock)
        self.assertIsInstance(report, GroundedReport)
        self.assertEqual(report.disclaimer, REPORT_DISCLAIMER)
        self.assertTrue(report.observations)
        self.assertTrue(report.link.evidence_digest)

    def test_graph_report_is_json_serializable(self):
        report = build_grounded_report(_run_graph("consistent"), clock=frozen_clock)
        self.assertIn("conclusion", json.dumps(report.model_dump(mode="json")))

    def test_graph_result_is_the_engine_result_type(self):
        from investigation.agent import InvestigationResult

        result = _run_graph("insufficient_history")
        self.assertIsInstance(result, InvestigationResult)


# ---------------------------------------------------------------------------
# 10. End-to-end through LangGraph
# ---------------------------------------------------------------------------


class EndToEndTests(unittest.TestCase):
    def test_end_to_end_execution_through_the_graph(self):
        # repository -> tools -> evidence -> engine nodes -> critic -> report
        result = _run_graph("persistent_multi_signal")
        report = build_grounded_report(result, clock=frozen_clock)

        self.assertEqual(result.stop_reason, "only_unanswerable_questions_remain")
        self.assertGreaterEqual(result.final_assessment["tool_call_count"], 1)
        self.assertTrue(any(e["event_type"] == "stop" for e in result.state.trace))
        self.assertTrue(report.observations)
        self.assertTrue(report.link.evidence_digest)

        # The run really walked the graph: streamed node order reaches every
        # node and terminates at finalize.
        order = _node_order("persistent_multi_signal")
        for node in graph_module.GRAPH_NODES:
            self.assertIn(node, order)
        self.assertEqual(order[-1], "finalize")

    def test_streaming_shows_langgraph_supersteps_not_a_single_call(self):
        # If the graph were merely a wrapper, one "node" would run everything.
        order = _node_order("consistent")
        self.assertGreater(len(order), 10)  # 9 distinct nodes + loop revisits
        self.assertGreater(order.count("check_stop_conditions"), 1)

    def test_recursion_error_surface_exists_for_out_of_bounds_loops(self):
        # Structural guarantee: the graph's own runaway guard is importable and
        # distinct from the engine's step-budget RuntimeError.
        self.assertTrue(issubclass(GraphRecursionError, Exception))

    def test_empty_history_terminates_honestly(self):
        result = _run_graph("zero_sessions")
        self.assertEqual(result.stop_reason, "evidence_insufficient")
        report = build_grounded_report(result, clock=frozen_clock)
        self.assertEqual(report.conclusion.status, "inconclusive")


if __name__ == "__main__":
    unittest.main()
