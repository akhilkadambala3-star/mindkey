"""Old-vs-new orchestrator regression tests (Phase 5).

The reference implementation is ``engine.run_investigation`` (the hand-written
bounded state machine); the candidate is ``graph.run_investigation_graph``
(the LangGraph StateGraph over the *same* engine node bodies). This module
runs all 11 evaluation scenarios through both orchestrators and requires
equivalent behavior, not byte-identical output.

Equivalence dimensions (per the migration contract):
stop reason, persistence summary, hypothesis statuses, alternative statuses,
evidence digest, tool selection (names, availability and order), tool-call
count, conclusion status/basis, uncertainty level, trace content, and the
privacy/safety scans.

Trace equivalence: the graph routes from an empty tool selection straight to
the stop check (``route_after_select``), exactly like the reference machine,
so its traces are event-for-event identical -- same node names (the engine's
own node bodies), same event types, same sequence numbers. The regression
suite asserts full trace equality, and the Phase 4 service payload built
from a graph run equals the frozen service's output byte for byte.
"""

import json
import unittest
from datetime import datetime, timezone
from unittest import mock

import investigation.agent.engine as engine_module
import investigation.agent.graph as graph_module
from investigation import repository as repository_module
from investigation.agent import (
    build_grounded_report,
    evaluate_scenarios,
    run_investigation,
    run_investigation_graph,
)
from tests.agent import scenarios
from tests.agent.scenarios import (
    EVALUATION_SCENARIOS,
    USER,
    clinical_terms_in,
    frozen_clock,
    get_scenario,
    outcome_texts,
    raw_field_names_in,
)

FROZEN_NOW = frozen_clock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_old(scenario):
    """Run one scenario through the reference (old) orchestrator."""
    if scenario.default_repository_unavailable:
        with mock.patch.object(
            engine_module,
            "default_repository",
            side_effect=repository_module.RepositoryError("outage"),
        ):
            return run_investigation(
                USER,
                scenario.session_id,
                clock=frozen_clock,
                as_of=FROZEN_NOW,
                alt_windows=scenario.alt_windows,
            )
    return run_investigation(
        USER,
        scenario.session_id,
        repository=scenario.repository_factory(),
        clock=frozen_clock,
        as_of=FROZEN_NOW,
        alt_windows=scenario.alt_windows,
    )


def _run_new(scenario):
    """Run one scenario through the LangGraph orchestrator."""
    if scenario.default_repository_unavailable:
        # The graph module imports its own name, so the outage is mocked there.
        with mock.patch.object(
            graph_module,
            "default_repository",
            side_effect=repository_module.RepositoryError("outage"),
        ):
            return run_investigation_graph(
                USER,
                scenario.session_id,
                clock=frozen_clock,
                as_of=FROZEN_NOW,
                alt_windows=scenario.alt_windows,
            )
    return run_investigation_graph(
        USER,
        scenario.session_id,
        repository=scenario.repository_factory(),
        clock=frozen_clock,
        as_of=FROZEN_NOW,
        alt_windows=scenario.alt_windows,
    )


class OrchestratorPair:
    """Old/new results plus reports for one scenario."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.old = _run_old(scenario)
        self.new = _run_new(scenario)
        self.old_report = build_grounded_report(self.old, clock=frozen_clock)
        self.new_report = build_grounded_report(self.new, clock=frozen_clock)

    def assert_equivalent(self, case):
        old, new = self.old, self.new

        # Stopping conditions.
        case.assertEqual(new.stop_reason, old.stop_reason, self.scenario.key)

        # Evidence: same facts registered (digest covers kind/values/statements).
        case.assertEqual(new.evidence_digest, old.evidence_digest, self.scenario.key)
        case.assertEqual(
            [item["id"] for item in new.state.evidence],
            [item["id"] for item in old.state.evidence],
            self.scenario.key,
        )

        # Tool selection: names, availability and order are identical.
        old_tools = [(c["tool"], c["available"]) for c in old.tool_calls]
        new_tools = [(c["tool"], c["available"]) for c in new.tool_calls]
        case.assertEqual(new_tools, old_tools, self.scenario.key)

        # Hypotheses and alternatives.
        case.assertEqual(
            [(h.hypothesis, h.status) for h in new.hypotheses],
            [(h.hypothesis, h.status) for h in old.hypotheses],
            self.scenario.key,
        )
        case.assertEqual(
            [(a.candidate, a.status) for a in new.alternatives],
            [(a.candidate, a.status) for a in old.alternatives],
            self.scenario.key,
        )

        # The engine's own summary.
        for key in (
            "persistence",
            "supported_hypotheses",
            "open_questions",
            "iterations",
            "tool_call_count",
            "evidence_digest",
            "disclaimer",
            "data_quality",
        ):
            case.assertEqual(
                new.final_assessment.get(key),
                old.final_assessment.get(key),
                f"{self.scenario.key}:{key}",
            )

        # Report semantics.
        case.assertEqual(
            self.new_report.conclusion.status, self.old_report.conclusion.status, self.scenario.key
        )
        case.assertEqual(
            self.new_report.conclusion.basis, self.old_report.conclusion.basis, self.scenario.key
        )
        case.assertEqual(
            self.new_report.uncertainty.level,
            self.old_report.uncertainty.level,
            self.scenario.key,
        )
        case.assertEqual(
            self.new_report.uncertainty.reasons,
            self.old_report.uncertainty.reasons,
            self.scenario.key,
        )
        case.assertEqual(
            self.new_report.rejected_claims, self.old_report.rejected_claims, self.scenario.key
        )
        case.assertEqual(
            [c.statement for c in self.new_report.observations],
            [c.statement for c in self.old_report.observations],
            self.scenario.key,
        )

        # State continuity: same bookkeeping into AgentState.
        case.assertEqual(new.state.stop_reason, old.state.stop_reason, self.scenario.key)
        case.assertEqual(
            new.state.investigation_plan, old.state.investigation_plan, self.scenario.key
        )
        case.assertEqual(
            [(t["tool"], t["available"]) for t in new.state.tool_results],
            [(t["tool"], t["available"]) for t in old.state.tool_results],
            self.scenario.key,
        )
        case.assertEqual(
            new.state.final_assessment, old.state.final_assessment, self.scenario.key
        )


# ---------------------------------------------------------------------------
# Equivalence across the whole catalog
# ---------------------------------------------------------------------------


class CatalogEquivalenceTests(unittest.TestCase):
    def test_every_scenario_is_equivalent_old_vs_new(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                OrchestratorPair(scenario).assert_equivalent(self)

    def test_stop_reasons_match_per_scenario(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                pair = OrchestratorPair(scenario)
                self.assertEqual(pair.new.stop_reason, pair.old.stop_reason)

    def test_evidence_digests_match_per_scenario(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                pair = OrchestratorPair(scenario)
                self.assertEqual(pair.new.evidence_digest, pair.old.evidence_digest)

    def test_tool_selection_matches_per_scenario(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                pair = OrchestratorPair(scenario)
                old_seq = [c["tool"] for c in pair.old.tool_calls]
                new_seq = [c["tool"] for c in pair.new.tool_calls]
                self.assertEqual(new_seq, old_seq)
                self.assertEqual(
                    pair.new.final_assessment["tool_call_count"],
                    pair.old.final_assessment["tool_call_count"],
                )

    def test_conclusion_semantics_match_per_scenario(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                pair = OrchestratorPair(scenario)
                self.assertEqual(
                    (
                        pair.new_report.conclusion.status,
                        pair.new_report.conclusion.basis,
                        pair.new_report.uncertainty.level,
                    ),
                    (
                        pair.old_report.conclusion.status,
                        pair.old_report.conclusion.basis,
                        pair.old_report.uncertainty.level,
                    ),
                )

    def test_outage_scenario_matches_through_the_graph(self):
        pair = OrchestratorPair(get_scenario("repository_unavailable"))
        self.assertEqual(pair.new.stop_reason, "data_unavailable")
        self.assertEqual(pair.old.stop_reason, "data_unavailable")
        self.assertEqual(pair.new.evidence_digest, pair.old.evidence_digest)
        self.assertEqual(
            pair.new_report.conclusion.basis, pair.old_report.conclusion.basis
        )


# ---------------------------------------------------------------------------
# Documented divergence: trace event counts
# ---------------------------------------------------------------------------


class TraceEquivalenceTests(unittest.TestCase):
    def test_graph_traces_are_identical_to_the_reference(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                pair = OrchestratorPair(scenario)
                self.assertEqual(
                    pair.new.state.trace, pair.old.state.trace, scenario.key
                )

    def test_early_stop_paths_have_identical_trace_lengths(self):
        # Scenarios that stop without a collection loop have the shortest
        # traces; equality here guards against extra head/tail events.
        for key in ("insufficient_history", "zero_sessions", "ml_anomaly_only"):
            with self.subTest(scenario=key):
                pair = OrchestratorPair(get_scenario(key))
                self.assertEqual(
                    len(pair.new.state.trace), len(pair.old.state.trace), key
                )

    def test_looping_paths_record_the_engine_vocabulary(self):
        from investigation.agent.engine import NODE_NAMES

        pair = OrchestratorPair(get_scenario("consistent"))
        allowed = set(NODE_NAMES) | {
            "evaluate_hypotheses",
            "assess_alternatives",
            "critic",
        }
        for event in pair.new.state.trace:
            self.assertIn(event["node"], allowed)


# ---------------------------------------------------------------------------
# Determinism of the graph orchestrator
# ---------------------------------------------------------------------------


class GraphDeterminismTests(unittest.TestCase):
    def test_repeated_graph_runs_are_identical(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                first = _run_new(scenario)
                second = _run_new(scenario)
                self.assertEqual(first.evidence_digest, second.evidence_digest)
                self.assertEqual(
                    [c["tool"] for c in first.tool_calls],
                    [c["tool"] for c in second.tool_calls],
                )
                self.assertEqual(
                    first.state.trace, second.state.trace
                )

    def test_graph_determinism_matches_reference_determinism(self):
        scenario = get_scenario("persistent_multi_signal")
        pair_a = OrchestratorPair(scenario)
        pair_b = OrchestratorPair(scenario)
        self.assertEqual(
            pair_a.new.evidence_digest, pair_b.new.evidence_digest
        )
        self.assertEqual(
            pair_a.new_report.model_dump(mode="json"),
            pair_b.new_report.model_dump(mode="json"),
        )


# ---------------------------------------------------------------------------
# Privacy / safety parity
# ---------------------------------------------------------------------------


class PrivacySafetyParityTests(unittest.TestCase):
    def _surfaces(self, result, report):
        return {
            "report": report.model_dump_json(),
            "trace": json.dumps(result.state.trace, default=str),
            "assessment": json.dumps(result.final_assessment, default=str),
            "next_action": result.state.next_action or "",
        }

    def test_no_clinical_language_on_any_graph_surface(self):
        for scenario in EVALUATION_SCENARIOS:
            pair = OrchestratorPair(scenario)
            for name, text in self._surfaces(pair.new, pair.new_report).items():
                with self.subTest(scenario=scenario.key, surface=name):
                    self.assertEqual(clinical_terms_in(text), ())

    def test_no_raw_stored_field_names_on_any_graph_surface(self):
        for scenario in EVALUATION_SCENARIOS:
            pair = OrchestratorPair(scenario)
            for name, text in self._surfaces(pair.new, pair.new_report).items():
                with self.subTest(scenario=scenario.key, surface=name):
                    self.assertEqual(raw_field_names_in(text), ())

    def test_trace_carries_fingerprints_not_arguments(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                pair = OrchestratorPair(scenario)
                for event in pair.new.state.trace:
                    self.assertNotIn("args", event)
                    trace_text = json.dumps(event, default=str)
                    self.assertNotIn(scenario.session_id, trace_text)
                    if event["event_type"] == "tool_call":
                        self.assertEqual(len(event["args_fingerprint"]), 16)

    def test_rendered_report_has_no_markup_and_carries_the_disclaimer(self):
        from investigation.agent import render

        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                pair = OrchestratorPair(scenario)
                text = render(pair.new_report, pair.new)
                self.assertNotIn("<script", text)
                self.assertIn(pair.new_report.disclaimer, text)


# ---------------------------------------------------------------------------
# End-to-end: the executed path really is the LangGraph graph
# ---------------------------------------------------------------------------


class EndToEndLangGraphPathTests(unittest.TestCase):
    def test_scenario_executes_through_langgraph_supersteps(self):
        scenario = get_scenario("persistent_multi_signal")
        repository = scenario.repository_factory()
        ctx = graph_module._GraphContext(
            user_id=USER,
            session_id=scenario.session_id,
            caching=graph_module.CachingRepository(repository),
            trace=graph_module.InvestigationTrace(clock=frozen_clock),
            state=graph_module.AgentState(),
            registry=graph_module.EvidenceRegistry(),
            as_of=FROZEN_NOW,
            max_iterations=4,
            max_tool_calls=8,
            alt_windows=scenario.alt_windows,
            blocking_reason=None,
        )
        compiled = graph_module.build_investigation_graph()
        order = []
        for chunk in compiled.stream(
            {"ctx": ctx, **graph_module._projection(ctx)},
            config={"recursion_limit": graph_module._recursion_limit(4)},
            stream_mode="values",
        ):
            if "last_node" in chunk:
                order.append(chunk["last_node"])

        # START -> intake -> plan -> hypotheses -> [select -> execute ->
        # assess -> update -> check]* -> finalize: at least one full loop plus
        # the head and the tail, i.e. many supersteps, not one wrapped call.
        self.assertEqual(order[0], "incident_intake")
        self.assertEqual(order[-1], "finalize")
        self.assertGreater(order.count("select_tool"), 1)
        self.assertGreater(order.count("check_stop_conditions"), 1)
        self.assertEqual(order.count("incident_intake"), 1)

        result = graph_module._build_result(ctx)
        self.assertEqual(result.stop_reason, "only_unanswerable_questions_remain")
        report = build_grounded_report(result, clock=frozen_clock)
        self.assertTrue(report.observations)

    def test_full_pipeline_repository_to_report_through_the_graph(self):
        # repository -> tools -> evidence -> engine nodes -> critic -> report
        scenario = get_scenario("persistent_multi_signal")
        result = _run_new(scenario)
        report = build_grounded_report(result, clock=frozen_clock)

        self.assertEqual(result.stop_reason, "only_unanswerable_questions_remain")
        self.assertEqual(report.conclusion.status, "preliminary")
        self.assertTrue(result.final_assessment["tool_call_count"] >= 1)
        self.assertTrue(any(e["event_type"] == "stop" for e in result.state.trace))
        # The critic events continue the engine trace (one continuous sequence).
        seqs = [e["seq"] for e in report.critic_events]
        engine_seqs = [e["seq"] for e in result.state.trace]
        if seqs:
            self.assertEqual(min(seqs), max(engine_seqs) + 1)


class Phase4ServiceEquivalenceTests(unittest.TestCase):
    """Checkpoint 10: the graph result survives the Phase 4 payload boundary.

    ``services.investigations.run_for_user`` is frozen and still calls the
    reference engine. This test builds the exact same service payload from a
    graph-orchestrated run and requires it to equal the frozen service's
    output field for field -- i.e. the LangGraph migration is invisible to
    the API layer. (Automated coverage: in-memory repository. The live
    Supabase leg -- POST /typing/session -> Isolation Forest -> stored
    anomaly row -> investigation -- remains manual smoke-test coverage;
    the repository layer and row shapes it depends on are exercised here.)
    """

    def _graph_payload(self, scenario, result):
        from investigation.agent import render_timeline_for

        report = build_grounded_report(result, clock=frozen_clock)
        timeline = render_timeline_for(report, result, clock=frozen_clock)
        return {
            "schema_version": result.schema_version,
            "user_id": result.user_id,
            "session_id": str(result.session_id),
            "stop_reason": result.stop_reason,
            "evidence_digest": result.evidence_digest,
            "report": report.model_dump(mode="json"),
            "timeline": list(timeline),
            "limitations": list(report.limitations),
            "disclaimer": report.disclaimer,
        }

    def test_service_payload_from_the_graph_equals_the_frozen_service_output(self):
        from services.investigations import run_for_user

        for key in ("persistent_multi_signal", "insufficient_history", "consistent"):
            with self.subTest(scenario=key):
                scenario = get_scenario(key)
                expected = run_for_user(
                    USER,
                    scenario.session_id,
                    repository=scenario.repository_factory(),
                    clock=frozen_clock,
                )
                # The frozen service resolves as_of from the wall clock, so the
                # graph run must use the same convention for the payloads to
                # describe the same windows.
                result = run_investigation_graph(
                    USER,
                    scenario.session_id,
                    repository=scenario.repository_factory(),
                    clock=frozen_clock,
                    alt_windows=scenario.alt_windows,
                )
                actual = self._graph_payload(scenario, result)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
