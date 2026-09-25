"""Tests for the Phase 3.3 deterministic investigation engine.

These tests pin the engine's *behaviour*, not just its output: node order, tool
selection, stop reasons, bounds, single persistence registration, read-only data
access, privacy, determinism, and medical-language safety. All fixtures are
in-memory; no network, no Supabase, no LLM, and no environment variables.
"""

import json
import unittest
from datetime import datetime, timezone
from typing import get_args
from unittest import mock

from investigation.agent import (
    ALT_WINDOWS,
    DEFAULT_RECENT_LIMIT,
    ENGINE_SCHEMA_VERSION,
    MAX_ITERATIONS,
    MAX_TOOL_CALLS,
    NODE_NAMES,
    EvidenceItem,
    InvestigationResult,
    InvestigationTrace,
    StopReason,
    args_fingerprint,
    run_investigation,
    select_next_collection,
)
from investigation.agent.evidence import SOURCE_ADAPTER
from investigation.agent.trace import TraceEvent
from investigation.repository import RepositoryError
from investigation.state import AgentState
from tests import fixtures

USER = "user-1"
FIXED_CLOCK = lambda: datetime(2026, 9, 24, 9, 42, tzinfo=timezone.utc)  # noqa: E731

#: The approved node set.
EXPECTED_NODES = (
    "ingest_signal",
    "check_data_quality",
    "initialize_hypotheses",
    "investigate",
    "collect_required_evidence",
    "assess_persistence",
    "update_hypotheses",
    "check_stop_conditions",
    "finalize",
)

#: Vocabulary that must never appear in engine-authored behavioral text.
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

#: Phase 2's fixed non-diagnostic negation, the only permitted exemption.
DISCLAIMER = "an anomaly score is not a diagnosis."

CONTENT_KEYS = ("text", "content", "keystrokes", "message", "note_text")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _persistent_repo():
    """Five recent sessions with aligned movement across six signals."""
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 36)
    ]
    sessions += [
        fixtures.make_session(
            session_id=f"R{i:04d}",
            days_ago=i,
            typing_speed=200.0,
            dwell_mean=0.20,
            flight_mean=0.14,
            rhythm_variability=0.30,
            correction_rate=0.12,
            pause_count=9,
        )
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(
        sessions=sessions,
        anomalies={"R0001": {"anomaly_score": 0.62, "is_anomaly": True}},
    )


def _shifted_repo():
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 36)
    ]
    sessions += [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=233.0)
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(sessions=sessions)


def _stable_repo():
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
            for i in range(1, 35)
        ]
    )


def _thin_repo():
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i) for i in range(3)
        ]
    )


def _invalid_repo():
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(1, 20)
    ]
    sessions.append(fixtures.make_invalid_session(session_id="BAD1", days_ago=3))
    return fixtures.FakeRepository(sessions=sessions)


def _run(repository, session_id="R0001", **kwargs):
    kwargs.setdefault("as_of", datetime.now(timezone.utc))
    trace = kwargs.pop("trace", None)
    if trace is None:
        trace = InvestigationTrace(clock=FIXED_CLOCK)
    return run_investigation(
        USER, session_id, repository=repository, trace=trace, **kwargs
    )


def _node_enters(result):
    return [
        event["node"]
        for event in result.state.trace
        if event["event_type"] == "node_enter"
    ]


def _events(result, event_type):
    return [e for e in result.state.trace if e["event_type"] == event_type]


# ---------------------------------------------------------------------------
# Input validation and result shape
# ---------------------------------------------------------------------------


class InputAndResultTests(unittest.TestCase):
    def test_empty_user_id_is_rejected(self):
        with self.assertRaises(ValueError):
            run_investigation("", "R0001", repository=_stable_repo())

    def test_empty_session_id_is_rejected(self):
        with self.assertRaises(ValueError):
            run_investigation(USER, "", repository=_stable_repo())

    def test_result_is_an_investigation_result(self):
        result = _run(_persistent_repo())
        self.assertIsInstance(result, InvestigationResult)
        self.assertIsInstance(result.state, AgentState)

    def test_result_ids_and_schema_version(self):
        result = _run(_persistent_repo())
        self.assertEqual(result.user_id, USER)
        self.assertEqual(result.session_id, "R0001")
        self.assertEqual(result.schema_version, ENGINE_SCHEMA_VERSION)

    def test_result_is_json_serializable(self):
        result = _run(_persistent_repo())
        json.dumps(result.model_dump(mode="json"))

    def test_constants_are_the_approved_values(self):
        self.assertEqual(MAX_ITERATIONS, 4)
        self.assertEqual(MAX_TOOL_CALLS, 8)
        self.assertEqual(ALT_WINDOWS, ((14, 14),))
        self.assertEqual(DEFAULT_RECENT_LIMIT, 500)
        self.assertEqual(NODE_NAMES, EXPECTED_NODES)

    def test_stop_reason_is_from_the_closed_vocabulary(self):
        allowed = set(get_args(StopReason))
        for repository, session_id in (
            (_persistent_repo(), "R0001"),
            (_shifted_repo(), "R0001"),
            (_stable_repo(), "S0001"),
            (_thin_repo(), "S0001"),
        ):
            with self.subTest(session=session_id):
                self.assertIn(_run(repository, session_id).stop_reason, allowed)


# ---------------------------------------------------------------------------
# Node order and trace
# ---------------------------------------------------------------------------


class FlowAndTraceTests(unittest.TestCase):
    def test_node_sequence_for_an_adequate_investigation(self):
        result = _run(_persistent_repo())
        self.assertEqual(
            _node_enters(result),
            [
                "ingest_signal",
                "check_data_quality",
                "initialize_hypotheses",
                "investigate",
                "collect_required_evidence",
                "assess_persistence",
                "check_stop_conditions",
                "investigate",
                "collect_required_evidence",
                "assess_persistence",
                "update_hypotheses",
                "check_stop_conditions",
                "investigate",
                "check_stop_conditions",
                "finalize",
            ],
        )

    def test_trace_starts_with_ingest_and_ends_with_finalize(self):
        result = _run(_shifted_repo())
        enters = _node_enters(result)
        self.assertEqual(enters[0], "ingest_signal")
        self.assertEqual(enters[-1], "finalize")

    def test_every_node_entered_is_in_the_fixed_node_set(self):
        result = _run(_persistent_repo())
        for node in _node_enters(result):
            self.assertIn(node, NODE_NAMES)

    def test_trace_sequence_numbers_are_contiguous(self):
        result = _run(_persistent_repo())
        self.assertEqual(
            [event["seq"] for event in result.state.trace],
            list(range(1, len(result.state.trace) + 1)),
        )

    def test_tool_calls_and_results_pair_up(self):
        result = _run(_persistent_repo())
        calls = [e for e in _events(result, "tool_call") if e["tool"] is not None]
        results = [e for e in _events(result, "tool_result") if e["tool"] is not None]
        self.assertEqual(len(calls), len(results))
        for call, outcome in zip(calls, results):
            self.assertEqual(call["tool"], outcome["tool"])
            self.assertIsNotNone(call["args_fingerprint"])

    def test_decision_events_are_recorded_for_routing(self):
        result = _run(_persistent_repo())
        details = [e["detail"] for e in _events(result, "decision")]
        self.assertTrue(any(d.startswith("collect:") for d in details))
        self.assertIn("no_collectible_evidence", details)

    def test_stop_event_records_the_reason(self):
        result = _run(_persistent_repo())
        stops = _events(result, "stop")
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0]["detail"], result.stop_reason)

    def test_nodes_visited_matches_the_trace(self):
        result = _run(_persistent_repo())
        self.assertEqual(
            result.state.context["derived"]["nodes_visited"], _node_enters(result)
        )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class RoutingTests(unittest.TestCase):
    def _fp(self, tool, **args):
        return args_fingerprint(tool, **args)

    def test_plan_steps_are_served_in_order(self):
        request = select_next_collection(
            user_id="u", plan=["recent_sessions", "window_robustness"]
        )
        self.assertEqual(request.step, "recent_sessions")
        self.assertEqual(request.tool, "get_recent_sessions")

    def test_collected_plan_steps_are_skipped(self):
        request = select_next_collection(
            user_id="u",
            plan=["recent_sessions", "window_robustness"],
            collected_plan=["recent_sessions"],
        )
        self.assertEqual(request.tool, "calculate_behavioral_drift")

    def test_a_called_fingerprint_is_never_repeated(self):
        fingerprint = self._fp(
            "get_recent_sessions", user_id="u", limit=500, window_days=None
        )
        request = select_next_collection(
            user_id="u",
            plan=["recent_sessions"],
            called_fingerprints=[fingerprint],
        )
        self.assertIsNone(request)

    def test_dead_end_question_returns_none(self):
        fingerprint = self._fp(
            "get_recent_sessions", user_id="u", limit=500, window_days=None
        )
        self.assertIsNone(
            select_next_collection(
                user_id="u",
                open_questions=["recent_session_depth", "additional_sessions"],
                called_fingerprints=[fingerprint],
            )
        )

    def test_unanswerable_questions_are_never_selected(self):
        self.assertIsNone(
            select_next_collection(
                user_id="u", open_questions=["context_factors", "capture_change"]
            )
        )

    def test_window_robustness_falls_back_to_a_window_comparison(self):
        drift = self._fp(
            "calculate_behavioral_drift",
            user_id="u",
            recent_window_days=14,
            baseline_window_days=14,
        )
        request = select_next_collection(
            user_id="u",
            open_questions=["window_robustness"],
            called_fingerprints=[drift],
        )
        self.assertEqual(request.tool, "compare_time_windows")
        self.assertEqual(request.question, "window_robustness")

    def test_questions_follow_canonical_order(self):
        request = select_next_collection(
            user_id="u", open_questions=["context_factors", "window_robustness"]
        )
        self.assertEqual(request.question, "window_robustness")

    def test_exhausted_budget_returns_none(self):
        self.assertIsNone(
            select_next_collection(
                user_id="u", plan=["recent_sessions"], budget_remaining=0
            )
        )

    def test_request_carries_step_tool_and_args(self):
        request = select_next_collection(
            user_id="u", plan=["window_robustness"], alt_windows=((9, 21),)
        )
        self.assertEqual(request.step, "window_robustness")
        self.assertEqual(request.tool, "calculate_behavioral_drift")
        self.assertEqual(request.args["recent_window_days"], 9)
        self.assertEqual(request.args["baseline_window_days"], 21)

    def test_routing_is_deterministic(self):
        args = {"user_id": "u", "open_questions": ["window_robustness"]}
        first = select_next_collection(**args)
        second = select_next_collection(**args)
        self.assertEqual(first, second)

    def test_no_repository_parameter(self):
        import inspect

        self.assertNotIn(
            "repository", inspect.signature(select_next_collection).parameters
        )


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------


class StopConditionTests(unittest.TestCase):
    def test_persistent_scenario_stops_evidence_sufficient(self):
        result = _run(_persistent_repo())
        self.assertEqual(result.stop_reason, "evidence_sufficient")
        self.assertTrue(result.final_assessment["persistence"]["eligible"])

    def test_thin_history_stops_evidence_insufficient(self):
        result = _run(_thin_repo(), "S0001")
        self.assertEqual(result.stop_reason, "evidence_insufficient")

    def test_zero_sessions_stops_evidence_insufficient(self):
        result = _run(fixtures.FakeRepository(sessions=[]), "X0001")
        self.assertEqual(result.stop_reason, "evidence_insufficient")

    def test_iteration_limit_is_enforced(self):
        result = _run(_persistent_repo(), max_iterations=1)
        self.assertEqual(result.stop_reason, "iteration_limit")
        self.assertLessEqual(len(result.tool_calls), 1)

    def test_tool_budget_is_enforced(self):
        result = _run(_thin_repo(), "S0001", max_tool_calls=0)
        self.assertEqual(result.stop_reason, "tool_budget_exhausted")
        self.assertEqual(result.tool_calls, [])

    def test_no_further_evidence_stop(self):
        result = _run(_shifted_repo())
        self.assertEqual(result.stop_reason, "no_further_evidence")

    def test_only_unanswerable_questions_remain(self):
        # An alternative that reproduces the default windows agrees, so window
        # robustness is resolved and only context questions are left open.
        result = _run(_shifted_repo(), alt_windows=((7, 30),))
        self.assertEqual(result.stop_reason, "only_unanswerable_questions_remain")
        self.assertTrue(all(not q["answerable"] for q in result.final_assessment["open_questions"]))

    def test_insufficient_data_never_becomes_a_persistent_claim(self):
        result = _run(_thin_repo(), "S0001")
        self.assertFalse(result.final_assessment["persistence"]["eligible"])
        self.assertNotEqual(
            result.final_assessment["persistence"]["status"], "persistent_change"
        )

    def test_repository_failure_stops_data_unavailable(self):
        with mock.patch(
            "investigation.agent.engine.default_repository",
            side_effect=RepositoryError("no client"),
        ):
            result = _run(None, "R0001")
        self.assertEqual(result.stop_reason, "data_unavailable")
        self.assertEqual(result.state.evidence, [])
        self.assertEqual(len(result.hypotheses), 4)

    def test_broken_repository_degrades_to_insufficient(self):
        result = _run(fixtures.FailingRepository(), "R0001")
        self.assertEqual(result.stop_reason, "evidence_insufficient")
        unavailable = [
            item for item in result.state.evidence if item["kind"] == "tool_unavailable"
        ]
        self.assertTrue(unavailable)


# ---------------------------------------------------------------------------
# AgentState progression
# ---------------------------------------------------------------------------


class StateProgressionTests(unittest.TestCase):
    def test_state_bookkeeping_is_populated(self):
        result = _run(_persistent_repo())
        state = result.state
        self.assertEqual(state.trigger["session_id"], "R0001")
        self.assertIsNotNone(state.ml_evidence)
        self.assertEqual(state.investigation_plan, ["recent_sessions", "window_robustness"])
        self.assertEqual(state.iteration, result.iterations)
        self.assertEqual(state.stop_reason, result.stop_reason)
        self.assertTrue(state.next_action)
        self.assertTrue(state.limitations)
        self.assertTrue(state.evidence)
        self.assertTrue(state.trace)
        self.assertEqual(state.tools_called, [c["tool"] for c in result.tool_calls])

    def test_context_derived_namespace_is_separate(self):
        result = _run(_persistent_repo())
        self.assertIn("derived", result.state.context)
        derived = result.state.context["derived"]
        self.assertEqual(derived["stop_reason"], result.stop_reason)
        self.assertIn("evidence_digest", derived)
        self.assertIn("persistence", derived)
        # User-provided context is never fabricated.
        self.assertNotIn("checkins", result.state.context)
        self.assertNotIn("symptoms", result.state.context)

    def test_state_evidence_dicts_are_valid_items(self):
        result = _run(_persistent_repo())
        for payload in result.state.evidence:
            self.assertIsInstance(EvidenceItem(**payload), EvidenceItem)

    def test_state_trace_dicts_are_valid_events(self):
        result = _run(_persistent_repo())
        for payload in result.state.trace:
            self.assertIsInstance(TraceEvent(**payload), TraceEvent)

    def test_mirrored_result_fields_come_from_state(self):
        result = _run(_persistent_repo())
        self.assertEqual(result.hypotheses, result.state.hypotheses)
        self.assertEqual(result.missing_evidence, result.state.missing_evidence)
        self.assertEqual(result.final_assessment, result.state.final_assessment)
        self.assertEqual(result.stop_reason, result.state.stop_reason)
        self.assertEqual(result.limitations, result.state.limitations)

    def test_alternatives_are_recorded_in_derived_context(self):
        result = _run(_persistent_repo())
        derived = result.state.context["derived"]["alternatives"]
        self.assertEqual(
            [item["candidate"] for item in derived],
            [item.candidate for item in result.alternatives],
        )


# ---------------------------------------------------------------------------
# Single persistence assessment (Phase 3.2 is authoritative)
# ---------------------------------------------------------------------------


class PersistenceAuthorityTests(unittest.TestCase):
    def test_persistence_evidence_is_registered_exactly_once(self):
        result = _run(_persistent_repo())
        for kind in ("persistence_depth", "persistence_robustness", "persistence"):
            with self.subTest(kind=kind):
                items = [i for i in result.state.evidence if i["kind"] == kind]
                self.assertEqual(len(items), 1)

    def test_assessment_runs_once_in_the_trace(self):
        result = _run(_persistent_repo())
        assessments = [
            e
            for e in _events(result, "decision")
            if e["detail"].startswith("persistence_assessed")
        ]
        self.assertEqual(len(assessments), 1)
        already = [
            e
            for e in _events(result, "decision")
            if e["detail"] == "persistence_already_assessed"
        ]
        self.assertEqual(len(already), 0)

    def test_alternative_window_is_collected_and_fed_to_robustness(self):
        result = _run(_persistent_repo())
        self.assertIn(
            "calculate_behavioral_drift",
            [call["tool"] for call in result.tool_calls],
        )
        self.assertNotEqual(
            result.final_assessment["persistence"]["robustness"], "not_assessed"
        )

    def test_phase2_verdict_is_carried_not_recomputed(self):
        result = _run(_shifted_repo())
        persistence = result.final_assessment["persistence"]
        self.assertEqual(persistence["status"], "recent_variation")
        self.assertFalse(persistence["eligible"])
        self.assertEqual(persistence["downgrade_reason"], "phase2_status_not_persistent")

    def test_hypotheses_are_the_fixed_catalog(self):
        result = _run(_persistent_repo())
        self.assertEqual(
            [h.hypothesis for h in result.hypotheses],
            [
                "Temporary behavioral variation",
                "Contextual disruption (sleep, fatigue, stress or workload)",
                "Persistent behavioral change",
                "Data-quality or capture artifact",
            ],
        )


# ---------------------------------------------------------------------------
# Determinism, read-only access, privacy
# ---------------------------------------------------------------------------


class CountingRepository(fixtures.FakeRepository):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_calls = 0
        self.anomaly_calls = 0

    def list_sessions(self, user_id):
        self.list_calls += 1
        return super().list_sessions(user_id)

    def get_anomaly_result(self, session_id):
        self.anomaly_calls += 1
        return super().get_anomaly_result(session_id)


class WriteSpyRepository(fixtures.FakeRepository):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writes = []

    def insert(self, *args, **kwargs):
        self.writes.append("insert")
        raise AssertionError("the engine must never write")

    def update(self, *args, **kwargs):
        self.writes.append("update")
        raise AssertionError("the engine must never write")

    def delete(self, *args, **kwargs):
        self.writes.append("delete")
        raise AssertionError("the engine must never write")

    def upsert(self, *args, **kwargs):
        self.writes.append("upsert")
        raise AssertionError("the engine must never write")


class DeterminismTests(unittest.TestCase):
    def test_repeated_runs_have_the_same_digest_and_tool_sequence(self):
        repository = _persistent_repo()
        first = _run(repository)
        second = _run(repository)
        self.assertEqual(first.evidence_digest, second.evidence_digest)
        self.assertEqual(
            [c["tool"] for c in first.tool_calls],
            [c["tool"] for c in second.tool_calls],
        )
        self.assertEqual(first.stop_reason, second.stop_reason)

    def test_fixed_clock_produces_identical_traces(self):
        repository = _persistent_repo()
        first = _run(repository)
        second = _run(repository)
        self.assertEqual(first.state.trace, second.state.trace)

    def test_no_tool_is_called_twice(self):
        for repository, session_id in (
            (_persistent_repo(), "R0001"),
            (_shifted_repo(), "R0001"),
            (_stable_repo(), "S0001"),
        ):
            with self.subTest(session=session_id):
                result = _run(repository, session_id)
                fingerprints = [c["fingerprint"] for c in result.tool_calls]
                self.assertEqual(len(fingerprints), len(set(fingerprints)))

    def test_iterations_count_executed_collections(self):
        result = _run(_shifted_repo())
        self.assertEqual(result.iterations, len(result.tool_calls))
        self.assertEqual(result.state.iteration, result.iterations)


class ReadOnlyTests(unittest.TestCase):
    def test_the_engine_never_writes(self):
        upstream = WriteSpyRepository(
            sessions=[
                fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
                for i in range(1, 15)
            ]
        )
        result = _run(upstream, "S0001")
        self.assertEqual(upstream.writes, [])
        self.assertTrue(result.state.evidence)

    def test_session_history_is_read_once_through_the_cache(self):
        upstream = CountingRepository(
            sessions=[
                fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
                for i in range(1, 15)
            ]
        )
        _run(upstream, "S0001")
        self.assertEqual(upstream.list_calls, 1)

    def test_repository_rows_are_not_mutated(self):
        repository = _persistent_repo()
        before = [dict(row) for row in repository.sessions]
        _run(repository)
        self.assertEqual([dict(row) for row in repository.sessions], before)


class PrivacyTests(unittest.TestCase):
    def test_trace_bytes_contain_no_raw_values(self):
        result = _run(_persistent_repo())
        blob = json.dumps(
            [
                {key: event[key] for key in ("node", "event_type", "tool", "detail")}
                for event in result.state.trace
            ]
        )
        for raw in ("285", "200", "0.12", "anomaly_score", "R0001"):
            self.assertNotIn(raw, blob)

    def test_argument_fingerprints_are_hashes_not_arguments(self):
        result = _run(_persistent_repo())
        for event in _events(result, "tool_call"):
            if event["args_fingerprint"] is not None:
                fingerprint = event["args_fingerprint"]
                self.assertEqual(len(fingerprint), 16)
                self.assertTrue(all(c in "0123456789abcdef" for c in fingerprint))

    def test_evidence_has_no_content_keys(self):
        result = _run(_persistent_repo())
        keys = set()
        for item in result.state.evidence:
            keys |= set(item)
        for forbidden in CONTENT_KEYS:
            self.assertNotIn(forbidden, keys)

    def test_no_free_text_anywhere_in_the_result(self):
        result = _run(_persistent_repo())
        blob = json.dumps(result.model_dump(mode="json")).lower()
        for forbidden in ("keystroke", "\"typed_text\"", "\"message\"", "\"password\""):
            self.assertNotIn(forbidden, blob)


# ---------------------------------------------------------------------------
# Medical-language safety
# ---------------------------------------------------------------------------


class SafetyTests(unittest.TestCase):
    def _blob(self, result):
        parts = [item["statement"] for item in result.state.evidence]
        parts += [h.hypothesis for h in result.hypotheses]
        parts += [f"{a.label} {a.reason}" for a in result.alternatives]
        parts += [json.dumps(result.final_assessment)]
        return " ".join(parts).lower()

    def test_no_clinical_language_in_engine_output(self):
        result = _run(_persistent_repo())
        blob = self._blob(result)
        self.assertIn(DISCLAIMER, blob)
        blob = blob.replace(DISCLAIMER, "")
        for term in CLINICAL_TERMS:
            self.assertNotIn(term, blob)

    def test_final_assessment_carries_the_disclaimer(self):
        result = _run(_persistent_repo())
        self.assertIn("establish", result.final_assessment["disclaimer"])
        self.assertIn(
            SOURCE_ADAPTER,
            {item["source_tool"] for item in result.state.evidence},
        )


# ---------------------------------------------------------------------------
# Integration scenarios
# ---------------------------------------------------------------------------


class ScenarioTests(unittest.TestCase):
    def _status(self, result):
        return {h.hypothesis: h.status for h in result.hypotheses}

    def test_persistent_scenario_supports_persistent_change(self):
        result = _run(_persistent_repo())
        self.assertEqual(
            self._status(result)["Persistent behavioral change"], "supported"
        )
        self.assertEqual(result.final_assessment["persistence"]["classification"], "sustained")

    def test_shifted_scenario_supports_temporary_variation(self):
        result = _run(_shifted_repo())
        self.assertEqual(
            self._status(result)["Temporary behavioral variation"], "supported"
        )
        self.assertNotEqual(
            self._status(result)["Persistent behavioral change"], "supported"
        )

    def test_stable_scenario_never_supports_persistent_change(self):
        result = _run(_stable_repo(), "S0001")
        self.assertNotEqual(
            self._status(result)["Persistent behavioral change"], "supported"
        )
        self.assertEqual(
            result.final_assessment["persistence"]["classification"],
            "no_current_deviation",
        )

    def test_thin_scenario_weakens_persistent_change(self):
        result = _run(_thin_repo(), "S0001")
        self.assertEqual(
            self._status(result)["Persistent behavioral change"], "weakened"
        )
        self.assertEqual(
            self._status(result)["Data-quality or capture artifact"], "supported"
        )

    def test_invalid_rows_are_excluded_and_handled(self):
        result = _run(_invalid_repo(), "S0001")
        invalid = [i for i in result.state.evidence if i["kind"] == "quality_invalid"]
        self.assertTrue(invalid)
        self.assertIn(result.stop_reason, set(get_args(StopReason)))

    def test_supported_hypotheses_always_cite_evidence(self):
        for repository, session_id in (
            (_persistent_repo(), "R0001"),
            (_thin_repo(), "S0001"),
        ):
            with self.subTest(session=session_id):
                result = _run(repository, session_id)
                ids = {item["id"] for item in result.state.evidence}
                for hypothesis in result.hypotheses:
                    if hypothesis.status == "supported":
                        self.assertTrue(hypothesis.supporting_evidence, hypothesis.hypothesis)
                    for cited in (
                        hypothesis.supporting_evidence + hypothesis.contradicting_evidence
                    ):
                        self.assertIn(cited, ids)


if __name__ == "__main__":
    unittest.main()
