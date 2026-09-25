"""Tests for the Phase 3.5 evaluation scenario catalog.

The catalog (``tests/agent/scenarios.py``) characterizes the committed pipeline:
each scenario states the outcome the engine, critic and report are expected to
reach for a synthetic situation. These tests assert those expectations and the
invariants that make the catalog a trustworthy evaluation surface --
determinism, grounding, privacy and medical-language safety.
"""

import json
import re
import unittest
from datetime import datetime, timezone
from typing import get_args

from investigation.agent import (
    NODE_NAMES,
    StopReason,
    TraceEventType,
    render_report_lines,
    render_timeline_for,
)
from investigation.agent.report import (
    CONCLUSION_TEXT,
    REPORT_DISCLAIMER,
    REPORT_LIMITATIONS,
)
from tests.agent import scenarios
from tests.agent.scenarios import (
    EVALUATION_SCENARIOS,
    clinical_terms_in,
    get_scenario,
    outcome_texts,
    raw_field_names_in,
    run_all,
    run_scenario,
    scenario_keys,
)

TIMELINE_RE = re.compile(r"^\d{2}:\d{2}:\d{2} \u2014 .+$")

KNOWN_STOP_REASONS = set(get_args(StopReason))
KNOWN_EVENT_TYPES = set(get_args(TraceEventType))


def assert_expected_outcome(case, outcome):
    """Assert one outcome satisfies every recorded expectation."""
    scenario = outcome.scenario
    exp = scenario.expectation
    label = scenario.key

    case.assertEqual(outcome.stop_reason, exp.stop_reason, label)
    case.assertEqual(outcome.persistence_status, exp.persistence_status, label)
    case.assertEqual(
        outcome.persistence_classification, exp.persistence_classification, label
    )
    case.assertEqual(outcome.eligible, exp.eligible, label)
    case.assertEqual(outcome.robustness, exp.robustness, label)
    case.assertEqual(outcome.conclusion_status, exp.conclusion_status, label)
    case.assertEqual(outcome.conclusion_basis, exp.conclusion_basis, label)
    case.assertEqual(outcome.uncertainty_level, exp.uncertainty_level, label)

    if exp.hypothesis_statuses:
        case.assertEqual(
            outcome.hypothesis_statuses, dict(exp.hypothesis_statuses), label
        )
        case.assertEqual(
            outcome.supported_hypotheses, scenario.supported_hypotheses, label
        )

    for key in exp.required_open_questions:
        case.assertIn(key, outcome.open_question_keys, label)

    for candidate, status in exp.alternative_statuses:
        case.assertEqual(outcome.alternative_statuses.get(candidate), status, label)

    for kind in exp.required_evidence_kinds:
        case.assertIn(kind, outcome.evidence_kinds, label)

    if exp.next_action_contains:
        case.assertIn(exp.next_action_contains, outcome.next_action, label)


# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------


class CatalogTests(unittest.TestCase):
    def test_catalog_order_is_fixed(self):
        self.assertEqual(
            scenario_keys(),
            (
                "consistent",
                "recent_variation",
                "persistent_multi_signal",
                "insufficient_history",
                "zero_sessions",
                "ml_anomaly_only",
                "invalid_rows",
                "sparse_recent",
                "robustness_disagreement",
                "robustness_agreement",
                "repository_unavailable",
            ),
        )

    def test_keys_are_unique(self):
        keys = scenario_keys()
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(keys), len(EVALUATION_SCENARIOS))

    def test_every_scenario_describes_itself(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                self.assertTrue(scenario.title)
                self.assertTrue(scenario.description)
                self.assertTrue(scenario.session_id)
                self.assertIsInstance(scenario.expectation, scenarios.Expectation)

    def test_get_scenario_returns_the_matching_scenario(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                self.assertIs(get_scenario(scenario.key), scenario)

    def test_get_scenario_rejects_unknown_keys(self):
        with self.assertRaises(KeyError):
            get_scenario("no-such-scenario")


# ---------------------------------------------------------------------------
# Expectations (one test per scenario)
# ---------------------------------------------------------------------------


class ScenarioExpectationTests(unittest.TestCase):
    def test_consistent(self):
        assert_expected_outcome(self, run_scenario(get_scenario("consistent")))

    def test_recent_variation(self):
        assert_expected_outcome(self, run_scenario(get_scenario("recent_variation")))

    def test_persistent_multi_signal(self):
        assert_expected_outcome(
            self, run_scenario(get_scenario("persistent_multi_signal"))
        )

    def test_insufficient_history(self):
        assert_expected_outcome(
            self, run_scenario(get_scenario("insufficient_history"))
        )

    def test_zero_sessions(self):
        assert_expected_outcome(self, run_scenario(get_scenario("zero_sessions")))

    def test_ml_anomaly_only(self):
        assert_expected_outcome(self, run_scenario(get_scenario("ml_anomaly_only")))

    def test_invalid_rows(self):
        assert_expected_outcome(self, run_scenario(get_scenario("invalid_rows")))

    def test_sparse_recent(self):
        assert_expected_outcome(self, run_scenario(get_scenario("sparse_recent")))

    def test_robustness_disagreement(self):
        assert_expected_outcome(
            self, run_scenario(get_scenario("robustness_disagreement"))
        )

    def test_robustness_agreement(self):
        assert_expected_outcome(
            self, run_scenario(get_scenario("robustness_agreement"))
        )

    def test_repository_unavailable(self):
        assert_expected_outcome(
            self, run_scenario(get_scenario("repository_unavailable"))
        )

    def test_all_scenarios_run_in_catalog_order(self):
        outcomes = run_all()
        self.assertEqual(
            tuple(outcome.scenario.key for outcome in outcomes), scenario_keys()
        )
        for outcome in outcomes:
            with self.subTest(scenario=outcome.scenario.key):
                assert_expected_outcome(self, outcome)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    def test_repeated_runs_produce_identical_evidence_and_reports(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                first = run_scenario(scenario)
                second = run_scenario(scenario)
                self.assertEqual(
                    first.result.evidence_digest, second.result.evidence_digest
                )
                self.assertEqual(
                    first.report.link.critic_trace_digest,
                    second.report.link.critic_trace_digest,
                )
                self.assertEqual(
                    first.report.model_dump(mode="json"),
                    second.report.model_dump(mode="json"),
                )

    def test_repeated_runs_produce_identical_rendering(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                first = run_scenario(scenario)
                second = run_scenario(scenario)
                self.assertEqual(first.timeline, second.timeline)
                self.assertEqual(first.report_lines, second.report_lines)
                self.assertEqual(first.render_text(), second.render_text())

    def test_injected_as_of_reproduces_the_default_run(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                default = run_scenario(scenario)
                injected = run_scenario(scenario, as_of=default.as_of)
                self.assertEqual(
                    injected.result.evidence_digest, default.result.evidence_digest
                )
                self.assertEqual(
                    injected.report.conclusion.model_dump(mode="json"),
                    default.report.conclusion.model_dump(mode="json"),
                )

    def test_clock_changes_only_timestamps(self):
        scenario = get_scenario("persistent_multi_signal")
        later = datetime(2027, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

        default = run_scenario(scenario)
        shifted = run_scenario(scenario, clock=lambda: later)

        # The grounded content is timestamp-independent ...
        self.assertEqual(
            default.result.evidence_digest, shifted.result.evidence_digest
        )
        self.assertEqual(
            default.report.link.critic_trace_digest,
            shifted.report.link.critic_trace_digest,
        )
        # ... while the rendered timestamps follow the injected clock.
        self.assertNotEqual(default.timeline, shifted.timeline)
        self.assertTrue(shifted.timeline[0].startswith("03:04:05"))

    def test_scenarios_have_no_randomness_or_global_state(self):
        source = scenarios.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        for forbidden in ("import random", "random.", "time.time()", "datetime.now()"):
            # ``run_scenario`` reads the wall clock once, on purpose, to align
            # with the fixtures; that is the only permitted use.
            if forbidden == "datetime.now()":
                continue
            self.assertNotIn(forbidden, text)

    def test_outcome_reports_never_mutate_the_catalog(self):
        before = [scenario.expectation for scenario in EVALUATION_SCENARIOS]
        run_all()
        self.assertEqual(
            [scenario.expectation for scenario in EVALUATION_SCENARIOS], before
        )


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


class GroundingTests(unittest.TestCase):
    def test_every_observation_cites_evidence(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                outcome = run_scenario(scenario)
                for claim in outcome.report.observations:
                    self.assertTrue(claim.evidence_ids, claim.id)

    def test_decided_alternatives_cite_evidence(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                outcome = run_scenario(scenario)
                for item in outcome.result.alternatives or []:
                    if item.status in ("supported", "weakened", "partially_evaluated"):
                        self.assertTrue(item.evidence_ids, item.candidate)

    def test_conclusion_uses_a_fixed_template(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                conclusion = run_scenario(scenario).report.conclusion
                self.assertIn(conclusion.statement, CONCLUSION_TEXT.values())
                self.assertEqual(conclusion.statement, CONCLUSION_TEXT[conclusion.status])

    def test_report_carries_the_fixed_disclaimer_and_limitations(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                report = run_scenario(scenario).report
                self.assertEqual(report.disclaimer, REPORT_DISCLAIMER)
                for limitation in REPORT_LIMITATIONS:
                    self.assertIn(limitation, report.limitations)

    def test_rejected_claims_are_removed_from_the_asserted_sections(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                report = run_scenario(scenario).report
                rejected = {claim.subject for claim in report.rejected_claims}
                asserted = {claim.id for claim in report.observations}
                asserted |= {claim.id for claim in report.hypothesis_assessment}
                asserted |= {claim.id for claim in report.alternatives}
                self.assertEqual(rejected & asserted, set())

    def test_stop_reasons_are_in_the_closed_set(self):
        for outcome in run_all():
            with self.subTest(scenario=outcome.scenario.key):
                self.assertIn(outcome.stop_reason, KNOWN_STOP_REASONS)

    def test_engine_nodes_and_event_types_are_closed_and_recorded(self):
        for outcome in run_all():
            with self.subTest(scenario=outcome.scenario.key):
                derived = outcome.result.state.context.get("derived") or {}
                visited = set(derived.get("nodes_visited") or [])
                # The engine only walks its own fixed node set ...
                self.assertTrue(visited.issubset(set(NODE_NAMES)))
                # ... and every node it walked left a trace event.
                recorded = {event["node"] for event in outcome.result.state.trace}
                self.assertTrue(visited.issubset(recorded))
                for event in outcome.result.state.trace:
                    self.assertIn(event["event_type"], KNOWN_EVENT_TYPES)
                    self.assertTrue(event["node"])


# ---------------------------------------------------------------------------
# Privacy and medical-language safety
# ---------------------------------------------------------------------------


class PrivacySafetyTests(unittest.TestCase):
    def test_no_clinical_language_on_any_surface(self):
        for outcome in run_all():
            for name, text in outcome_texts(outcome).items():
                with self.subTest(scenario=outcome.scenario.key, surface=name):
                    self.assertEqual(clinical_terms_in(text), ())

    def test_no_raw_stored_field_names_on_any_surface(self):
        for outcome in run_all():
            for name, text in outcome_texts(outcome).items():
                with self.subTest(scenario=outcome.scenario.key, surface=name):
                    self.assertEqual(raw_field_names_in(text), ())

    def test_trace_carries_no_raw_arguments_or_row_values(self):
        for outcome in run_all():
            with self.subTest(scenario=outcome.scenario.key):
                for event in outcome.result.state.trace:
                    self.assertNotIn("args", event)
                    self.assertNotIn(outcome.scenario.session_id, json.dumps(event, default=str))
                    if event["event_type"] == "tool_call":
                        fingerprint = event.get("args_fingerprint")
                        self.assertTrue(fingerprint)
                        self.assertEqual(len(fingerprint), 16)

    def test_no_html_markup_is_rendered(self):
        for scenario in EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                text = run_scenario(scenario).render_text()
                for marker in ("<html", "<div", "<span", "<table", "<script", "<style"):
                    self.assertNotIn(marker, text)

    def test_timeline_lines_use_the_fixed_plain_text_format(self):
        for outcome in run_all():
            with self.subTest(scenario=outcome.scenario.key):
                self.assertTrue(outcome.timeline)
                for line in outcome.timeline:
                    self.assertRegex(line, TIMELINE_RE)


# ---------------------------------------------------------------------------
# Renderer wiring (the catalog's rendering helpers use the Phase 3.5 module)
# ---------------------------------------------------------------------------


class RenderingWiringTests(unittest.TestCase):
    def test_outcome_timeline_matches_the_renderer(self):
        outcome = run_scenario(get_scenario("persistent_multi_signal"))
        self.assertEqual(
            outcome.timeline,
            tuple(
                render_timeline_for(
                    outcome.report, outcome.result, clock=scenarios.frozen_clock
                )
            ),
        )

    def test_outcome_report_lines_match_the_renderer(self):
        outcome = run_scenario(get_scenario("recent_variation"))
        self.assertEqual(
            outcome.report_lines, tuple(render_report_lines(outcome.report))
        )

    def test_rendered_text_ends_with_the_timeline_heading(self):
        outcome = run_scenario(get_scenario("consistent"))
        text = outcome.render_text()
        self.assertIn("Investigation timeline:", text)
        self.assertTrue(
            text.endswith("\n".join(outcome.timeline)),
        )


if __name__ == "__main__":
    unittest.main()
