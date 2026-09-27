"""Tests for the quantitative evaluation metrics (Checkpoint 9).

The metrics module (``investigation/agent/eval_metrics.py``) computes rates
and counts over run outcomes. These tests pin the exact formulas, the
denominators, the determinism, and the honest handling of edge cases, and
they verify the metrics against the committed scenario catalog.
"""

import json
import unittest
from datetime import datetime, timezone
from unittest import mock

from investigation.agent import (
    CONTRADICTION_SCENARIOS,
    FAILURE_SCENARIOS,
    GUARDRAIL_SCENARIOS,
    duplicate_tool_call_rate,
    evaluate_scenarios,
    failure_recovered,
    guardrail_triggered,
    contradiction_detected,
    tool_call_counts,
    traceability_rate,
    unsupported_claim_rate,
)
from investigation.agent.engine import InvestigationResult
from tests.agent import scenarios
from tests.agent.scenarios import (
    EVALUATION_SCENARIOS,
    get_scenario,
    run_all,
    run_scenario,
)

FROZEN_NOW = datetime(2026, 9, 24, 9, 42, tzinfo=timezone.utc)


def _outcome(key):
    return run_scenario(get_scenario(key))


class FormulaTests(unittest.TestCase):
    def test_unsupported_claim_rate_is_zero_for_sound_runs(self):
        for key in ("consistent", "persistent_multi_signal", "zero_sessions"):
            with self.subTest(scenario=key):
                outcome = _outcome(key)
                self.assertEqual(unsupported_claim_rate(outcome.report), 0.0)

    def test_unsupported_claim_rate_rejects_over_asserted_claims(self):
        class FakeReport:
            rejected_claims = [object(), object()]
            observations = [object() for _ in range(6)]
            hypothesis_assessment = [object() for _ in range(4)]
            alternatives = [object() for _ in range(3)]

        # 2 rejected / (2 + 6 + 4 + 3) examined = 2/15
        self.assertAlmostEqual(unsupported_claim_rate(FakeReport()), 2 / 15)

    def test_unsupported_claim_rate_of_nothing_is_zero(self):
        class Empty:
            rejected_claims = []
            observations = []
            hypothesis_assessment = []
            alternatives = []

        self.assertEqual(unsupported_claim_rate(Empty()), 0.0)

    def test_duplicate_tool_call_rate_is_zero_for_clean_runs(self):
        outcome = _outcome("consistent")
        self.assertEqual(duplicate_tool_call_rate(outcome.result), 0.0)

    def test_duplicate_tool_call_rate_counts_repeats(self):
        class FakeResult:
            state = mock.Mock()

        events = [
            {"event_type": "tool_call", "tool": "t", "args_fingerprint": "a"},
            {"event_type": "tool_call", "tool": "t", "args_fingerprint": "a"},
            {"event_type": "tool_call", "tool": "t", "args_fingerprint": "b"},
            {"event_type": "tool_result", "tool": "t", "args_fingerprint": "a"},
        ]
        FakeResult.state.trace = events
        # 1 repeat / 3 tool calls
        self.assertAlmostEqual(duplicate_tool_call_rate(FakeResult()), 1 / 3)

    def test_traceability_rate_is_one_for_catalog_runs(self):
        for outcome in run_all():
            with self.subTest(scenario=outcome.scenario.key):
                self.assertEqual(
                    traceability_rate(outcome.result, outcome.report), 1.0
                )

    def test_traceability_rate_ignores_undecided_claims(self):
        class Claim:
            def __init__(self, status, ids):
                self.status = status
                self.evidence_ids = ids

        class FakeReport:
            observations = []
            hypothesis_assessment = [Claim("uncertain", [])]
            alternatives = []

        class FakeResult:
            state = mock.Mock()

        FakeResult.state.evidence = [{"id": "E1"}]
        # The only decided denominator entry is empty -> defined as 1.0.
        self.assertEqual(traceability_rate(FakeResult(), FakeReport()), 1.0)


class CatalogPredicateTests(unittest.TestCase):
    def test_group_scenarios_are_subsets_of_the_catalog(self):
        catalog = set(scenarios.scenario_keys())
        for group in (CONTRADICTION_SCENARIOS, GUARDRAIL_SCENARIOS, FAILURE_SCENARIOS):
            self.assertTrue(set(group) <= catalog, group)

    def test_contradiction_detection_hits_both_conflict_scenarios(self):
        for key in CONTRADICTION_SCENARIOS:
            with self.subTest(scenario=key):
                self.assertTrue(contradiction_detected(_outcome(key)))

    def test_non_conflict_scenarios_do_not_trigger_the_predicate(self):
        for key in ("consistent", "recent_variation", "zero_sessions"):
            with self.subTest(scenario=key):
                self.assertFalse(contradiction_detected(_outcome(key)))

    def test_guardrail_predicate_hits_every_guardrail_scenario(self):
        for key in GUARDRAIL_SCENARIOS:
            with self.subTest(scenario=key):
                self.assertTrue(guardrail_triggered(_outcome(key)))

    def test_guardrail_predicate_ignores_ordinary_completions(self):
        for key in ("consistent", "recent_variation"):
            with self.subTest(scenario=key):
                self.assertFalse(guardrail_triggered(_outcome(key)))

    def test_failure_recovery_is_measured_for_the_outage(self):
        for key in FAILURE_SCENARIOS:
            with self.subTest(scenario=key):
                self.assertTrue(failure_recovered(_outcome(key)))

    def test_predicate_on_foreign_scenarios_is_false(self):
        foreign = mock.Mock()
        foreign.scenario = mock.Mock(key="not-in-catalog")
        self.assertFalse(contradiction_detected(foreign))
        self.assertFalse(guardrail_triggered(foreign))
        self.assertFalse(failure_recovered(foreign))


class AggregateTests(unittest.TestCase):
    def test_metrics_over_the_whole_catalog(self):
        metrics = evaluate_scenarios(run_all())

        self.assertEqual(metrics["schema_version"], "1.0")
        self.assertEqual(len(metrics["scenarios"]), len(EVALUATION_SCENARIOS))

        # Sound pipeline: nothing rejected anywhere.
        self.assertEqual(metrics["unsupported_claim_rate"], 0.0)
        # No duplicated tool calls anywhere.
        self.assertEqual(metrics["unnecessary_tool_call_rate"], 0.0)
        # Every decided claim resolves in the registry.
        self.assertEqual(metrics["traceability_rate"], 1.0)

        self.assertEqual(metrics["contradiction_detection_rate"], 1.0)
        self.assertEqual(metrics["contradiction_detection_detail"]["numerator"], 2)
        self.assertEqual(metrics["contradiction_detection_detail"]["denominator"], 2)

        self.assertEqual(metrics["guardrail_trigger_rate"], 1.0)
        self.assertEqual(metrics["guardrail_trigger_detail"]["numerator"], 5)
        self.assertEqual(metrics["guardrail_trigger_detail"]["denominator"], 5)

        self.assertEqual(metrics["failure_recovery_rate"], 1.0)
        self.assertEqual(metrics["failure_recovery_detail"]["numerator"], 1)

        counts = metrics["per_scenario_tool_calls"]
        self.assertEqual(counts["repository_unavailable"], 0)
        self.assertEqual(metrics["max_tool_calls"], max(counts.values()))
        self.assertEqual(metrics["min_tool_calls"], min(counts.values()))
        self.assertAlmostEqual(
            metrics["mean_tool_calls"],
            sum(counts.values()) / len(counts),
        )

    def test_metrics_are_json_serializable(self):
        text = json.dumps(evaluate_scenarios(run_all()))
        self.assertIn("unsupported_claim_rate", text)

    def test_metrics_are_deterministic(self):
        first = evaluate_scenarios(run_all())
        second = evaluate_scenarios(run_all())
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_subset_catalog_reports_honest_denominators(self):
        outcomes = [_outcome("consistent")]
        metrics = evaluate_scenarios(outcomes)
        # No guardrail/contradiction/failure scenarios present -> None, not 0.
        self.assertIsNone(metrics["guardrail_trigger_rate"])
        self.assertIsNone(metrics["contradiction_detection_rate"])
        self.assertIsNone(metrics["failure_recovery_rate"])
        self.assertEqual(metrics["guardrail_trigger_detail"]["denominator"], 0)
        self.assertEqual(metrics["guardrail_trigger_detail"]["numerator"], 0)
        self.assertEqual(metrics["guardrail_trigger_detail"]["scenarios_present"], [])

    def test_tool_call_counts_of_nothing_is_none(self):
        self.assertIsNone(tool_call_counts([])["mean_tool_calls"])
        self.assertEqual(tool_call_counts([])["per_scenario"], {})

    def test_single_scenario_tool_count(self):
        outcome = _outcome("persistent_multi_signal")
        counts = tool_call_counts([outcome])
        self.assertEqual(counts["per_scenario"]["persistent_multi_signal"], 2)
        self.assertEqual(counts["max_tool_calls"], 2)


if __name__ == "__main__":
    unittest.main()
