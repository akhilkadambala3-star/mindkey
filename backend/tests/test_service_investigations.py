"""Unit tests for the Phase 4 investigation service (``services.investigations``).

These tests pin three things: the JSON-safe contract payload, that every
substantive output stays authoritative to the frozen ``investigation.agent``
kernel (the Phase 3.5 scenario expectations), and the service boundaries —
determinism under an injected clock/as_of, privacy/safety of every surface,
strict JSON serializability, read-only access, and explicit failure modes.
"""

import json
import re
import typing
import unittest
from unittest import mock

from investigation.agent import (
    ENGINE_SCHEMA_VERSION,
    REPORT_DISCLAIMER,
    REPORT_SCHEMA_VERSION,
    StopReason,
)
from investigation.repository import RepositoryError
from services import investigations, reads
from services.investigations import SessionNotFound, run_for_user
from tests import fixtures
from tests.agent.scenarios import (
    USER,
    clinical_terms_in,
    get_scenario,
    raw_field_names_in,
)
from tests.api_support import (
    FROZEN_NOW,
    INVESTIGATION_KEYS,
    ReadOnlyRepository,
    frozen_clock,
)

#: Same fixed plain-text timeline format the Phase 3.5 renderer guarantees.
TIMELINE_RE = re.compile(r"^\d{2}:\d{2}:\d{2} \u2014 .+$")


def _run(key):
    """Run one catalog scenario through the Phase 4 service."""
    scenario = get_scenario(key)
    return run_for_user(
        USER,
        scenario.session_id,
        repository=scenario.repository_factory(),
        clock=frozen_clock,
    )


class PayloadContractTests(unittest.TestCase):
    def test_contract_keys_are_exact(self):
        payload = _run("persistent_multi_signal")
        self.assertEqual(frozenset(payload), INVESTIGATION_KEYS)
        self.assertTrue(payload["evidence_digest"])
        self.assertTrue(payload["report"])
        self.assertTrue(payload["timeline"])

    def test_payload_is_strictly_json_serializable(self):
        payload = _run("persistent_multi_signal")
        text = json.dumps(payload)  # no default=: anything else fails loudly
        self.assertIn('"disclaimer"', text)

    def test_schema_versions_come_from_the_kernel(self):
        payload = _run("consistent")
        self.assertEqual(payload["schema_version"], ENGINE_SCHEMA_VERSION)
        self.assertEqual(payload["report"]["schema_version"], REPORT_SCHEMA_VERSION)

    def test_stop_reason_is_in_the_frozen_vocabulary(self):
        vocabulary = set(typing.get_args(StopReason))
        for key in ("consistent", "insufficient_history"):
            with self.subTest(scenario=key):
                payload = _run(key)
                self.assertIn(payload["stop_reason"], vocabulary)

    def test_defaults_to_the_latest_stored_session(self):
        repo = fixtures.FakeRepository(sessions=fixtures.make_history(6))
        payload = run_for_user(USER, repository=repo, clock=frozen_clock)
        self.assertEqual(payload["session_id"], "S0006")
        self.assertEqual(
            payload["session_id"],
            reads.latest_session_id(USER, repository=repo),
        )

    def test_explicit_session_is_honored(self):
        repo = fixtures.FakeRepository(sessions=fixtures.make_history(5))
        payload = run_for_user(USER, "S0002", repository=repo, clock=frozen_clock)
        self.assertEqual(payload["session_id"], "S0002")

    def test_explicit_unknown_session_raises(self):
        repo = fixtures.FakeRepository(sessions=fixtures.make_history(3))
        with self.assertRaises(SessionNotFound):
            run_for_user(USER, "GHOST", repository=repo, clock=frozen_clock)

    def test_session_not_found_carries_the_requested_id(self):
        repo = fixtures.FakeRepository(sessions=fixtures.make_history(3))
        with self.assertRaises(SessionNotFound) as ctx:
            run_for_user(USER, "GHOST", repository=repo)
        self.assertEqual(ctx.exception.session_id, "GHOST")

    def test_a_session_belonging_to_another_user_is_not_found(self):
        repo = fixtures.FakeRepository(
            sessions=fixtures.make_history(3, user_id="someone-else")
        )
        with self.assertRaises(SessionNotFound):
            run_for_user(USER, "S0001", repository=repo)

    def test_empty_store_returns_a_reportless_honest_payload(self):
        repo = fixtures.FakeRepository(sessions=[])
        payload = run_for_user(USER, repository=repo, clock=frozen_clock)
        self.assertEqual(frozenset(payload), INVESTIGATION_KEYS)
        self.assertIsNone(payload["session_id"])
        self.assertIsNone(payload["report"])
        self.assertIsNone(payload["evidence_digest"])
        self.assertEqual(payload["timeline"], [])
        self.assertEqual(payload["limitations"], [])
        self.assertEqual(payload["stop_reason"], "evidence_insufficient")
        self.assertEqual(payload["disclaimer"], REPORT_DISCLAIMER)
        json.dumps(payload)  # still strictly JSON-safe

    def test_unknown_user_gets_the_same_honest_payload(self):
        repo = fixtures.FakeRepository(
            sessions=fixtures.make_history(3, user_id="someone-else")
        )
        payload = run_for_user(USER, repository=repo, clock=frozen_clock)
        self.assertIsNone(payload["report"])
        self.assertEqual(payload["stop_reason"], "evidence_insufficient")

    def test_repository_error_propagates(self):
        with self.assertRaises(RepositoryError):
            run_for_user(USER, repository=fixtures.FailingRepository())

    def test_default_repository_failure_propagates(self):
        with mock.patch.object(
            investigations,
            "default_repository",
            side_effect=RepositoryError("store down"),
        ):
            with self.assertRaises(RepositoryError):
                run_for_user(USER, "S0001")


class KernelAuthorityTests(unittest.TestCase):
    """The service must reproduce the Phase 3.5 scenario expectations exactly."""

    def _assert_expectation(self, key):
        payload = _run(key)
        expectation = get_scenario(key).expectation
        self.assertEqual(payload["stop_reason"], expectation.stop_reason)
        conclusion = payload["report"]["conclusion"]
        self.assertEqual(conclusion["status"], expectation.conclusion_status)
        self.assertEqual(conclusion["basis"], expectation.conclusion_basis)
        self.assertEqual(
            payload["report"]["uncertainty"]["level"],
            expectation.uncertainty_level,
        )

    def test_consistent_history_reports_no_deviation(self):
        self._assert_expectation("consistent")

    def test_recent_variation_stays_preliminary(self):
        self._assert_expectation("recent_variation")

    def test_persistent_multi_signal_change_is_grounded(self):
        self._assert_expectation("persistent_multi_signal")

    def test_invalid_rows_stay_a_data_quality_artifact(self):
        self._assert_expectation("invalid_rows")

    def test_window_disagreement_stays_preliminary(self):
        self._assert_expectation("robustness_disagreement")

    def test_insufficient_history_is_honest_and_inconclusive(self):
        payload = _run("insufficient_history")
        expectation = get_scenario("insufficient_history").expectation
        self.assertEqual(payload["stop_reason"], "evidence_insufficient")
        conclusion = payload["report"]["conclusion"]
        self.assertEqual(conclusion["status"], "inconclusive")
        self.assertEqual(conclusion["basis"], "insufficient_history")
        self.assertEqual(conclusion["basis"], expectation.conclusion_basis)
        self.assertEqual(payload["report"]["uncertainty"]["level"], "high")


class DeterminismTests(unittest.TestCase):
    def test_identical_requests_produce_identical_payloads(self):
        repo = get_scenario("persistent_multi_signal").repository_factory()
        first = run_for_user(
            USER, repository=repo, clock=frozen_clock, as_of=FROZEN_NOW
        )
        second = run_for_user(
            USER, repository=repo, clock=frozen_clock, as_of=FROZEN_NOW
        )
        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first), json.dumps(second))

    def test_all_trace_stamps_come_from_the_injected_clock(self):
        payload = _run("persistent_multi_signal")
        events = payload["report"]["critic_events"]
        self.assertTrue(events)
        for event in events:
            self.assertTrue(
                event["ts"].startswith("2026-09-24T09:42:00"), event["ts"]
            )


class PrivacyAndSafetyTests(unittest.TestCase):
    def test_no_clinical_language_anywhere_in_the_payload(self):
        for key in ("consistent", "persistent_multi_signal", "insufficient_history"):
            with self.subTest(scenario=key):
                blob = json.dumps(_run(key), default=str)
                self.assertEqual(clinical_terms_in(blob), ())

    def test_no_raw_stored_field_names_anywhere_in_the_payload(self):
        for key in ("consistent", "persistent_multi_signal", "insufficient_history"):
            with self.subTest(scenario=key):
                blob = json.dumps(_run(key), default=str)
                self.assertEqual(raw_field_names_in(blob), ())

    def test_disclaimer_is_the_fixed_kernel_notice(self):
        payload = _run("persistent_multi_signal")
        self.assertEqual(payload["disclaimer"], REPORT_DISCLAIMER)
        self.assertEqual(payload["report"]["disclaimer"], REPORT_DISCLAIMER)

    def test_report_trace_events_carry_no_raw_arguments(self):
        payload = _run("persistent_multi_signal")
        events = payload["report"]["critic_events"]
        self.assertTrue(events)
        for event in events:
            self.assertNotIn("args", event)

    def test_timeline_lines_follow_the_fixed_plain_text_format(self):
        payload = _run("persistent_multi_signal")
        self.assertTrue(payload["timeline"])
        for line in payload["timeline"]:
            self.assertRegex(line, TIMELINE_RE)


class ReadOnlyTests(unittest.TestCase):
    def test_only_protocol_reads_happen(self):
        repo = ReadOnlyRepository(
            get_scenario("consistent").repository_factory()
        )
        run_for_user(USER, repository=repo, clock=frozen_clock)
        allowed = {"list_sessions", "get_anomaly_result"}
        self.assertTrue(repo.calls)
        for call, _argument in repo.calls:
            self.assertIn(call, allowed)

    def test_injected_repository_never_falls_back_to_the_production_store(self):
        repo = get_scenario("consistent").repository_factory()
        with mock.patch.object(
            investigations,
            "default_repository",
            side_effect=AssertionError("production store must not be constructed"),
        ):
            payload = run_for_user(USER, repository=repo, clock=frozen_clock)
        self.assertEqual(payload["user_id"], USER)


if __name__ == "__main__":
    unittest.main()
