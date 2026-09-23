"""Tests for the deterministic build_behavioral_evidence adapter."""

import unittest

from investigation.adapter import build_behavioral_evidence
from investigation.bridge import MODEL_MINIMUM_SESSIONS
from investigation.units import CANONICAL_FEATURE_KEYS
from tests import fixtures


def _sufficient_stable_repo():
    """Enough valid sessions (>= minimum) across both comparison windows."""
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(1, 36)
    ]
    return fixtures.FakeRepository(sessions=sessions)


class DataQualityTests(unittest.TestCase):
    def test_insufficient_history_is_flagged(self):
        repo = fixtures.FakeRepository(
            sessions=[
                fixtures.make_session(session_id=f"S{i:04d}", days_ago=i) for i in range(3)
            ]
        )
        evidence = build_behavioral_evidence("user-1", "S0001", repository=repo)
        self.assertFalse(evidence.data_quality.meets_model_minimum)
        self.assertEqual(evidence.data_quality.model_minimum_sessions, MODEL_MINIMUM_SESSIONS)
        self.assertGreater(evidence.data_quality.valid_sessions, 0)

    def test_invalid_sessions_are_counted_separately(self):
        repo = fixtures.FakeRepository(
            sessions=[
                fixtures.make_session(session_id="A", days_ago=1),
                fixtures.make_invalid_session(session_id="B", days_ago=2),
            ]
        )
        evidence = build_behavioral_evidence("user-1", "A", repository=repo)
        self.assertEqual(evidence.data_quality.total_sessions, 2)
        self.assertEqual(evidence.data_quality.valid_sessions, 1)
        self.assertEqual(evidence.data_quality.invalid_sessions, 1)


class PersistenceTests(unittest.TestCase):
    def test_persistence_is_not_inferred_from_insufficient_data(self):
        repo = fixtures.FakeRepository(
            sessions=[
                fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=200.0)
                for i in range(3)
            ]
        )
        evidence = build_behavioral_evidence("user-1", "S0001", repository=repo)
        self.assertEqual(
            evidence.temporal_analysis.persistence.status, "insufficient_data"
        )

    def test_stable_history_is_stable(self):
        evidence = build_behavioral_evidence(
            "user-1", "S0001", repository=_sufficient_stable_repo()
        )
        self.assertEqual(evidence.temporal_analysis.persistence.status, "stable")

    def test_single_moved_signal_is_recent_variation(self):
        sessions = [
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
            for i in range(7, 36)
        ]
        # Only typing speed shifts in the recent window.
        sessions += [
            fixtures.make_session(session_id=f"R{i}", days_ago=i, typing_speed=200.0)
            for i in range(1, 6)
        ]
        evidence = build_behavioral_evidence(
            "user-1", "S0001", repository=fixtures.FakeRepository(sessions=sessions)
        )
        self.assertEqual(
            evidence.temporal_analysis.persistence.status, "recent_variation"
        )

    def test_multiple_aligned_signals_are_persistent_change(self):
        sessions = [
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
            for i in range(7, 36)
        ]
        sessions += [
            fixtures.make_session(
                session_id=f"R{i}",
                days_ago=i,
                typing_speed=200.0,      # decrease
                dwell_mean=0.20,         # increase
                flight_mean=0.14,        # increase
                rhythm_variability=0.30,  # increase
                correction_rate=0.12,    # increase
                pause_count=9,           # increase
            )
            for i in range(1, 6)
        ]
        evidence = build_behavioral_evidence(
            "user-1", "S0001", repository=fixtures.FakeRepository(sessions=sessions)
        )
        self.assertEqual(
            evidence.temporal_analysis.persistence.status, "persistent_change"
        )


class PrivacyAndShapeTests(unittest.TestCase):
    def test_signals_only_contain_canonical_keys(self):
        evidence = build_behavioral_evidence(
            "user-1", "S0001", repository=_sufficient_stable_repo()
        )
        self.assertEqual(
            [signal.key for signal in evidence.signals], list(CANONICAL_FEATURE_KEYS)
        )

    def test_evidence_dump_contains_no_free_text_fields(self):
        evidence = build_behavioral_evidence(
            "user-1", "S0001", repository=_sufficient_stable_repo()
        )
        dumped = evidence.model_dump()
        # No typed content or user-authored text may leak into the contract.
        for forbidden in ("text", "content", "message", "keystrokes", "note_text"):
            self.assertNotIn(forbidden, dumped)

    def test_units_are_documented_in_the_contract(self):
        evidence = build_behavioral_evidence(
            "user-1", "S0001", repository=_sufficient_stable_repo()
        )
        self.assertEqual(evidence.units.get("typing_speed_cpm"), "characters per minute")
        self.assertEqual(evidence.units.get("dwell_mean_s"), "seconds")


class ContextAndLimitationsTests(unittest.TestCase):
    def test_context_is_reported_as_unavailable(self):
        evidence = build_behavioral_evidence(
            "user-1", "S0001", repository=_sufficient_stable_repo()
        )
        self.assertEqual(evidence.context.source, "unavailable")
        self.assertEqual(evidence.context.checkins, [])
        self.assertIsNone(evidence.context.symptoms)

    def test_missing_ml_evidence_is_explicit(self):
        evidence = build_behavioral_evidence(
            "user-1", "S0001", repository=_sufficient_stable_repo()
        )
        # No anomaly fixture was supplied, so this must not be invented.
        self.assertEqual(evidence.uncertainty.level in ("moderate", "high", "low"), True)
        self.assertTrue(
            any("anomaly" in limitation.lower() for limitation in evidence.limitations)
        )

    def test_limitations_include_non_diagnostic_statement(self):
        evidence = build_behavioral_evidence(
            "user-1", "S0001", repository=_sufficient_stable_repo()
        )
        self.assertTrue(
            any("not a" in limitation.lower() for limitation in evidence.limitations)
        )


class RepositoryFailureTests(unittest.TestCase):
    def test_unavailable_repository_still_returns_evidence(self):
        evidence = build_behavioral_evidence(
            "user-1", "S0001", repository=fixtures.FailingRepository()
        )
        # The adapter must not crash; it reports what is unavailable.
        self.assertEqual(evidence.data_quality.valid_sessions, 0)
        self.assertFalse(evidence.data_quality.meets_model_minimum)
        self.assertEqual(
            evidence.temporal_analysis.persistence.status, "insufficient_data"
        )
        self.assertTrue(evidence.limitations)


if __name__ == "__main__":
    unittest.main()
