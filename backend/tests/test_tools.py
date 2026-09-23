"""Tests for the deterministic read-only evidence tools."""

import unittest

from pydantic import ValidationError

from investigation import tools
from tests import fixtures


class RepositoryUnavailableTests(unittest.TestCase):
    def test_get_recent_sessions_degrades_gracefully(self):
        result = tools.get_recent_sessions(
            "user-1", repository=fixtures.FailingRepository()
        )
        self.assertFalse(result.available)
        self.assertEqual(result.unavailable_reason, "database unavailable")

    def test_get_ml_evidence_degrades_gracefully(self):
        result = tools.get_ml_evidence(
            "user-1", "S0001", repository=fixtures.FailingRepository()
        )
        self.assertFalse(result.available)

    def test_historical_baseline_degrades_gracefully(self):
        result = tools.get_historical_baseline(
            "user-1", repository=fixtures.FailingRepository()
        )
        self.assertFalse(result.available)

    def test_drift_degrades_gracefully(self):
        result = tools.calculate_behavioral_drift(
            "user-1", repository=fixtures.FailingRepository()
        )
        self.assertFalse(result.available)

    def test_compare_windows_degrades_gracefully(self):
        result = tools.compare_time_windows(
            "user-1", repository=fixtures.FailingRepository()
        )
        self.assertFalse(result.available)


class GetMLEvidenceTests(unittest.TestCase):
    def test_returns_stored_result(self):
        repo = fixtures.FakeRepository(
            sessions=[fixtures.make_session(session_id="S0001")],
            anomalies={"S0001": {"session_id": "S0001", "anomaly_score": 0.83, "is_anomaly": True}},
        )
        result = tools.get_ml_evidence("user-1", "S0001", repository=repo)
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.anomaly_score, 0.83)
        self.assertTrue(result.is_anomaly)

    def test_missing_record_is_reported_not_invented(self):
        repo = fixtures.FakeRepository(sessions=[fixtures.make_session()])
        result = tools.get_ml_evidence("user-1", "S0001", repository=repo)
        self.assertFalse(result.available)
        self.assertEqual(result.unavailable_reason, "no_anomaly_record")
        self.assertIsNone(result.anomaly_score)


class GetRecentSessionsTests(unittest.TestCase):
    def test_returns_canonical_units_oldest_first(self):
        repo = fixtures.FakeRepository(
            sessions=[
                fixtures.make_session(session_id="S0002", days_ago=1),
                fixtures.make_session(session_id="S0001", days_ago=5),
            ]
        )
        result = tools.get_recent_sessions("user-1", repository=repo)
        self.assertTrue(result.available)
        self.assertEqual([s.session_id for s in result.sessions], ["S0001", "S0002"])
        # Stored typing_speed is CPM and dwell is seconds; canonical names say so.
        self.assertAlmostEqual(result.sessions[0].typing_speed_cpm, 285.0)
        self.assertAlmostEqual(result.sessions[0].dwell_mean_s, 0.12)

    def test_limit_is_applied_to_most_recent(self):
        repo = fixtures.FakeRepository(
            sessions=[fixtures.make_session(session_id=f"S{i:04d}", days_ago=i) for i in range(10)]
        )
        result = tools.get_recent_sessions("user-1", limit=3, repository=repo)
        self.assertEqual(result.count, 3)
        # The three closest to now are days_ago 0, 1, 2 (returned oldest first).
        self.assertEqual(
            [s.session_id for s in result.sessions], ["S0002", "S0001", "S0000"]
        )

    def test_window_filters_older_sessions(self):
        repo = fixtures.FakeRepository(
            sessions=[
                fixtures.make_session(session_id="OLD", days_ago=40),
                fixtures.make_session(session_id="NEW", days_ago=2),
            ]
        )
        result = tools.get_recent_sessions("user-1", window_days=7, repository=repo)
        self.assertEqual([s.session_id for s in result.sessions], ["NEW"])

    def test_invalid_limit_is_rejected(self):
        with self.assertRaises(ValidationError):
            tools.get_recent_sessions("user-1", limit=0)

    def test_empty_user_has_no_sessions(self):
        repo = fixtures.FakeRepository(sessions=[fixtures.make_session(user_id="other")])
        result = tools.get_recent_sessions("user-1", repository=repo)
        self.assertTrue(result.available)
        self.assertEqual(result.count, 0)


class HistoricalBaselineTests(unittest.TestCase):
    def test_computes_mean_over_valid_sessions(self):
        repo = fixtures.FakeRepository(
            sessions=[
                fixtures.make_session(session_id="A", days_ago=2, typing_speed=200.0),
                fixtures.make_session(session_id="B", days_ago=3, typing_speed=300.0),
                fixtures.make_invalid_session(session_id="C", days_ago=4),
            ]
        )
        result = tools.get_historical_baseline("user-1", window_days=30, repository=repo)
        self.assertTrue(result.available)
        self.assertEqual(result.sample_count, 2)
        stats = {s.key: s for s in result.stats}
        self.assertAlmostEqual(stats["typing_speed_cpm"].mean, 250.0)
        self.assertEqual(stats["typing_speed_cpm"].canonical_unit, "characters per minute")

    def test_no_valid_sessions_is_unavailable(self):
        repo = fixtures.FakeRepository(sessions=[fixtures.make_invalid_session()])
        result = tools.get_historical_baseline("user-1", repository=repo)
        self.assertFalse(result.available)
        self.assertEqual(result.unavailable_reason, "no_valid_sessions_in_window")

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValidationError):
            tools.get_historical_baseline("user-1", window_days=0)


class DriftTests(unittest.TestCase):
    def _repo_with_speed_drop(self):
        baseline = [
            fixtures.make_session(session_id=f"B{i}", days_ago=10 + i, typing_speed=285.0)
            for i in range(10)
        ]
        recent = [
            fixtures.make_session(session_id=f"R{i}", days_ago=i, typing_speed=200.0)
            for i in range(5)
        ]
        return fixtures.FakeRepository(sessions=baseline + recent)

    def test_detects_relative_decrease(self):
        result = tools.calculate_behavioral_drift(
            "user-1", recent_window_days=7, baseline_window_days=30,
            repository=self._repo_with_speed_drop(),
        )
        self.assertTrue(result.available)
        self.assertTrue(result.sufficient)
        speed = next(f for f in result.features if f.key == "typing_speed_cpm")
        self.assertEqual(speed.direction, "decrease")
        self.assertLess(speed.relative_change, 0)
        self.assertAlmostEqual(speed.absolute_change, 200.0 - 285.0)

    def test_insufficient_when_windows_are_thin(self):
        repo = fixtures.FakeRepository(
            sessions=[fixtures.make_session(session_id="A", days_ago=1)]
        )
        result = tools.calculate_behavioral_drift("user-1", repository=repo)
        self.assertTrue(result.available)
        self.assertFalse(result.sufficient)

    def test_no_sessions_in_windows_is_unavailable(self):
        repo = fixtures.FakeRepository(
            sessions=[fixtures.make_session(session_id="A", days_ago=400)]
        )
        result = tools.calculate_behavioral_drift("user-1", repository=repo)
        self.assertFalse(result.available)

    def test_recent_and_baseline_windows_do_not_overlap(self):
        result = tools.calculate_behavioral_drift(
            "user-1", repository=self._repo_with_speed_drop()
        )
        # 5 recent sessions inside 7 days, none of the 10 baseline sessions.
        self.assertEqual(result.recent_sample_count, 5)
        self.assertEqual(result.baseline_sample_count, 10)


class CompareTimeWindowsTests(unittest.TestCase):
    def test_deltas_compare_recent_against_previous(self):
        repo = fixtures.FakeRepository(
            sessions=[
                fixtures.make_session(session_id=f"B{i}", days_ago=10 + i, typing_speed=285.0)
                for i in range(10)
            ]
            + [
                fixtures.make_session(session_id=f"R{i}", days_ago=i, typing_speed=200.0)
                for i in range(5)
            ]
        )
        result = tools.compare_time_windows(
            "user-1", window_a_days=7, window_b_days=30, repository=repo
        )
        self.assertTrue(result.available)
        self.assertEqual(result.window_a.label, "recent")
        self.assertEqual(result.window_b.label, "previous")
        speed = next(d for d in result.deltas if d.key == "typing_speed_cpm")
        self.assertEqual(speed.direction, "decrease")

    def test_empty_windows_is_unavailable(self):
        repo = fixtures.FakeRepository(
            sessions=[fixtures.make_session(session_id="A", days_ago=400)]
        )
        result = tools.compare_time_windows("user-1", repository=repo)
        self.assertFalse(result.available)


if __name__ == "__main__":
    unittest.main()
