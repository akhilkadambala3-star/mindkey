"""Unit tests for the ML anomaly-detection layer (stdlib ``unittest`` only).

IMPORTANT: the synthetic sessions below are test fixtures used ONLY to
exercise the code paths of this module. They are NOT production training
data — the real model trains exclusively on valid historical sessions
stored per user.
"""

import random
import unittest

from anomaly_detector import (
    AnomalyDetector,
    InsufficientDataError,
    MIN_TRAINING_SESSIONS,
)
from features import FEATURE_KEYS, is_valid_session

SEED = 42


def make_session(**overrides):
    """A valid typing-session dict with typical mid-range values."""
    session = {
        "dwell_mean": 0.10,
        "flight_mean": 0.25,
        "typing_speed": 40.0,
        "correction_rate": 0.10,
        "rhythm_variability": 0.15,
        "pause_count": 3,
    }
    session.update(overrides)
    return session


def make_normal_sessions(count, seed=SEED):
    """Generate plausible "normal" sessions around a common typing style."""
    rng = random.Random(seed)
    sessions = []
    for _ in range(count):
        sessions.append(
            make_session(
                dwell_mean=rng.uniform(0.07, 0.14),
                flight_mean=rng.uniform(0.18, 0.35),
                typing_speed=rng.uniform(25.0, 55.0),
                correction_rate=rng.uniform(0.03, 0.18),
                rhythm_variability=rng.uniform(0.06, 0.25),
                pause_count=rng.randint(0, 8),
            )
        )
    return sessions


def feature_centroid(sessions):
    """Feature-wise mean of the given sessions (an in-distribution point)."""
    return {
        key: sum(s[key] for s in sessions) / len(sessions)
        for key in FEATURE_KEYS
    }


OUTLIER_SESSION = make_session(
    dwell_mean=1.5,
    flight_mean=3.0,
    typing_speed=1.0,
    correction_rate=0.9,
    rhythm_variability=2.0,
    pause_count=60,
)


class InsufficientDataTests(unittest.TestCase):
    def test_train_raises_with_current_test_users_three_sessions(self):
        """3 valid sessions (like the current test user) must not train."""
        detector = AnomalyDetector()
        sessions = make_normal_sessions(3)

        with self.assertRaises(InsufficientDataError):
            detector.train(sessions)

        self.assertFalse(detector.is_trained)
        self.assertEqual(detector.available_sessions, 3)

    def test_evaluate_reports_insufficient_data_status(self):
        """Before training, evaluate returns a clear insufficient-data status."""
        detector = AnomalyDetector()
        result = detector.evaluate(make_session())

        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["available_sessions"], 0)
        self.assertEqual(result["required_sessions"], MIN_TRAINING_SESSIONS)

    def test_invalid_sessions_are_filtered_out_of_training(self):
        """Invalid sessions must not count toward the training minimum."""
        detector = AnomalyDetector()
        sessions = make_normal_sessions(9) + [make_session(typing_speed=0.0)]

        with self.assertRaises(InsufficientDataError):
            detector.train(sessions)

        # The broken session was filtered out: only 9 valid sessions remain.
        self.assertEqual(detector.available_sessions, 9)


class TrainingTests(unittest.TestCase):
    def test_train_succeeds_with_enough_valid_sessions(self):
        detector = AnomalyDetector()
        sessions = make_normal_sessions(MIN_TRAINING_SESSIONS)

        result = detector.train(sessions)

        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["sample_count"], MIN_TRAINING_SESSIONS)
        self.assertTrue(detector.is_trained)

    def test_all_feature_keys_are_validated(self):
        for key in FEATURE_KEYS:
            self.assertTrue(is_valid_session(make_session()), key)
            self.assertFalse(is_valid_session(make_session(**{key: None})), key)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = AnomalyDetector()
        cls.training_sessions = make_normal_sessions(MIN_TRAINING_SESSIONS + 2)
        cls.detector.train(cls.training_sessions)

    def test_anomaly_score_is_higher_for_anomalous_session(self):
        """Corrected convention: HIGHER anomaly_score = MORE ANOMALOUS."""
        normal = feature_centroid(self.training_sessions)
        normal_result = self.detector.evaluate(normal)
        outlier_result = self.detector.evaluate(OUTLIER_SESSION)

        self.assertEqual(normal_result["status"], "ok")
        self.assertEqual(outlier_result["status"], "ok")

        self.assertGreater(
            outlier_result["anomaly_score"], normal_result["anomaly_score"]
        )
        self.assertGreater(
            outlier_result["anomaly_score_raw"], normal_result["anomaly_score_raw"]
        )
        self.assertTrue(outlier_result["is_anomaly"])
        self.assertFalse(normal_result["is_anomaly"])

    def test_anomaly_score_is_normalized_to_unit_interval(self):
        result = self.detector.evaluate(OUTLIER_SESSION)
        self.assertGreaterEqual(result["anomaly_score"], 0.0)
        self.assertLessEqual(result["anomaly_score"], 1.0)

    def test_invalid_session_gets_invalid_status(self):
        result = self.detector.evaluate(make_session(correction_rate=1.5))
        self.assertEqual(result["status"], "invalid_session")


if __name__ == "__main__":
    unittest.main()