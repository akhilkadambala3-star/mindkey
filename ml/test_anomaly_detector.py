import unittest
import math
import tempfile
import joblib
import pytest
from ml.features import validate_features
import numpy as np

from ml.anomaly_detector import (
    AnomalyDetector,
    InsufficientDataError,
    MIN_TRAINING_SESSIONS,
)

from ml.features import (
    FEATURE_KEYS,
    is_valid_session,
)

def valid_features():
    return {
        "dwell_mean": 0.10,
        "flight_mean": 0.20,
        "typing_speed": 5.0,
        "correction_rate": 0.10,
        "pause_count": 5,
        "session_duration": 60.0,
    }

def make_session(
    dwell_mean=0.10,
    flight_mean=0.25,
    typing_speed=40.0,
    correction_rate=0.10,
    pause_count=3,
    session_duration=60.0,
):
    """Create a valid synthetic session."""

    return {
        "dwell_mean": dwell_mean,
        "flight_mean": flight_mean,
        "typing_speed": typing_speed,
        "correction_rate": correction_rate,
        "pause_count": pause_count,
        "session_duration": session_duration,
    }

def make_normal_sessions(count=20):
    """Create a small synthetic historical dataset."""

    sessions = []

    for index in range(count):

        sessions.append(
            make_session(
                dwell_mean=0.10 + (index % 3) * 0.005,
                flight_mean=0.25 + (index % 4) * 0.01,
                typing_speed=40.0 + (index % 5),
                correction_rate=0.10 + (index % 3) * 0.01,
                session_duration=60.0 + index,
                pause_count=3 + (index % 3),
            )
        )

    return sessions


OUTLIER_SESSION = make_session(
    dwell_mean=1.50,
    flight_mean=1.50,
    typing_speed=1.0,
    correction_rate=0.95,
    session_duration=300.0,
    pause_count=50,
)


class TestAnomalyDetector(unittest.TestCase):

    def test_insufficient_data(self):
        detector = AnomalyDetector()

        sessions = make_normal_sessions(
            MIN_TRAINING_SESSIONS - 1
        )

        with self.assertRaises(
            InsufficientDataError
        ):
            detector.train(sessions)

    def test_invalid_sessions_are_filtered(self):
        detector = AnomalyDetector()

        sessions = make_normal_sessions(12)

        sessions.append(
            {
                "dwell_mean": float("nan"),
                "flight_mean": 0.25,
                "typing_speed": 40.0,
                "correction_rate": 0.10,
                "session_duration": 60.0,
                "pause_count": 3,
            }
        )

        result = detector.train(
            sessions
        )

        self.assertEqual(
            result["sample_count"],
            12,
        )

    def test_training(self):
        detector = AnomalyDetector()

        sessions = make_normal_sessions(20)

        result = detector.train(
            sessions
        )

        self.assertEqual(
            result["status"],
            "trained",
        )

        self.assertEqual(
            result["sample_count"],
            20,
        )

    def test_all_feature_keys_are_validated(self):
        valid_session = make_session()

        self.assertTrue(
            is_valid_session(
                valid_session
            )
        )

        for feature_name in FEATURE_KEYS:

            broken = dict(
                valid_session
            )

            broken.pop(
                feature_name
            )

            self.assertFalse(
                is_valid_session(
                    broken
                )
            )

    def test_anomaly_score_ordering(self):
        detector = AnomalyDetector()

        sessions = make_normal_sessions(30)

        detector.train(
            sessions
        )

        normal_result = detector.evaluate(
            sessions[0]
        )

        outlier_result = detector.evaluate(
            OUTLIER_SESSION
        )

        self.assertEqual(
            normal_result["status"],
            "ok",
        )

        self.assertEqual(
            outlier_result["status"],
            "ok",
        )

        self.assertTrue(
        outlier_result["is_anomaly"]
        )

    def test_model_save_and_load(self):
        detector = AnomalyDetector()

        sessions = make_normal_sessions(20)

        detector.train(
            sessions
        )

        original_result = detector.evaluate(
            OUTLIER_SESSION
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = f"{temp_dir}/test_mindkey_model.joblib"

            detector.save(
                model_path
            )

            loaded_detector = AnomalyDetector.load(
                model_path
            )

            loaded_result = loaded_detector.evaluate(
                OUTLIER_SESSION
            )

        self.assertEqual(
            loaded_result["status"],
            "ok",
        )

        self.assertEqual(
            loaded_result["is_anomaly"],
            original_result["is_anomaly"],
        )

        self.assertAlmostEqual(
            loaded_result["anomaly_score"],
            original_result["anomaly_score"],
            places=10,
        )

        self.assertAlmostEqual(
            loaded_result["normalized_anomaly_score"],
            original_result["normalized_anomaly_score"],
            places=10,
        )

    def test_model_schema_mismatch_is_rejected(self):
        detector = AnomalyDetector()

        detector.train(
            make_normal_sessions(20)
        )

        model_path = "test_mindkey_model.joblib"

        detector.save(
            model_path
        )

        artifact = joblib.load(
            model_path
        )

        artifact["feature_keys"] = [
            "dwell_mean",
            "flight_mean",
            "typing_speed",
            "correction_rate",
            "pause_count",
            "wrong_feature",
        ]

        joblib.dump(
            artifact,
            model_path
        )

        with self.assertRaises(
            ValueError,
            msg="Schema mismatch should prevent model loading",
        ):
            AnomalyDetector.load(
                model_path
            )


    def test_normalized_score_is_between_zero_and_one(self):
        detector = AnomalyDetector()

        sessions = make_normal_sessions(30)

        detector.train(
            sessions
        )

        result = detector.evaluate(
            OUTLIER_SESSION
        )

        score = result[
            "normalized_anomaly_score"
        ]

        self.assertGreaterEqual(
            score,
            0.0,
        )

        self.assertLessEqual(
            score,
            1.0,
        )

    def test_invalid_session_status(self):
        detector = AnomalyDetector()

        detector.train(
            make_normal_sessions(20)
        )

        invalid_session = make_session(
            correction_rate=1.5
        )

        result = detector.evaluate(
            invalid_session
        )

        self.assertEqual(
            result["status"],
            "invalid_session",
        )

    def test_untrained_detector(self):
        detector = AnomalyDetector()

        result = detector.evaluate(
            make_session()
        )

        self.assertEqual(
            result["status"],
            "insufficient_data",
        )


if __name__ == "__main__":
    unittest.main()

def test_valid_feature_vector():
    features = valid_features()

    assert validate_features(features) is True

def test_missing_feature():
    features = valid_features()
    features.pop("dwell_mean")

    with pytest.raises(
        ValueError,
        match="Missing required features"
    ):
        validate_features(features)

def test_none_feature():
    features = valid_features()
    features["dwell_mean"] = None

    with pytest.raises(
        ValueError,
        match="dwell_mean cannot be None"
    ):
        validate_features(features)

def test_nan_feature():
    features = valid_features()
    features["dwell_mean"] = math.nan

    with pytest.raises(
        ValueError,
        match="dwell_mean must be finite"
    ):
        validate_features(features)


def test_positive_infinity():
    features = valid_features()
    features["flight_mean"] = math.inf

    with pytest.raises(
        ValueError,
        match="flight_mean must be finite"
    ):
        validate_features(features)

def test_negative_infinity():
    features = valid_features()
    features["flight_mean"] = -math.inf

    with pytest.raises(
        ValueError,
        match="flight_mean must be finite"
    ):
        validate_features(features)

def test_negative_dwell_mean():
    features = valid_features()
    features["dwell_mean"] = -0.1

    with pytest.raises(
        ValueError,
        match="dwell_mean must be >= 0"
    ):
        validate_features(features)

def test_negative_flight_mean():
    features = valid_features()
    features["flight_mean"] = -0.1

    with pytest.raises(
        ValueError,
        match="flight_mean must be >= 0"
    ):
        validate_features(features)

def test_zero_typing_speed():
    features = valid_features()
    features["typing_speed"] = 0.0

    with pytest.raises(
        ValueError,
        match="typing_speed must be >"
    ):
        validate_features(features)

def test_negative_typing_speed():
    features = valid_features()
    features["typing_speed"] = -1.0

    with pytest.raises(
        ValueError,
        match="typing_speed must be >"
    ):
        validate_features(features)

def test_negative_correction_rate():
    features = valid_features()
    features["correction_rate"] = -0.1

    with pytest.raises(
        ValueError,
        match="correction_rate must be between"
    ):
        validate_features(features)

def test_correction_rate_above_one():
    features = valid_features()
    features["correction_rate"] = 1.1

    with pytest.raises(
        ValueError,
        match="correction_rate must be between"
    ):
        validate_features(features)

def test_negative_pause_count():
    features = valid_features()
    features["pause_count"] = -1

    with pytest.raises(
        ValueError,
        match="pause_count must be >="
    ):
        validate_features(features)

def test_negative_session_duration():
    features = valid_features()
    features["session_duration"] = -10.0

    with pytest.raises(
        ValueError,
        match="session_duration must be >="
    ):
        validate_features(features)

def test_non_numeric_feature():
    features = valid_features()
    features["dwell_mean"] = "invalid"

    with pytest.raises(
        TypeError,
        match="dwell_mean must be numeric"
    ):
        validate_features(features)