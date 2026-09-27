import unittest

from ml.analysis import MLAnalysisEngine
from ml.anomaly_detector import AnomalyDetector
from ml.baseline import build_personal_baseline
from ml.decision import build_ml_assessment

from ml.test_baseline import (
    make_history,
    make_session,
)


class TestMLDecision(unittest.TestCase):

    def build_engine(self):

        history = make_history(20)

        detector = AnomalyDetector()
        detector.train(history)

        baseline = build_personal_baseline(
            history
        )

        return MLAnalysisEngine(
            anomaly_detector=detector,
            baseline=baseline,
        )

    def test_assessment_contains_main_sections(self):

        engine = self.build_engine()

        result = engine.analyze(
            make_session()
        )

        assessment = build_ml_assessment(
            result
        )

        self.assertEqual(
            assessment["status"],
            "ok",
        )

        self.assertTrue(
            assessment[
                "assessment_available"
            ]
        )

        self.assertIn(
            "anomaly",
            assessment,
        )

        self.assertIn(
            "pattern",
            assessment,
        )

        self.assertIn(
            "feature_evidence",
            assessment,
        )

    def test_assessment_contains_all_features(self):

        engine = self.build_engine()

        result = engine.analyze(
            make_session()
        )

        assessment = build_ml_assessment(
            result
        )

        feature_evidence = assessment[
            "feature_evidence"
        ]

        expected_features = {
            "dwell_mean",
            "flight_mean",
            "typing_speed",
            "correction_rate",
            "pause_count",
            "session_duration",
        }

        self.assertEqual(
            set(feature_evidence.keys()),
            expected_features,
        )

    def test_assessment_contains_pattern_summary(self):

        engine = self.build_engine()

        result = engine.analyze(
            make_session()
        )

        assessment = build_ml_assessment(
            result
        )

        pattern = assessment[
            "pattern"
        ]

        self.assertIn(
            "pattern_strength",
            pattern,
        )

        self.assertIn(
            "dominant_deviation",
            pattern,
        )

        self.assertIn(
            "dominant_direction",
            pattern,
        )

        self.assertIn(
            "strong_deviation_count",
            pattern,
        )

    def test_extreme_session_produces_high_pattern_strength(self):

        engine = self.build_engine()

        extreme_session = make_session(
            dwell_mean=0.80,
            flight_mean=0.90,
            typing_speed=5.0,
            correction_rate=0.90,
            pause_count=30,
            session_duration=300.0,
        )

        result = engine.analyze(
            extreme_session
        )

        assessment = build_ml_assessment(
            result
        )

        self.assertEqual(
            assessment["status"],
            "ok",
        )

        self.assertEqual(
            assessment["pattern"][
                "pattern_strength"
            ],
            "high",
        )

        self.assertGreaterEqual(
            assessment["pattern"][
                "strong_deviation_count"
            ],
            4,
        )

    def test_invalid_session_returns_unavailable_assessment(self):

        engine = self.build_engine()

        invalid_session = make_session(
            correction_rate=1.5
        )

        result = engine.analyze(
            invalid_session
        )

        assessment = build_ml_assessment(
            result
        )

        self.assertEqual(
            assessment["status"],
            "invalid_session",
        )

        self.assertFalse(
            assessment[
                "assessment_available"
            ]
        )

    def test_missing_anomaly_evidence_is_rejected(self):

        engine = self.build_engine()

        result = engine.analyze(
            make_session()
        )

        result.pop(
            "anomaly"
        )

        with self.assertRaises(
            ValueError
        ):

            build_ml_assessment(
                result
            )

    def test_missing_baseline_evidence_is_rejected(self):

        engine = self.build_engine()

        result = engine.analyze(
            make_session()
        )

        result.pop(
            "baseline"
        )

        with self.assertRaises(
            ValueError
        ):

            build_ml_assessment(
                result
            )


if __name__ == "__main__":
    unittest.main()