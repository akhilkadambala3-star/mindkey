import unittest

from ml.analysis import MLAnalysisEngine
from ml.anomaly_detector import AnomalyDetector
from ml.baseline import build_personal_baseline

from ml.test_baseline import make_history, make_session


class TestMLAnalysisEngine(unittest.TestCase):

    def test_analysis_returns_combined_evidence(self):

        history = make_history(20)

        detector = AnomalyDetector()
        detector.train(history)

        baseline = build_personal_baseline(
            history
        )

        engine = MLAnalysisEngine(
            anomaly_detector=detector,
            baseline=baseline,
        )

        result = engine.analyze(
            history[0]
        )

        self.assertEqual(
            result["status"],
            "ok",
        )

        self.assertIn(
            "anomaly",
            result,
        )

        self.assertIn(
            "baseline",
            result,
        )

        self.assertIn(
            "feature_scores",
            result["baseline"],
        )

    def test_all_six_features_are_present(self):

        history = make_history(20)

        detector = AnomalyDetector()
        detector.train(history)

        baseline = build_personal_baseline(
            history
        )

        engine = MLAnalysisEngine(
            anomaly_detector=detector,
            baseline=baseline,
        )

        result = engine.analyze(
            history[0]
        )

        feature_scores = result[
            "baseline"
        ]["feature_scores"]

        expected_features = {
            "dwell_mean",
            "flight_mean",
            "typing_speed",
            "correction_rate",
            "pause_count",
            "session_duration",
        }

        self.assertEqual(
            set(feature_scores.keys()),
            expected_features,
        )

    def test_invalid_session_is_rejected(self):

        history = make_history(20)

        detector = AnomalyDetector()
        detector.train(history)

        baseline = build_personal_baseline(
            history
        )

        engine = MLAnalysisEngine(
            anomaly_detector=detector,
            baseline=baseline,
        )

        invalid_session = make_session(
            correction_rate=1.5
        )

        result = engine.analyze(
            invalid_session
        )

        self.assertEqual(
            result["status"],
            "invalid_session",
        )

    def test_extreme_session_produces_evidence(self):

        history = make_history(20)

        detector = AnomalyDetector()
        detector.train(history)

        baseline = build_personal_baseline(
            history
        )

        engine = MLAnalysisEngine(
            anomaly_detector=detector,
            baseline=baseline,
        )

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

        self.assertEqual(
            result["status"],
            "ok",
        )

        self.assertTrue(
            result["anomaly"]["is_anomaly"]
        )

        deviations = [
            feature_result[
                "absolute_deviation"
            ]
            for feature_result
            in result["baseline"][
                "feature_scores"
            ].values()
        ]

        self.assertGreater(
            max(deviations),
            3.0,
        )

    def test_analysis_contains_evidence_interpretation(self):

        history = make_history(20)

        detector = AnomalyDetector()
        detector.train(history)

        baseline = build_personal_baseline(
            history
        )

        engine = MLAnalysisEngine(
            anomaly_detector=detector,
            baseline=baseline,
        )

        result = engine.analyze(
            make_session()
        )

        feature_scores = result[
            "baseline"
        ]["feature_scores"]

        for feature_data in feature_scores.values():

            self.assertIn(
                "deviation_band",
                feature_data,
            )

            self.assertIn(
                "direction",
                feature_data,
            )

            self.assertIn(
                feature_data["deviation_band"],
                {
                    "normal",
                    "mild_deviation",
                    "moderate_deviation",
                    "strong_deviation",
                },
            )

            self.assertIn(
                feature_data["direction"],
                {
                    "above_baseline",
                    "below_baseline",
                    "at_baseline",
                },
            )

    def test_normal_session_produces_normal_evidence(self):

        history = make_history(20)

        detector = AnomalyDetector()
        detector.train(history)

        baseline = build_personal_baseline(
            history
        )

        engine = MLAnalysisEngine(
            anomaly_detector=detector,
            baseline=baseline,
        )

        normal_session = make_session(
            dwell_mean=0.105,
            flight_mean=0.265,
            typing_speed=42.0,
            correction_rate=0.11,
            pause_count=4,
            session_duration=69.5,
        )

        result = engine.analyze(
            normal_session
        )

        feature_scores = result[
            "baseline"
        ]["feature_scores"]

        normal_count = sum(
            feature_data["deviation_band"] == "normal"
            for feature_data
            in feature_scores.values()
        )

        self.assertGreaterEqual(
            normal_count,
            5,
        )

    def test_extreme_session_produces_strong_evidence(self):

        history = make_history(20)

        detector = AnomalyDetector()
        detector.train(history)

        baseline = build_personal_baseline(
            history
        )

        engine = MLAnalysisEngine(
            anomaly_detector=detector,
            baseline=baseline,
        )

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

        feature_scores = result[
            "baseline"
        ]["feature_scores"]

        strong_count = sum(
            feature_data["deviation_band"]
            == "strong_deviation"
            for feature_data
            in feature_scores.values()
        )

        self.assertGreaterEqual(
            strong_count,
            4,
        )

    def test_direction_matches_z_score(self):

        history = make_history(20)

        detector = AnomalyDetector()
        detector.train(history)

        baseline = build_personal_baseline(
            history
        )

        engine = MLAnalysisEngine(
            anomaly_detector=detector,
            baseline=baseline,
        )

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

        feature_scores = result[
            "baseline"
        ]["feature_scores"]

        for feature_data in feature_scores.values():

            z_score = feature_data["z_score"]
            direction = feature_data["direction"]

            if z_score > 0:

                self.assertEqual(
                    direction,
                    "above_baseline",
                )

            elif z_score < 0:

                self.assertEqual(
                    direction,
                    "below_baseline",
                )

            else:

                self.assertEqual(
                    direction,
                    "at_baseline",
                )


if __name__ == "__main__":
    unittest.main()