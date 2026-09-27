import unittest

from ml.patterns import aggregate_behavioral_pattern


class TestBehavioralPatternAggregation(unittest.TestCase):

    def test_all_normal_features_produce_low_strength(self):

        feature_scores = {
            "dwell_mean": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "flight_mean": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "typing_speed": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "correction_rate": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "pause_count": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "session_duration": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
        }

        result = aggregate_behavioral_pattern(
            feature_scores
        )

        self.assertEqual(
            result["total_features"],
            6,
        )

        self.assertEqual(
            result["pattern_strength"],
            "low",
        )

        self.assertEqual(
            result["strong_deviation_count"],
            0,
        )

    def test_multiple_strong_deviations_produce_high_strength(self):

        feature_scores = {
            "dwell_mean": {
                "deviation_band": "strong_deviation",
                "direction": "above_baseline",
            },
            "flight_mean": {
                "deviation_band": "strong_deviation",
                "direction": "above_baseline",
            },
            "typing_speed": {
                "deviation_band": "strong_deviation",
                "direction": "below_baseline",
            },
            "correction_rate": {
                "deviation_band": "strong_deviation",
                "direction": "above_baseline",
            },
            "pause_count": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "session_duration": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
        }

        result = aggregate_behavioral_pattern(
            feature_scores
        )

        self.assertEqual(
            result["strong_deviation_count"],
            4,
        )

        self.assertEqual(
            result["pattern_strength"],
            "high",
        )

        self.assertEqual(
            result["dominant_direction"],
            "above_baseline",
        )

    def test_moderate_pattern_is_detected(self):

        feature_scores = {
            "dwell_mean": {
                "deviation_band": "moderate_deviation",
                "direction": "above_baseline",
            },
            "flight_mean": {
                "deviation_band": "moderate_deviation",
                "direction": "above_baseline",
            },
            "typing_speed": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "correction_rate": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "pause_count": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "session_duration": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
        }

        result = aggregate_behavioral_pattern(
            feature_scores
        )

        self.assertEqual(
            result["moderate_deviation_count"],
            2,
        )

        self.assertEqual(
            result["pattern_strength"],
            "moderate",
        )

    def test_direction_counts_are_correct(self):

        feature_scores = {
            "dwell_mean": {
                "deviation_band": "strong_deviation",
                "direction": "above_baseline",
            },
            "flight_mean": {
                "deviation_band": "moderate_deviation",
                "direction": "above_baseline",
            },
            "typing_speed": {
                "deviation_band": "strong_deviation",
                "direction": "below_baseline",
            },
            "correction_rate": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "pause_count": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
            "session_duration": {
                "deviation_band": "normal",
                "direction": "at_baseline",
            },
        }

        result = aggregate_behavioral_pattern(
            feature_scores
        )

        self.assertEqual(
            result["direction_counts"][
                "above_baseline"
            ],
            2,
        )

        self.assertEqual(
            result["direction_counts"][
                "below_baseline"
            ],
            1,
        )

        self.assertEqual(
            result["direction_counts"][
                "at_baseline"
            ],
            3,
        )

    def test_invalid_deviation_band_is_rejected(self):

        feature_scores = {
            "dwell_mean": {
                "deviation_band": "unknown",
                "direction": "above_baseline",
            }
        }

        with self.assertRaises(ValueError):

            aggregate_behavioral_pattern(
                feature_scores
            )

    def test_invalid_direction_is_rejected(self):

        feature_scores = {
            "dwell_mean": {
                "deviation_band": "normal",
                "direction": "unknown",
            }
        }

        with self.assertRaises(ValueError):

            aggregate_behavioral_pattern(
                feature_scores
            )

    def test_empty_feature_scores_are_rejected(self):

        with self.assertRaises(ValueError):

            aggregate_behavioral_pattern({})


if __name__ == "__main__":
    unittest.main()