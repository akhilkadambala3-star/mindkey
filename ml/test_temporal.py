import unittest

from ml.temporal import (
    analyze_temporal_behavior,
)


def make_assessment(
    is_anomaly=False,
    pattern_strength="low",
):

    return {
        "status": "ok",
        "assessment_available": True,

        "anomaly": {
            "is_anomaly": is_anomaly,
        },

        "pattern": {
            "pattern_strength": pattern_strength,
        },
    }


class TestTemporalBehavior(unittest.TestCase):

    def test_all_normal_sessions_are_stable(self):

        assessments = [
            make_assessment(
                is_anomaly=False,
                pattern_strength="low",
            )
            for _ in range(5)
        ]

        result = analyze_temporal_behavior(
            assessments
        )

        self.assertEqual(
            result["status"],
            "ok",
        )

        self.assertEqual(
            result["sessions_analyzed"],
            5,
        )

        self.assertEqual(
            result["anomalous_sessions"],
            0,
        )

        self.assertEqual(
            result["anomaly_rate"],
            0.0,
        )

        self.assertEqual(
            result["strong_pattern_sessions"],
            0,
        )

        self.assertEqual(
            result["pattern_strength_trend"],
            "stable",
        )

    def test_anomaly_rate_is_calculated(self):

        assessments = [
            make_assessment(
                is_anomaly=True,
                pattern_strength="high",
            ),
            make_assessment(
                is_anomaly=False,
                pattern_strength="low",
            ),
            make_assessment(
                is_anomaly=True,
                pattern_strength="moderate",
            ),
            make_assessment(
                is_anomaly=False,
                pattern_strength="low",
            ),
        ]

        result = analyze_temporal_behavior(
            assessments
        )

        self.assertEqual(
            result["sessions_analyzed"],
            4,
        )

        self.assertEqual(
            result["anomalous_sessions"],
            2,
        )

        self.assertEqual(
            result["anomaly_rate"],
            0.5,
        )

    def test_strong_pattern_sessions_are_counted(self):

        assessments = [
            make_assessment(
                pattern_strength="high"
            ),
            make_assessment(
                pattern_strength="high"
            ),
            make_assessment(
                pattern_strength="moderate"
            ),
            make_assessment(
                pattern_strength="low"
            ),
        ]

        result = analyze_temporal_behavior(
            assessments
        )

        self.assertEqual(
            result["strong_pattern_sessions"],
            2,
        )

    def test_increasing_pattern_trend(self):

        assessments = [
            make_assessment(
                pattern_strength="low"
            ),
            make_assessment(
                pattern_strength="moderate"
            ),
            make_assessment(
                pattern_strength="high"
            ),
        ]

        result = analyze_temporal_behavior(
            assessments
        )

        self.assertEqual(
            result["pattern_strength_trend"],
            "increasing",
        )

    def test_decreasing_pattern_trend(self):

        assessments = [
            make_assessment(
                pattern_strength="high"
            ),
            make_assessment(
                pattern_strength="moderate"
            ),
            make_assessment(
                pattern_strength="low"
            ),
        ]

        result = analyze_temporal_behavior(
            assessments
        )

        self.assertEqual(
            result["pattern_strength_trend"],
            "decreasing",
        )

    def test_stable_pattern_trend(self):

        assessments = [
            make_assessment(
                pattern_strength="moderate"
            ),
            make_assessment(
                pattern_strength="moderate"
            ),
            make_assessment(
                pattern_strength="moderate"
            ),
        ]

        result = analyze_temporal_behavior(
            assessments
        )

        self.assertEqual(
            result["pattern_strength_trend"],
            "stable",
        )

    def test_single_assessment_has_insufficient_data(self):

        assessments = [
            make_assessment(
                pattern_strength="high"
            )
        ]

        result = analyze_temporal_behavior(
            assessments
        )

        self.assertEqual(
            result["pattern_strength_trend"],
            "insufficient_data",
        )

    def test_invalid_assessments_are_ignored(self):

        assessments = [
            make_assessment(
                pattern_strength="low"
            ),
            {
                "status": "invalid_session",
                "assessment_available": False,
            },
            make_assessment(
                pattern_strength="high"
            ),
        ]

        result = analyze_temporal_behavior(
            assessments
        )

        self.assertEqual(
            result["sessions_analyzed"],
            2,
        )

    def test_no_valid_assessments(self):

        assessments = [
            {
                "status": "invalid_session",
                "assessment_available": False,
            }
        ]

        result = analyze_temporal_behavior(
            assessments
        )

        self.assertEqual(
            result["status"],
            "no_valid_assessments",
        )

        self.assertEqual(
            result["sessions_analyzed"],
            0,
        )

        self.assertEqual(
            result["anomaly_rate"],
            0.0,
        )

        self.assertEqual(
            result["pattern_strength_trend"],
            "unknown",
        )

    def test_empty_assessments_are_rejected(self):

        with self.assertRaises(
            ValueError
        ):

            analyze_temporal_behavior(
                []
            )


if __name__ == "__main__":
    unittest.main()