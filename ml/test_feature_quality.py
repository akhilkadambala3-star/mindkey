import unittest

from ml.feature_quality import (
    analyze_feature_quality,
    summarize_session_quality,
)


def make_session():
    """
    Create a valid session using the canonical
    six-feature MindKey schema.
    """

    return {
        "dwell_mean": 0.10,
        "flight_mean": 0.25,
        "typing_speed": 40.0,
        "correction_rate": 0.10,
        "pause_count": 3,
        "session_duration": 60.0,
    }


class TestFeatureQuality(unittest.TestCase):

    def test_complete_dataset_has_full_validity(self):

        sessions = [
            make_session()
            for _ in range(10)
        ]

        result = summarize_session_quality(
            sessions
        )

        self.assertEqual(
            result["total_sessions"],
            10,
        )

        self.assertEqual(
            result["valid_sessions"],
            10,
        )

        self.assertEqual(
            result["invalid_sessions"],
            0,
        )

        self.assertEqual(
            result["valid_rate"],
            1.0,
        )

    def test_feature_quality_report(self):

        sessions = [
            make_session()
            for _ in range(10)
        ]

        report = analyze_feature_quality(
            sessions
        )

        self.assertEqual(
            len(report),
            6,
        )

        for feature_name, stats in (
            report.items()
        ):

            self.assertEqual(
                stats["valid_count"],
                10,
            )

            self.assertEqual(
                stats["missing_count"],
                0,
            )

            self.assertEqual(
                stats["invalid_count"],
                0,
            )

            self.assertEqual(
                stats["valid_rate"],
                1.0,
            )

    def test_missing_feature_is_detected(self):

        sessions = [
            make_session()
            for _ in range(5)
        ]

        sessions[0].pop(
            "typing_speed"
        )

        report = analyze_feature_quality(
            sessions
        )

        self.assertEqual(
            report["typing_speed"]["missing_count"],
            1,
        )

        self.assertEqual(
            report["typing_speed"]["valid_count"],
            4,
        )

    def test_invalid_feature_is_detected(self):

        sessions = [
            make_session()
            for _ in range(5)
        ]

        sessions[0]["correction_rate"] = 1.5

        report = analyze_feature_quality(
            sessions
        )

        self.assertEqual(
            report["correction_rate"]["invalid_count"],
            1,
        )

        self.assertEqual(
            report["correction_rate"]["valid_count"],
            4,
        )

    def test_empty_dataset_has_zero_valid_rate(self):

        result = summarize_session_quality(
            []
        )

        self.assertEqual(
            result["total_sessions"],
            0,
        )

        self.assertEqual(
            result["valid_sessions"],
            0,
        )

        self.assertEqual(
            result["invalid_sessions"],
            0,
        )

        self.assertEqual(
            result["valid_rate"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()