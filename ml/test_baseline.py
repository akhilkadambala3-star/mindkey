import unittest

from ml.baseline import (
    BaselineNotReadyError,
    build_personal_baseline,
)


def make_session(
    dwell_mean=0.10,
    flight_mean=0.25,
    typing_speed=40.0,
    correction_rate=0.10,
    pause_count=3,
    session_duration=60.0,
):
    return {
        "dwell_mean": dwell_mean,
        "flight_mean": flight_mean,
        "typing_speed": typing_speed,
        "correction_rate": correction_rate,
        "pause_count": pause_count,
        "session_duration": session_duration,
    }

def make_history(count=20):
    sessions = []

    for index in range(count):

        sessions.append(
            make_session(
                dwell_mean=0.10 + (index % 3) * 0.005,
                flight_mean=0.25 + (index % 4) * 0.01,
                typing_speed=40.0 + (index % 5),
                correction_rate=0.10 + (index % 3) * 0.01,
                pause_count=3 + (index % 3),
                session_duration=60.0 + index,
            )
        )

    return sessions

class TestPersonalBaseline(unittest.TestCase):
    def test_constant_feature_does_not_cause_division_by_zero(self):

        history = make_history(20)

        for session in history:
            session["typing_speed"] = 40.0

        baseline = build_personal_baseline(
            history
        )

        result = baseline.score_session(
            history[0]
        )

        self.assertEqual(
            result["typing_speed"]["z_score"],
            0.0,
        )

        self.assertEqual(
            result["typing_speed"]["absolute_deviation"],
            0.0,
        )

    def test_insufficient_history(self):

        with self.assertRaises(
            BaselineNotReadyError
        ):
            build_personal_baseline(
                make_history(5),
                minimum_sessions=10,
            )

    def test_baseline_contains_all_features(self):

        baseline = build_personal_baseline(
            make_history(20)
        )

        result = baseline.describe()

        self.assertEqual(
            result["sample_count"],
            20,
        )

        self.assertEqual(
            set(result["features"].keys()),
            {
            "dwell_mean",
            "flight_mean",
            "typing_speed",
            "correction_rate",
            "pause_count",
            "session_duration",
        },
        )

    def test_feature_statistics_exist(self):

        baseline = build_personal_baseline(
            make_history(20)
        )

        for feature_name, stats in (
            baseline.feature_statistics.items()
        ):

            self.assertIn(
                "mean",
                stats,
            )

            self.assertIn(
                "median",
                stats,
            )

            self.assertIn(
                "std",
                stats,
            )

            self.assertIn(
                "mad",
                stats,
            )

            self.assertIn(
                "min",
                stats,
            )

            self.assertIn(
                "max",
                stats,
            )

    def test_normal_session_has_small_deviation(self):

        history = make_history(20)

        baseline = build_personal_baseline(
            history
        )

        result = baseline.score_session(
            history[0]
        )

        for feature_result in result.values():

            self.assertLess(
                feature_result["absolute_deviation"],
                5.0,
            )

    def test_extreme_session_has_larger_deviation(self):

        history = make_history(20)

        baseline = build_personal_baseline(
            history
        )

        extreme_session = make_session(
        dwell_mean=0.80,
        flight_mean=0.90,
        typing_speed=5.0,
        correction_rate=0.90,
        pause_count=30,
        session_duration=300.0,
        )

        result = baseline.score_session(
            extreme_session
        )

        large_deviations = sum(
            1
            for feature_result in result.values()
            if feature_result["absolute_deviation"] > 3.0
        )

        self.assertGreater(
            large_deviations,
            0,
        )


if __name__ == "__main__":
    unittest.main()