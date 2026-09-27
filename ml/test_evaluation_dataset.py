import unittest

from ml.evaluation_dataset import (
    build_evaluation_dataset,
    make_normal_session,
    make_anomalous_session,
)


class TestEvaluationDataset(unittest.TestCase):

    def test_normal_session_contains_all_features(self):

        session = make_normal_session()

        expected_features = {
            "dwell_mean",
            "flight_mean",
            "typing_speed",
            "correction_rate",
            "pause_count",
            "session_duration",
        }

        self.assertEqual(
            set(session.keys()),
            expected_features,
        )

    def test_anomalous_session_contains_all_features(self):

        session = make_anomalous_session()

        expected_features = {
            "dwell_mean",
            "flight_mean",
            "typing_speed",
            "correction_rate",
            "pause_count",
            "session_duration",
        }

        self.assertEqual(
            set(session.keys()),
            expected_features,
        )

    def test_dataset_contains_normal_and_anomalous_sessions(self):

        (
            normal_sessions,
            anomalous_sessions,
            y_true,
        ) = build_evaluation_dataset()

        self.assertGreater(
            len(normal_sessions),
            0,
        )

        self.assertGreater(
            len(anomalous_sessions),
            0,
        )

        self.assertEqual(
            len(y_true),
            len(normal_sessions)
            + len(anomalous_sessions),
        )

    def test_labels_are_binary(self):

        (
            _,
            _,
            y_true,
        ) = build_evaluation_dataset()

        self.assertTrue(
            all(
                label in {0, 1}
                for label in y_true
            )
        )

    def test_dataset_has_balanced_classes(self):

        (
            normal_sessions,
            anomalous_sessions,
            y_true,
        ) = build_evaluation_dataset()

        self.assertEqual(
            y_true.count(0),
            len(normal_sessions),
        )

        self.assertEqual(
            y_true.count(1),
            len(anomalous_sessions),
        )

        self.assertEqual(
            len(normal_sessions),
            len(anomalous_sessions),
        )


if __name__ == "__main__":
    unittest.main()