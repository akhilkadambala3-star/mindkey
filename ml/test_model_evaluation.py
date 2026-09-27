import unittest

from ml.model_evaluation import (
    build_training_sessions,
    evaluate_model,
)


class TestModelEvaluation(unittest.TestCase):

    def test_training_dataset_has_enough_sessions(self):

        training_sessions = (
            build_training_sessions()
        )

        self.assertGreaterEqual(
            len(training_sessions),
            10,
        )

    def test_evaluation_report_is_created(self):

        report = evaluate_model()

        self.assertIsInstance(
            report,
            dict,
        )

        self.assertIn(
            "metrics",
            report,
        )

    def test_evaluation_contains_expected_metrics(self):

        report = evaluate_model()

        metrics = report["metrics"]

        expected_metrics = {
            "precision",
            "recall",
            "f1_score",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
            "false_positive_rate",
        }

        self.assertTrue(
            expected_metrics.issubset(
                metrics.keys()
            )
        )

    def test_evaluation_dataset_size_is_correct(self):

        report = evaluate_model()

        self.assertEqual(
            report["evaluation_sessions"],
            report["normal_evaluation_sessions"]
            + report["anomalous_evaluation_sessions"],
        )

    def test_metrics_are_valid(self):

        report = evaluate_model()

        metrics = report["metrics"]

        for metric_name in [
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
        ]:

            self.assertGreaterEqual(
                metrics[metric_name],
                0.0,
            )

            self.assertLessEqual(
                metrics[metric_name],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()