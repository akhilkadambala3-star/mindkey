import unittest

from ml.evaluation import (
    evaluate_anomaly_predictions,
)


class TestAnomalyEvaluation(unittest.TestCase):

    def test_perfect_predictions(self):

        y_true = [
            0,
            0,
            1,
            1,
        ]

        y_pred = [
            0,
            0,
            1,
            1,
        ]

        result = evaluate_anomaly_predictions(
            y_true,
            y_pred,
        )

        self.assertEqual(
            result["precision"],
            1.0,
        )

        self.assertEqual(
            result["recall"],
            1.0,
        )

        self.assertEqual(
            result["f1_score"],
            1.0,
        )

        self.assertEqual(
            result["false_positive_rate"],
            0.0,
        )

    def test_false_positive_is_detected(self):

        y_true = [
            0,
            0,
            1,
            1,
        ]

        y_pred = [
            1,
            0,
            1,
            1,
        ]

        result = evaluate_anomaly_predictions(
            y_true,
            y_pred,
        )

        self.assertEqual(
            result["false_positives"],
            1,
        )

        self.assertEqual(
            result["true_positives"],
            2,
        )

    def test_false_negative_is_detected(self):

        y_true = [
            0,
            0,
            1,
            1,
        ]

        y_pred = [
            0,
            0,
            1,
            0,
        ]

        result = evaluate_anomaly_predictions(
            y_true,
            y_pred,
        )

        self.assertEqual(
            result["false_negatives"],
            1,
        )

        self.assertEqual(
            result["true_positives"],
            1,
        )

    def test_metric_values_are_between_zero_and_one(self):

        y_true = [
            0,
            1,
            0,
            1,
        ]

        y_pred = [
            0,
            0,
            1,
            1,
        ]

        result = evaluate_anomaly_predictions(
            y_true,
            y_pred,
        )

        metrics = [
            result["precision"],
            result["recall"],
            result["f1_score"],
            result["false_positive_rate"],
        ]

        for metric in metrics:

            self.assertGreaterEqual(
                metric,
                0.0,
            )

            self.assertLessEqual(
                metric,
                1.0,
            )

    def test_mismatched_lengths_are_rejected(self):

        with self.assertRaises(
            ValueError
        ):

            evaluate_anomaly_predictions(
                [0, 1],
                [0],
            )

    def test_empty_data_is_rejected(self):

        with self.assertRaises(
            ValueError
        ):

            evaluate_anomaly_predictions(
                [],
                [],
            )


if __name__ == "__main__":
    unittest.main()