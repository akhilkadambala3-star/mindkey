"""
Evaluation utilities for the MindKey ML system.

Measures classification performance of anomaly detection
against labeled evaluation data.

This module evaluates statistical ML behavior.
It does not diagnose or infer medical conditions.
"""

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_anomaly_predictions(
    y_true,
    y_pred,
):
    """
    Evaluate anomaly predictions.

    Parameters
    ----------
    y_true : list
        Ground-truth labels.
        1 = anomaly
        0 = normal

    y_pred : list
        Predicted labels.
        1 = anomaly
        0 = normal

    Returns
    -------
    dict
        Classification metrics.
    """

    if len(y_true) != len(y_pred):

        raise ValueError(
            "y_true and y_pred must have "
            "the same length."
        )

    if len(y_true) == 0:

        raise ValueError(
            "Evaluation data cannot be empty."
        )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    total_normal = tn + fp

    if total_normal > 0:

        false_positive_rate = (
            fp / total_normal
        )

    else:

        false_positive_rate = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "false_positive_rate": (
            false_positive_rate
        ),
    }