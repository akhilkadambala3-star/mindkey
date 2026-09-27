"""
End-to-end evaluation of the MindKey anomaly detector.

This module connects:

    Evaluation dataset
        ↓
    AnomalyDetector
        ↓
    Predictions
        ↓
    Evaluation metrics

The evaluation uses controlled synthetic behavioral data.

The results measure statistical anomaly-detection behavior.
They do NOT establish medical validity or diagnostic capability.
"""

from ml.anomaly_detector import AnomalyDetector
from ml.evaluation import evaluate_anomaly_predictions
from ml.evaluation_dataset import (
    build_evaluation_dataset,
    make_normal_session,
)


def build_training_sessions():
    """
    Build a synthetic normal training dataset.

    The training data represents normal behavioral variation.
    It intentionally contains no labeled anomalies because
    Isolation Forest is an unsupervised anomaly detector.

    Returns
    -------
    list
        Synthetic normal sessions used to train the detector.
    """

    training_sessions = []

    for index in range(100):

        session = make_normal_session(
            dwell_mean=0.08 + (index % 10) * 0.005,
            flight_mean=0.20 + (index % 10) * 0.01,
            typing_speed=34.0 + (index % 13) * 1.0,
            correction_rate=0.05 + (index % 12) * 0.01,
            pause_count=1 + (index % 6),
            session_duration=50.0 + (index % 13) * 2.0,
        )

        training_sessions.append(session)

    return training_sessions

def evaluate_model():
    """
    Train and evaluate the MindKey anomaly detector.

    Returns
    -------
    dict
        Complete model evaluation report.
    """

    # ---------------------------------------------------------
    # STEP 1
    # Build normal training data.
    # ---------------------------------------------------------

    training_sessions = build_training_sessions()

    # ---------------------------------------------------------
    # STEP 2
    # Create the anomaly detector.
    #
    # IMPORTANT:
    # The existing AnomalyDetector constructor expects
    # configuration such as contamination/random_state.
    #
    # Training sessions are supplied separately to train().
    # ---------------------------------------------------------

    detector = AnomalyDetector()

    # ---------------------------------------------------------
    # STEP 3
    # Train the detector on the normal training sessions.
    # ---------------------------------------------------------

    training_result = detector.train(
        training_sessions
    )

    # ---------------------------------------------------------
    # STEP 4
    # Make sure training actually succeeded.
    # ---------------------------------------------------------

    if training_result.get("status") != "trained":

        raise RuntimeError(
            "Anomaly detector training failed: "
            f"{training_result}"
        )

    # ---------------------------------------------------------
    # STEP 5
    # Build the labeled evaluation dataset.
    #
    # IMPORTANT:
    # These sessions are separate from the training data.
    # ---------------------------------------------------------

    (
        normal_sessions,
        anomalous_sessions,
        y_true,
    ) = build_evaluation_dataset()

    evaluation_sessions = (
        normal_sessions
        + anomalous_sessions
    )

    # ---------------------------------------------------------
    # STEP 6
    # Generate predictions.
    #
    # Existing detector output:
    #
    #     is_anomaly = True  → 1
    #     is_anomaly = False → 0
    # ---------------------------------------------------------

    y_pred = []

    for session in evaluation_sessions:

        result = detector.evaluate(
            session
        )

        # -----------------------------------------------------
        # Evaluation should only continue if the detector
        # successfully analyzed the session.
        # -----------------------------------------------------

        if result.get("status") != "ok":

            raise RuntimeError(
                "Anomaly detector evaluation failed: "
                f"{result}"
            )

        if result["is_anomaly"]:

            y_pred.append(1)

        else:

            y_pred.append(0)

    # ---------------------------------------------------------
    # STEP 7
    # Calculate classification metrics.
    # ---------------------------------------------------------

    metrics = evaluate_anomaly_predictions(
        y_true,
        y_pred,
    )

    # ---------------------------------------------------------
    # STEP 8
    # Build the complete evaluation report.
    # ---------------------------------------------------------

    report = {
        "training_sessions": len(
            training_sessions
        ),

        "evaluation_sessions": len(
            evaluation_sessions
        ),

        "normal_evaluation_sessions": len(
            normal_sessions
        ),

        "anomalous_evaluation_sessions": len(
            anomalous_sessions
        ),

        "training_result": training_result,

        "metrics": metrics,
    }

    return report