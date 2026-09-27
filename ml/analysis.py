"""
Unified ML analysis layer for MindKey.

Combines:
    - Isolation Forest anomaly detection
    - Personal behavioral baseline scoring

This module does not diagnose or infer a medical condition.
It produces structured behavioral evidence for downstream systems.
"""

from ml.anomaly_detector import AnomalyDetector
from ml.baseline import PersonalBaseline
from ml.features import is_valid_session
from ml.evidence import interpret_feature_deviation


class MLAnalysisEngine:
    """
    Combines anomaly detection and personal baseline analysis.
    """

    def __init__(
        self,
        anomaly_detector: AnomalyDetector,
        baseline: PersonalBaseline,
    ):
        self.anomaly_detector = anomaly_detector
        self.baseline = baseline

    def analyze(self, session):
        """
        Analyze one validated session using both ML signals.

        Returns structured behavioral evidence.
        """

        if not is_valid_session(session):
            return {
                "status": "invalid_session",
                "message": "Session failed ML validation.",
            }

        anomaly_result = (
            self.anomaly_detector.evaluate(session)
        )

        baseline_result = (
            self.baseline.score_session(session)
        )

        for feature_name, feature_data in baseline_result.items():
            interpretation = interpret_feature_deviation(
                feature_data["z_score"],
                feature_data["absolute_deviation"],
            )

            feature_data.update(interpretation)
            
        return {
            "status": "ok",
            "anomaly": {
                "is_anomaly": anomaly_result["is_anomaly"],
                "anomaly_score": anomaly_result[
                    "anomaly_score"
                ],
                "normalized_anomaly_score": (
                    anomaly_result[
                        "normalized_anomaly_score"
                    ]
                ),
            },
            "baseline": {
                "feature_scores": baseline_result,
            },
        }