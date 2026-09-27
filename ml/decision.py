"""
ML decision and evidence layer for MindKey.

Combines the outputs of the ML analysis pipeline into
one structured assessment that can be consumed by
downstream systems such as the AI agent.

This module summarizes statistical evidence.

It does not diagnose or infer medical conditions.
"""


from ml.patterns import aggregate_behavioral_pattern


VALID_PATTERN_STRENGTHS = {
    "low",
    "moderate",
    "high",
}


def build_ml_assessment(analysis_result):
    """
    Build a structured ML assessment from an analysis result.

    Parameters
    ----------
    analysis_result : dict
        Output produced by MLAnalysisEngine.analyze().

    Returns
    -------
    dict
        Structured ML assessment.
    """

    if not isinstance(analysis_result, dict):

        raise TypeError(
            "analysis_result must be a dictionary."
        )

    status = analysis_result.get(
        "status"
    )

    if status != "ok":

        return {
            "status": status,
            "assessment_available": False,
            "message": analysis_result.get(
                "message",
                "ML assessment unavailable.",
            ),
        }

    if "anomaly" not in analysis_result:

        raise ValueError(
            "Analysis result is missing anomaly evidence."
        )

    if "baseline" not in analysis_result:

        raise ValueError(
            "Analysis result is missing baseline evidence."
        )

    feature_scores = analysis_result[
        "baseline"
    ].get(
        "feature_scores"
    )

    if not isinstance(
        feature_scores,
        dict,
    ):

        raise ValueError(
            "Baseline feature scores are missing or invalid."
        )

    pattern = aggregate_behavioral_pattern(
        feature_scores
    )

    anomaly = analysis_result[
        "anomaly"
    ]

    return {
        "status": "ok",

        "assessment_available": True,

        "anomaly": {
            "is_anomaly": anomaly[
                "is_anomaly"
            ],
            "anomaly_score": anomaly[
                "anomaly_score"
            ],
            "normalized_anomaly_score": anomaly[
                "normalized_anomaly_score"
            ],
        },

        "pattern": pattern,

        "feature_evidence": feature_scores,
    }
