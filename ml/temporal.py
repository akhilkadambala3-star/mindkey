"""
Temporal behavioral analysis for MindKey.

Analyzes ML assessments across multiple sessions to identify
statistical patterns over time.

This module does not diagnose or infer medical conditions.
It only summarizes temporal behavioral evidence.
"""


VALID_PATTERN_STRENGTHS = {
    "low",
    "moderate",
    "high",
}


def analyze_temporal_behavior(assessments):
    """
    Analyze a sequence of ML assessments.

    Parameters
    ----------
    assessments : list
        Ordered list of ML assessment dictionaries.
        The order must represent chronological order.

    Returns
    -------
    dict
        Temporal behavioral summary.
    """

    if not isinstance(
        assessments,
        list,
    ):

        raise TypeError(
            "assessments must be a list."
        )

    if not assessments:

        raise ValueError(
            "assessments cannot be empty."
        )

    valid_assessments = []

    for assessment in assessments:

        if not isinstance(
            assessment,
            dict,
        ):

            continue

        if assessment.get(
            "status"
        ) != "ok":

            continue

        if not assessment.get(
            "assessment_available",
            False,
        ):

            continue

        if "anomaly" not in assessment:
            continue

        if "pattern" not in assessment:
            continue

        valid_assessments.append(
            assessment
        )

    if not valid_assessments:

        return {
            "status": "no_valid_assessments",
            "sessions_analyzed": 0,
            "anomalous_sessions": 0,
            "anomaly_rate": 0.0,
            "strong_pattern_sessions": 0,
            "pattern_strength_trend": "unknown",
        }

    anomalous_sessions = sum(
        assessment["anomaly"]["is_anomaly"]
        for assessment
        in valid_assessments
    )

    strong_pattern_sessions = sum(
        assessment["pattern"][
            "pattern_strength"
        ] == "high"
        for assessment
        in valid_assessments
    )

    sessions_analyzed = len(
        valid_assessments
    )

    anomaly_rate = (
        anomalous_sessions
        / sessions_analyzed
    )

    pattern_strength_values = [
        assessment["pattern"][
            "pattern_strength"
        ]
        for assessment
        in valid_assessments
    ]

    pattern_strength_trend = (
        _determine_pattern_trend(
            pattern_strength_values
        )
    )

    return {
        "status": "ok",
        "sessions_analyzed": sessions_analyzed,
        "anomalous_sessions": anomalous_sessions,
        "anomaly_rate": anomaly_rate,
        "strong_pattern_sessions": strong_pattern_sessions,
        "pattern_strength_trend": (
            pattern_strength_trend
        ),
    }


def _determine_pattern_trend(
    pattern_strengths
):
    """
    Determine the directional trend of pattern strength.

    The sequence is interpreted chronologically.

    Returns
    -------
    str
        One of:
        - increasing
        - decreasing
        - stable
        - insufficient_data
    """

    if len(pattern_strengths) < 2:

        return "insufficient_data"

    strength_values = {
        "low": 0,
        "moderate": 1,
        "high": 2,
    }

    numeric_values = [
        strength_values[
            strength
        ]
        for strength
        in pattern_strengths
    ]

    first_value = numeric_values[0]
    last_value = numeric_values[-1]

    if last_value > first_value:

        return "increasing"

    if last_value < first_value:

        return "decreasing"

    return "stable"