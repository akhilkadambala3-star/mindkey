"""
Behavioral evidence interpretation for MindKey.

Converts standardized baseline deviations into descriptive
behavioral deviation bands.

These bands describe statistical deviation only.
They are not medical thresholds or diagnoses.
"""

NORMAL_THRESHOLD = 1.0
MILD_THRESHOLD = 2.0
MODERATE_THRESHOLD = 3.0

def classify_deviation(absolute_deviation):
    """
    Classify the strength of a behavioral deviation.

    Returns
    -------
    str
        One of:
        - normal
        - mild_deviation
        - moderate_deviation
        - strong_deviation
    """

    if absolute_deviation < NORMAL_THRESHOLD:
        return "normal"

    elif absolute_deviation < MILD_THRESHOLD:
        return "mild_deviation"

    elif absolute_deviation < MODERATE_THRESHOLD:
        return "moderate_deviation"

    else:
        return "strong_deviation"

    """
    Classify the strength of a behavioral deviation.

    Parameters
    ----------
    absolute_deviation:
        Non-negative standardized deviation from the
        user's personal baseline.

    Returns
    -------
    str
        One of:
        - normal
        - mild_deviation
        - moderate_deviation
        - strong_deviation
    """
def interpret_feature_deviation(z_score, absolute_deviation):
    """
    Convert a feature's standardized deviation into
    descriptive behavioral evidence.
    """

    if z_score > 0:
        direction = "above_baseline"
    elif z_score < 0:
        direction = "below_baseline"
    else:
        direction = "at_baseline"

    return {
        "deviation_band": classify_deviation(absolute_deviation),
        "direction": direction,
    }