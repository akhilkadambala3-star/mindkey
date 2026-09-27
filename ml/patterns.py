"""
Behavioral pattern aggregation for MindKey.

Aggregates feature-level statistical evidence into
higher-level behavioral patterns.

This module does not diagnose or infer medical conditions.
It only summarizes patterns already detected by the ML layer.
"""


DEVIATION_BANDS = {
    "normal",
    "mild_deviation",
    "moderate_deviation",
    "strong_deviation",
}

DIRECTIONS = {
    "above_baseline",
    "below_baseline",
    "at_baseline",
}


def aggregate_behavioral_pattern(feature_scores):
    """
    Aggregate feature-level evidence into a behavioral pattern summary.

    Parameters
    ----------
    feature_scores : dict
        Dictionary containing feature-level baseline evidence.

    Returns
    -------
    dict
        Aggregated behavioral pattern evidence.
    """

    if not isinstance(feature_scores, dict):
        raise TypeError(
            "feature_scores must be a dictionary."
        )

    if not feature_scores:
        raise ValueError(
            "feature_scores cannot be empty."
        )

    deviation_counts = {
        "normal": 0,
        "mild_deviation": 0,
        "moderate_deviation": 0,
        "strong_deviation": 0,
    }

    direction_counts = {
        "above_baseline": 0,
        "below_baseline": 0,
        "at_baseline": 0,
    }

    for feature_name, feature_data in feature_scores.items():

        if not isinstance(feature_data, dict):
            raise TypeError(
                f"Evidence for '{feature_name}' "
                "must be a dictionary."
            )

        deviation_band = feature_data.get(
            "deviation_band"
        )

        direction = feature_data.get(
            "direction"
        )

        if deviation_band not in DEVIATION_BANDS:
            raise ValueError(
                f"Invalid deviation band for "
                f"'{feature_name}': {deviation_band}"
            )

        if direction not in DIRECTIONS:
            raise ValueError(
                f"Invalid direction for "
                f"'{feature_name}': {direction}"
            )

        deviation_counts[
            deviation_band
        ] += 1

        direction_counts[
            direction
        ] += 1

    strong_count = deviation_counts[
        "strong_deviation"
    ]

    moderate_count = deviation_counts[
        "moderate_deviation"
    ]

    total_features = len(feature_scores)

    if strong_count >= 3:

        pattern_strength = "high"

    elif (
        strong_count >= 1
        or moderate_count >= 2
    ):

        pattern_strength = "moderate"

    else:

        pattern_strength = "low"

    dominant_deviation = max(
        deviation_counts,
        key=deviation_counts.get,
    )

    dominant_direction = max(
        direction_counts,
        key=direction_counts.get,
    )

    return {
        "total_features": total_features,

        "deviation_counts": deviation_counts,

        "direction_counts": direction_counts,

        "strong_deviation_count": strong_count,

        "moderate_deviation_count": moderate_count,

        "dominant_deviation": dominant_deviation,

        "dominant_direction": dominant_direction,

        "pattern_strength": pattern_strength,
    }