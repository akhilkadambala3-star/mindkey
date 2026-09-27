"""
Feature validation for the MindKey ML layer.

The feature extractor lives in ``agent/features.py``.

This module is responsible for validating the numerical feature vector
before it enters the ML pipeline.
"""

import math

from ml.feature_schema import FEATURE_NAMES

from ml.config import (
    MIN_TYPING_SPEED,
    MIN_CORRECTION_RATE,
    MAX_CORRECTION_RATE,
    MIN_PAUSE_COUNT,
    MIN_SESSION_DURATION,
    MIN_TIMING_VALUE,
)


# ----------------------------------------------------------------------
# Canonical feature order
# ----------------------------------------------------------------------

FEATURE_KEYS = FEATURE_NAMES


# ----------------------------------------------------------------------
# Phase 4 — Feature Validation
# ----------------------------------------------------------------------

def validate_features(features):
    """
    Validate the extracted feature vector before it reaches
    the anomaly detection model.
    """

    # ------------------------------------------------------------------
    # 1. Features must be a dictionary
    # ------------------------------------------------------------------

    if not isinstance(features, dict):
        raise TypeError(
            "Features must be provided as a dictionary"
        )

    # ------------------------------------------------------------------
    # 2. Check that all canonical features are present
    # ------------------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_KEYS
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # ------------------------------------------------------------------
    # 3–5. Check None, numeric type, NaN, and infinity
    # ------------------------------------------------------------------

    for feature in FEATURE_KEYS:

        value = features[feature]

        if value is None:
            raise ValueError(
                f"Invalid feature: {feature} cannot be None"
            )

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Invalid feature: {feature} must be numeric"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"Invalid feature: {feature} must be finite"
            )

    # ------------------------------------------------------------------
    # 6. Timing features must be >= 0
    # ------------------------------------------------------------------

    for feature in ["dwell_mean", "flight_mean"]:

        if features[feature] < MIN_TIMING_VALUE:
            raise ValueError(
                f"Invalid feature: {feature} must be >= "
                f"{MIN_TIMING_VALUE}"
            )

    # ------------------------------------------------------------------
    # 7. Typing speed must be > 0
    # ------------------------------------------------------------------

    if features["typing_speed"] <= MIN_TYPING_SPEED:
        raise ValueError(
            f"Invalid feature: typing_speed must be > "
            f"{MIN_TYPING_SPEED}"
        )

    # ------------------------------------------------------------------
    # 8. Correction rate must be between 0 and 1
    # ------------------------------------------------------------------

    correction_rate = features["correction_rate"]

    if not (
        MIN_CORRECTION_RATE
        <= correction_rate
        <= MAX_CORRECTION_RATE
    ):
        raise ValueError(
            "Invalid feature: correction_rate must be between "
            f"{MIN_CORRECTION_RATE} and {MAX_CORRECTION_RATE}"
        )

    # ------------------------------------------------------------------
    # 9. Pause count must be >= 0
    # ------------------------------------------------------------------

    if features["pause_count"] < MIN_PAUSE_COUNT:
        raise ValueError(
            f"Invalid feature: pause_count must be >= "
            f"{MIN_PAUSE_COUNT}"
        )

    # ------------------------------------------------------------------
    # 10. Session duration must be >= 0
    # ------------------------------------------------------------------

    if features["session_duration"] < MIN_SESSION_DURATION:
        raise ValueError(
            f"Invalid feature: session_duration must be >= "
            f"{MIN_SESSION_DURATION}"
        )

    return True


# ----------------------------------------------------------------------
# Backward Compatibility
# ----------------------------------------------------------------------

def is_valid_session(session):
    """
    Backward-compatible boolean validation for a session.

    Returns True when the feature dictionary passes validation.
    Returns False when validation fails.
    """

    try:
        validate_features(session)
        return True

    except (ValueError, TypeError):
        return False