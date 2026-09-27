"""
Canonical behavioral feature definitions for MindKey.

This module defines the single source of truth for the six behavioral
features produced by the MindKey typing feature extractor.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FeatureDefinition:
    """
    Definition of one behavioral feature.
    """

    name: str
    unit: str
    description: str


# ----------------------------------------------------------------------
# Canonical feature definitions
# ----------------------------------------------------------------------

FEATURE_DEFINITIONS: Tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        name="dwell_mean",
        unit="seconds",
        description="Mean duration for which a key is held.",
    ),
    FeatureDefinition(
        name="flight_mean",
        unit="seconds",
        description=(
            "Mean inter-key interval excluding pauses "
            "above the configured pause threshold."
        ),
    ),
    FeatureDefinition(
        name="typing_speed",
        unit="characters_per_second",
        description=(
            "Estimated typing speed based on the typing session."
        ),
    ),
    FeatureDefinition(
        name="correction_rate",
        unit="ratio",
        description=(
            "Fraction of key press events associated with "
            "Backspace/Delete corrections."
        ),
    ),
    FeatureDefinition(
        name="pause_count",
        unit="count",
        description=(
            "Number of inter-key intervals exceeding the "
            "configured pause threshold."
        ),
    ),
    FeatureDefinition(
        name="session_duration",
        unit="seconds",
        description="Total duration of the typing session.",
    ),
)


# ----------------------------------------------------------------------
# Canonical feature order
# ----------------------------------------------------------------------

FEATURE_NAMES = tuple(
    feature.name
    for feature in FEATURE_DEFINITIONS
)

FEATURE_KEYS = list(FEATURE_NAMES)


# ----------------------------------------------------------------------
# Feature metadata
# ----------------------------------------------------------------------

FEATURE_SCHEMA = {
    "dwell_mean": {
        "description": "Average time a key is held down",
        "unit": "seconds",
        "min_value": 0,
    },
    "flight_mean": {
        "description": "Average time between consecutive keystrokes",
        "unit": "seconds",
        "min_value": 0,
    },
    "typing_speed": {
        "description": "Typing speed",
        "unit": "characters/second",
        "min_value": 0,
        "exclusive_min": True,
    },
    "correction_rate": {
        "description": (
            "Proportion of typing actions involving corrections"
        ),
        "unit": "ratio",
        "min_value": 0,
        "max_value": 1,
    },
    "pause_count": {
        "description": "Number of significant pauses during the session",
        "unit": "count",
        "min_value": 0,
    },
    "session_duration": {
        "description": "Total duration of the typing session",
        "unit": "seconds",
        "min_value": 0,
    },
}


# ----------------------------------------------------------------------
# Consistency check
# ----------------------------------------------------------------------

assert list(FEATURE_SCHEMA.keys()) == FEATURE_KEYS


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def get_feature_definition(feature_name):
    """
    Return the canonical definition for a feature.

    Raises
    ------
    KeyError
        If the requested feature is not part of the canonical contract.
    """

    for feature in FEATURE_DEFINITIONS:
        if feature.name == feature_name:
            return feature

    raise KeyError(
        f"Unknown MindKey feature: {feature_name}"
    )