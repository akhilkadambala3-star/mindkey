"""
Personal behavioral baseline engine for MindKey.

Purpose
-------
Build a user's historical behavioral reference distribution.

The baseline is personal:
    user history -> user's own behavioral reference

It is not a population norm and it is not a medical baseline.

The baseline provides:
- central tendency
- variability
- robust variability
- sample count
- standardized deviation for new observations
"""

import math
import statistics

from ml.features import (
    FEATURE_KEYS,
    is_valid_session,
)

class BaselineNotReadyError(Exception):
    """Raised when there is insufficient valid history."""


class PersonalBaseline:
    """
    Personal behavioral baseline for one user.

    Each feature receives:
    - mean
    - median
    - standard deviation
    - median absolute deviation
    - sample count

    The median/MAD values are retained because behavioral data can contain
    occasional unusual sessions, and robust statistics are useful for
    longitudinal monitoring.
    """

    def __init__(
        self,
        feature_statistics,
        sample_count,
    ):
        self.feature_statistics = feature_statistics
        self.sample_count = sample_count

    def describe(self):
        """
        Return the complete baseline as a serializable dictionary.
        """

        return {
            "sample_count": self.sample_count,
            "features": self.feature_statistics,
        }

    def score_session(self, session):
        """
        Measure how far a session deviates from the personal baseline.

        Returns one robust standardized score per feature.

        Interpretation:
        - approximately 0 -> close to personal baseline
        - larger absolute value -> farther from baseline

        This is a deviation measure, not a diagnosis or probability.
        """

        if not is_valid_session(session):
            raise ValueError(
                "Session failed ML validation."
            )

        feature_scores = {}

        for feature_name in FEATURE_KEYS:

            value = float(
                session[feature_name]
            )

            stats = self.feature_statistics[
                feature_name
            ]

            robust_scale = stats[
                "mad"
            ]

            standard_deviation = stats[
                "std"
            ]

            # Prefer robust MAD when it provides useful scale.
            if robust_scale > 0:

                # 1.4826 makes MAD approximately comparable
                # to standard deviation for normally distributed data.
                robust_std = (
                    1.4826 * robust_scale
                )

                z_score = (
                    value - stats["median"]
                ) / robust_std

            elif standard_deviation > 0:

                z_score = (
                    value - stats["mean"]
                ) / standard_deviation

            else:
                z_score = 0.0

            feature_scores[feature_name] = {
                "value": value,
                "baseline_center": stats["median"],
                "z_score": float(z_score),
                "absolute_deviation": float(
                    abs(z_score)
                ),
            }

        return feature_scores


def _median_absolute_deviation(values, median):
    """
    Calculate Median Absolute Deviation (MAD).
    """

    absolute_deviations = [
        abs(value - median)
        for value in values
    ]

    if not absolute_deviations:
        return 0.0

    return statistics.median(
        absolute_deviations
    )


def build_personal_baseline(
    sessions,
    minimum_sessions=10,
):
    """
    Build a personal baseline from historical sessions.

    Invalid sessions are excluded.

    Parameters
    ----------
    sessions:
        Historical session dictionaries.

    minimum_sessions:
        Minimum valid sessions required.

    Returns
    -------
    PersonalBaseline

    Raises
    ------
    BaselineNotReadyError
        If insufficient valid sessions are available.
    """

    valid_sessions = [
        session
        for session in sessions
        if is_valid_session(session)
    ]

    if len(valid_sessions) < minimum_sessions:
        raise BaselineNotReadyError(
            f"Need at least {minimum_sessions} "
            f"valid sessions to build a personal "
            f"baseline; received "
            f"{len(valid_sessions)}."
        )

    feature_statistics = {}

    for feature_name in FEATURE_KEYS:

        values = [
            float(
                session[feature_name]
            )
            for session in valid_sessions
        ]

        mean_value = statistics.mean(
            values
        )

        median_value = statistics.median(
            values
        )

        std_value = (
            statistics.stdev(values)
            if len(values) >= 2
            else 0.0
        )

        mad_value = (
            _median_absolute_deviation(
                values,
                median_value,
            )
        )

        feature_statistics[feature_name] = {
            "mean": float(mean_value),
            "median": float(median_value),
            "std": float(std_value),
            "mad": float(mad_value),
            "min": float(min(values)),
            "max": float(max(values)),
            "sample_count": len(values),
        }

    return PersonalBaseline(
        feature_statistics=feature_statistics,
        sample_count=len(valid_sessions),
    )
