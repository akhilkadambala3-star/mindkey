"""
Feature quality analysis for MindKey.

Evaluates whether extracted ML sessions contain valid,
finite, and usable behavioral features.

This module performs data-quality validation only.
It does not diagnose or infer medical conditions.
"""

from ml.feature_schema import FEATURE_KEYS
from ml.features import is_valid_session


def summarize_session_quality(sessions):
    """
    Summarize the validity of a collection of sessions.

    Parameters
    ----------
    sessions : list
        List of session feature dictionaries.

    Returns
    -------
    dict
        Summary containing total, valid, invalid sessions,
        and overall validity rate.
    """

    total_sessions = len(sessions)

    valid_sessions = 0
    invalid_sessions = 0

    for session in sessions:

        if is_valid_session(session):

            valid_sessions += 1

        else:

            invalid_sessions += 1

    if total_sessions > 0:

        valid_rate = (
            valid_sessions / total_sessions
        )

    else:

        valid_rate = 0.0

    return {
        "total_sessions": total_sessions,
        "valid_sessions": valid_sessions,
        "invalid_sessions": invalid_sessions,
        "valid_rate": valid_rate,
    }


def analyze_feature_quality(sessions):
    """
    Analyze validity for each canonical ML feature.

    Each feature is evaluated independently.

    This means that if one feature is missing from a
    session, the other valid features can still count
    as valid.

    Parameters
    ----------
    sessions : list
        List of session feature dictionaries.

    Returns
    -------
    dict
        Per-feature quality statistics.
    """

    report = {}

    total_sessions = len(sessions)

    for feature_name in FEATURE_KEYS:

        valid_count = 0
        missing_count = 0
        invalid_count = 0

        for session in sessions:

            # Session must be a dictionary.
            if not isinstance(session, dict):

                invalid_count += 1
                continue

            # Check whether this particular feature
            # exists in the session.
            if feature_name not in session:

                missing_count += 1
                continue

            value = session[feature_name]

            # A present feature with None is invalid.
            if value is None:

                invalid_count += 1
                continue

            # Validate the feature using the canonical
            # session validation rules.
            try:

                session_is_valid = is_valid_session(
                    session
                )

            except (TypeError, ValueError):

                session_is_valid = False

            if session_is_valid:

                valid_count += 1

            else:

                invalid_count += 1

        if total_sessions > 0:

            valid_rate = (
                valid_count / total_sessions
            )

        else:

            valid_rate = 0.0

        report[feature_name] = {
            "total_count": total_sessions,
            "valid_count": valid_count,
            "missing_count": missing_count,
            "invalid_count": invalid_count,
            "valid_rate": valid_rate,
        }

    return report