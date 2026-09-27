"""
Labeled evaluation dataset for the MindKey ML system.

This module creates controlled synthetic behavioral sessions
for evaluating the anomaly detection pipeline.

Labels:
    0 = normal session
    1 = anomalous session

The anomalies represent statistical deviations in behavioral
features. They do NOT represent medical conditions or diagnoses.
"""


def make_normal_session(
    dwell_mean=0.10,
    flight_mean=0.25,
    typing_speed=40.0,
    correction_rate=0.10,
    pause_count=3,
    session_duration=60.0,
):
    """
    Create one normal behavioral session.

    The default values represent a typical synthetic
    behavioral session for testing.
    """

    return {
        "dwell_mean": dwell_mean,
        "flight_mean": flight_mean,
        "typing_speed": typing_speed,
        "correction_rate": correction_rate,
        "pause_count": pause_count,
        "session_duration": session_duration,
    }


def make_anomalous_session(
    dwell_mean=0.30,
    flight_mean=0.70,
    typing_speed=15.0,
    correction_rate=0.60,
    pause_count=15,
    session_duration=180.0,
):
    """
    Create one strongly deviated behavioral session.

    These values are intentionally far from the normal
    synthetic distribution so that the detector has
    controlled anomalies to evaluate.

    This represents statistical abnormality only.
    It does not represent a medical condition.
    """

    return {
        "dwell_mean": dwell_mean,
        "flight_mean": flight_mean,
        "typing_speed": typing_speed,
        "correction_rate": correction_rate,
        "pause_count": pause_count,
        "session_duration": session_duration,
    }


def build_evaluation_dataset():
    """
    Build a labeled evaluation dataset.

    Returns
    -------
    tuple
        (
            normal_sessions,
            anomalous_sessions,
            y_true
        )

    normal_sessions:
        List of normal behavioral sessions.

    anomalous_sessions:
        List of controlled anomalous sessions.

    y_true:
        Ground-truth labels where:
            0 = normal
            1 = anomaly
    """

    normal_sessions = [
        make_normal_session(
            dwell_mean=0.10,
            flight_mean=0.25,
            typing_speed=40.0,
            correction_rate=0.10,
            pause_count=3,
            session_duration=60.0,
        ),

        make_normal_session(
            dwell_mean=0.11,
            flight_mean=0.27,
            typing_speed=42.0,
            correction_rate=0.12,
            pause_count=4,
            session_duration=62.0,
        ),

        make_normal_session(
            dwell_mean=0.09,
            flight_mean=0.23,
            typing_speed=38.0,
            correction_rate=0.08,
            pause_count=2,
            session_duration=58.0,
        ),

        make_normal_session(
            dwell_mean=0.12,
            flight_mean=0.28,
            typing_speed=43.0,
            correction_rate=0.13,
            pause_count=5,
            session_duration=65.0,
        ),

        make_normal_session(
            dwell_mean=0.095,
            flight_mean=0.24,
            typing_speed=39.0,
            correction_rate=0.09,
            pause_count=3,
            session_duration=59.0,
        ),

        make_normal_session(
            dwell_mean=0.105,
            flight_mean=0.26,
            typing_speed=41.0,
            correction_rate=0.11,
            pause_count=4,
            session_duration=61.0,
        ),

        make_normal_session(
            dwell_mean=0.115,
            flight_mean=0.29,
            typing_speed=44.0,
            correction_rate=0.14,
            pause_count=5,
            session_duration=66.0,
        ),

        make_normal_session(
            dwell_mean=0.085,
            flight_mean=0.22,
            typing_speed=37.0,
            correction_rate=0.07,
            pause_count=2,
            session_duration=57.0,
        ),

        make_normal_session(
            dwell_mean=0.10,
            flight_mean=0.25,
            typing_speed=40.0,
            correction_rate=0.10,
            pause_count=3,
            session_duration=60.0,
        ),

        make_normal_session(
            dwell_mean=0.108,
            flight_mean=0.255,
            typing_speed=41.5,
            correction_rate=0.11,
            pause_count=4,
            session_duration=63.0,
        ),
    ]

    anomalous_sessions = [
        make_anomalous_session(
            dwell_mean=0.30,
            flight_mean=0.70,
            typing_speed=15.0,
            correction_rate=0.60,
            pause_count=15,
            session_duration=180.0,
        ),

        make_anomalous_session(
            dwell_mean=0.35,
            flight_mean=0.80,
            typing_speed=12.0,
            correction_rate=0.70,
            pause_count=18,
            session_duration=210.0,
        ),

        make_anomalous_session(
            dwell_mean=0.28,
            flight_mean=0.65,
            typing_speed=17.0,
            correction_rate=0.55,
            pause_count=14,
            session_duration=170.0,
        ),

        make_anomalous_session(
            dwell_mean=0.40,
            flight_mean=0.90,
            typing_speed=10.0,
            correction_rate=0.80,
            pause_count=20,
            session_duration=240.0,
        ),

        make_anomalous_session(
            dwell_mean=0.32,
            flight_mean=0.75,
            typing_speed=14.0,
            correction_rate=0.65,
            pause_count=16,
            session_duration=195.0,
        ),

        make_anomalous_session(
            dwell_mean=0.27,
            flight_mean=0.60,
            typing_speed=18.0,
            correction_rate=0.50,
            pause_count=12,
            session_duration=160.0,
        ),

        make_anomalous_session(
            dwell_mean=0.38,
            flight_mean=0.85,
            typing_speed=11.0,
            correction_rate=0.75,
            pause_count=19,
            session_duration=225.0,
        ),

        make_anomalous_session(
            dwell_mean=0.29,
            flight_mean=0.68,
            typing_speed=16.0,
            correction_rate=0.58,
            pause_count=13,
            session_duration=175.0,
        ),

        make_anomalous_session(
            dwell_mean=0.34,
            flight_mean=0.78,
            typing_speed=13.0,
            correction_rate=0.68,
            pause_count=17,
            session_duration=205.0,
        ),

        make_anomalous_session(
            dwell_mean=0.31,
            flight_mean=0.72,
            typing_speed=15.5,
            correction_rate=0.62,
            pause_count=15,
            session_duration=185.0,
        ),
    ]

    y_true = (
        [0] * len(normal_sessions)
        + [1] * len(anomalous_sessions)
    )

    return (
        normal_sessions,
        anomalous_sessions,
        y_true,
    )