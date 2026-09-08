"""Feature definitions and session validation for the MindKey ML layer.

The six numerical typing features are extracted per session by
``agent/features.py`` and stored by the backend. The ML layer consumes them
exactly as stored, in this fixed order.
"""

#: The six numerical typing features used by MindKey's ML layer.
#: Order matters: it defines the column layout of the feature matrix.
FEATURE_KEYS = (
    "dwell_mean",
    "flight_mean",
    "typing_speed",
    "correction_rate",
    "rhythm_variability",
    "pause_count",
)


def is_valid_session(session):
    """Return True if a typing session can be used as model input.

    A session is valid when all six features are present and physically
    plausible. The rules mirror the ones already used by
    ``backend/routes/baseline.py`` so the ML layer trains and evaluates on
    exactly the same notion of a "valid historical session":

    - every one of the six features must be present (not None)
    - ``typing_speed`` must be greater than 0
    - ``correction_rate`` must be in the range [0, 1]

    Sessions failing these checks (e.g. an aborted or broken recording) are
    excluded from training and refused at evaluation time.
    """
    for key in FEATURE_KEYS:
        if session.get(key) is None:
            return False

    typing_speed = session["typing_speed"]
    correction_rate = session["correction_rate"]

    if typing_speed <= 0:
        return False
    if correction_rate < 0 or correction_rate > 1:
        return False

    return True