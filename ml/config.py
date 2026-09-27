"""
Central configuration for MindKey behavioral ML.

The purpose of this file is to prevent important ML assumptions from
being scattered across multiple modules.
"""

# ----------------------------------------------------------------------
# Feature validation thresholds
# ----------------------------------------------------------------------

MIN_TIMING_VALUE = 0.0

MIN_TYPING_SPEED = 0.0

MIN_CORRECTION_RATE = 0.0
MAX_CORRECTION_RATE = 1.0

MIN_PAUSE_COUNT = 0

MIN_SESSION_DURATION = 0.0

# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------

# Current listener session duration.
SESSION_DURATION_SECONDS = 20.0

# Maximum number of completed sessions collected per local calendar day.
DAILY_SESSION_LIMIT = 100

# Minimum number of key presses considered useful for a session.
MINIMUM_KEYS_FOR_SESSION = 20


# ---------------------------------------------------------------------------
# Feature extraction configuration
# ---------------------------------------------------------------------------

# Flight intervals above this threshold are considered meaningful pauses
# rather than normal inter-key timing.
PAUSE_THRESHOLD_SECONDS = 1.0

# Standard WPM approximation:
# approximately five keystrokes represent one word.
KEYSTROKES_PER_WORD = 5.0


# ---------------------------------------------------------------------------
# Baseline configuration
# ---------------------------------------------------------------------------

# Minimum valid historical sessions required before a personal baseline
# can be constructed.
MINIMUM_BASELINE_SESSIONS = 10

# Minimum sessions required before normal behavioral monitoring is allowed.
MINIMUM_BASELINE_SESSIONS_FOR_MONITORING = 10


# ---------------------------------------------------------------------------
# Anomaly detection configuration
# ---------------------------------------------------------------------------

ISOLATION_FOREST_ESTIMATORS = 100

ISOLATION_FOREST_CONTAMINATION = 0.10

ISOLATION_FOREST_RANDOM_STATE = 42