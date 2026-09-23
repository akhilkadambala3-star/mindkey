"""Canonical units for MindKey behavioral evidence (Phase 2).

Why this module exists
----------------------
The existing pipeline has a documented unit mismatch between layers:

- ``agent/features.py`` (capture) and the Supabase ``typing_sessions`` table
  store ``typing_speed`` as **characters per minute (CPM)**, ``dwell_mean`` /
  ``flight_mean`` / ``rhythm_variability`` as **seconds**, ``correction_rate``
  as a **ratio in [0, 1]**, and ``pause_count`` as a **count**.
- ``dashboard/data.js`` assumes ``wpm = typing_speed / 5`` and displays dwell
  in **milliseconds** (``dwell_mean * 1000``).

The evidence layer resolves this ambiguity by choosing the **stored units as
canonical** and giving every contract field an explicit unit-suffixed name, so
that no consumer can misread a value. The values themselves are NOT changed and
nothing is migrated: only the naming of the fields in the new evidence
contract is made unambiguous.

WPM and milliseconds are deliberately NOT duplicated in the contract. They
remain a presentation concern; the ``to_wpm`` / ``to_milliseconds`` helpers
below exist purely to document that conversion and are never applied to stored
data by this layer.
"""

#: Canonical contract field name -> human-readable canonical unit.
CANONICAL_UNIT_NAMES = {
    "typing_speed_cpm": "characters per minute",
    "dwell_mean_s": "seconds",
    "flight_mean_s": "seconds",
    "rhythm_variability_s": "seconds (standard deviation)",
    "correction_rate": "ratio (0-1)",
    "pause_count": "count",
}

#: Stored column name (as written by the existing pipeline) -> canonical field.
STORED_TO_CANONICAL = {
    "typing_speed": "typing_speed_cpm",
    "dwell_mean": "dwell_mean_s",
    "flight_mean": "flight_mean_s",
    "rhythm_variability": "rhythm_variability_s",
    "correction_rate": "correction_rate",
    "pause_count": "pause_count",
}

#: The six canonical feature keys, in the fixed order used across the layer.
CANONICAL_FEATURE_KEYS = (
    "typing_speed_cpm",
    "dwell_mean_s",
    "flight_mean_s",
    "correction_rate",
    "rhythm_variability_s",
    "pause_count",
)

#: Display labels for the canonical features.
FEATURE_LABELS = {
    "typing_speed_cpm": "Typing speed",
    "dwell_mean_s": "Dwell time",
    "flight_mean_s": "Flight time",
    "correction_rate": "Correction rate",
    "rhythm_variability_s": "Rhythm variability",
    "pause_count": "Pauses",
}

#: Presentation-only conversion factor mirrored from the dashboard.
_KEYSTROKES_PER_WORD = 5.0


def canonical_units():
    """Return a copy of the canonical feature -> unit documentation mapping."""
    return dict(CANONICAL_UNIT_NAMES)


def canonical_name(stored_key):
    """Map a stored column name to its canonical contract field name.

    Returns ``None`` if the stored key has no canonical equivalent.
    """
    return STORED_TO_CANONICAL.get(stored_key)


def unit_for(canonical_key):
    """Return the canonical unit string for a canonical feature key."""
    return CANONICAL_UNIT_NAMES.get(canonical_key, "unknown")


def label_for(canonical_key):
    """Return the display label for a canonical feature key."""
    return FEATURE_LABELS.get(canonical_key, canonical_key)


def to_wpm(typing_speed_cpm):
    """Presentation-only: convert canonical CPM to words per minute.

    This mirrors the dashboard assumption (5 keystrokes per word). It is never
    applied to stored data by the evidence layer.
    """
    if typing_speed_cpm is None:
        return None
    return typing_speed_cpm / _KEYSTROKES_PER_WORD


def to_milliseconds(seconds):
    """Presentation-only: convert canonical seconds to milliseconds."""
    if seconds is None:
        return None
    return seconds * 1000.0


def to_seconds(milliseconds):
    """Presentation-only: convert milliseconds to canonical seconds."""
    if milliseconds is None:
        return None
    return milliseconds / 1000.0
