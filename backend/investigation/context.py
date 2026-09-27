"""User-reported context (wellbeing check-ins) for the evidence layer.

A check-in is one self-reported factor for one day, such as "poor sleep" or
"feeling well". It is context, not a measurement: it can make a contextual
explanation more or less plausible, but it never establishes a cause and it
never overrides the measured behavioral evidence.

Rules this module encodes:

- Only the day and the factor id are used. Free-text notes are never read by
  the investigation, so nothing a user writes in a note can reach a claim.
- Unknown factor ids are ignored rather than guessed.
- A missing store is different from an empty one. ``None`` means "no check-in
  store is available" and the adapter reports context as unavailable, exactly
  as before this module existed. An empty list means "the store exists but the
  user reported nothing in this window".
"""

from datetime import date, datetime, timedelta, timezone

#: Stored factor id -> (reporting label, alternative candidate it bears on).
#: ``None`` as the candidate means the factor is not a disruption.
CHECKIN_FACTORS: dict[str, tuple[str, str | None]] = {
    "feeling_well": ("feeling well", None),
    "tired": ("tiredness", "fatigue"),
    "stressed": ("stress", "stress"),
    "poor_sleep": ("poor sleep", "poor_sleep"),
    "unwell": ("feeling unwell", "illness_or_mood"),
    "distracted": ("being busy or distracted", "distraction"),
    "other": ("something else", None),
}

#: The factor that counts as a report *against* a contextual disruption.
POSITIVE_FACTOR = "feeling_well"

#: Stable reporting order for factors.
FACTOR_ORDER: tuple[str, ...] = tuple(CHECKIN_FACTORS)

#: Source label recorded on ``ContextEvidence.source`` when check-ins are read.
SOURCE_CHECKINS = "user_checkins"

#: Maximum stored note length accepted by the API (notes never reach the agent).
MAX_NOTE_LENGTH = 280


def candidate_for(factor):
    """The alternative-explanation candidate a factor bears on, if any."""
    entry = CHECKIN_FACTORS.get(factor)
    return None if entry is None else entry[1]


def label_for_factor(factor):
    """Reporting label for a factor id (``None`` for unknown ids)."""
    entry = CHECKIN_FACTORS.get(factor)
    return None if entry is None else entry[0]


def _as_day(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def normalize_checkin(row):
    """Reduce a stored check-in row to ``{"date", "factor"}`` or ``None``.

    Rows without a usable day or with an unknown factor are dropped. Notes and
    every other column are deliberately discarded here.
    """
    if not isinstance(row, dict):
        return None
    factor = row.get("factor")
    day = _as_day(row.get("date") or row.get("created_at"))
    if factor not in CHECKIN_FACTORS or day is None:
        return None
    return {"date": day.isoformat(), "factor": factor}


def checkins_in_window(rows, reference, window_days):
    """Normalized, de-duplicated check-ins whose day falls in the window.

    The window is the ``window_days`` calendar days ending on the reference
    day (inclusive), which matches how the recent session window is read.
    Returned in (date, factor-order) order so evidence is deterministic.
    """
    if reference is None:
        reference = datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    end = reference.astimezone(timezone.utc).date()
    start = end - timedelta(days=max(int(window_days), 1) - 1)

    seen = set()
    kept = []
    for row in rows or []:
        item = normalize_checkin(row)
        if item is None:
            continue
        day = date.fromisoformat(item["date"])
        if not (start <= day <= end):
            continue
        key = (item["date"], item["factor"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    kept.sort(key=lambda c: (c["date"], FACTOR_ORDER.index(c["factor"])))
    return kept


def factor_days(checkins):
    """``factor -> number of distinct days reported``, in factor order."""
    days: dict[str, set] = {}
    for item in checkins or []:
        days.setdefault(item["factor"], set()).add(item["date"])
    return {f: len(days[f]) for f in FACTOR_ORDER if f in days}


def read_checkins(repository, user_id):
    """Read stored check-ins through an optional repository capability.

    Returns ``None`` when the repository has no check-in store (or the store
    cannot be read), and a list of raw rows otherwise. A read failure is
    reported by the caller as "context unavailable", never as an empty list,
    so an outage can never look like "the user reported nothing".
    """
    reader = getattr(repository, "list_checkins", None)
    if reader is None:
        return None
    try:
        rows = reader(user_id)
    except Exception:  # store unreachable / table missing
        return None
    if rows is None:
        return None
    return list(rows)
