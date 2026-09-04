import statistics

# Flight time above this is counted as a meaningful pause, not normal inter-key delay.
PAUSE_THRESHOLD_S = 1.0

# Standard WPM assumption: 5 keystrokes ≈ 1 word.
KEYSTROKES_PER_WORD = 5.0


def _mean(values):
    if not values:
        return 0.0
    return statistics.mean(values)


def extract_features(
    dwell_times,
    flight_times,
    key_press_count,
    correction_count,
    session_duration_s,
):
    """
    Build session summary features from timing/event counts only.

    dwell_times: seconds a key was held (press to matching release)
    flight_times: seconds from one release to the next press
    key_press_count: total key-down events (no key identities)
    correction_count: Backspace/Delete press count (no characters stored)
    session_duration_s: elapsed typing time in seconds
    """
    dwell_times = list(dwell_times or [])
    flight_times = list(flight_times or [])
    key_press_count = max(0, int(key_press_count or 0))
    correction_count = max(0, int(correction_count or 0))
    session_duration_s = float(session_duration_s or 0.0)

    dwell_mean = _mean(dwell_times)

    normal_flight_times = [
        flight for flight in flight_times if flight <= PAUSE_THRESHOLD_S
    ]
    pause_count = sum(1 for flight in flight_times if flight > PAUSE_THRESHOLD_S)

    flight_mean = _mean(normal_flight_times)

    if session_duration_s > 0:
        net_keystrokes = max(0, key_press_count - correction_count)
        words = net_keystrokes / KEYSTROKES_PER_WORD
        minutes = session_duration_s / 60.0
        typing_speed = words / minutes
    else:
        typing_speed = 0.0

    if key_press_count > 0:
        correction_rate = correction_count / key_press_count
    else:
        correction_rate = 0.0

    if len(normal_flight_times) >= 2:
        rhythm_variability = statistics.stdev(normal_flight_times)
    else:
        rhythm_variability = 0.0

    return {
        "dwell_mean": dwell_mean,
        "flight_mean": flight_mean,
        "typing_speed": typing_speed,
        "correction_rate": correction_rate,
        "rhythm_variability": rhythm_variability,
        "pause_count": pause_count,
    }
