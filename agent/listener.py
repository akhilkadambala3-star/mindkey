"""MindKey typing listener with an automatic 2-minute session lifecycle.

Sessions no longer require pressing Esc to end:

- A session starts on the first meaningful keypress (non-Esc, non-repeat)
  and ends automatically exactly 2 minutes later.
- When a session ends, its timing features are extracted and uploaded to
  the existing ``/typing/session`` endpoint exactly as before.
- After the upload the listener waits for the next meaningful keypress
  before starting another session.
- At most 10 completed sessions are collected per calendar day; the
  counter resets when the local calendar date changes.

Only aggregate timing statistics are kept and printed. Keystroke
identities are never stored, printed, or sent anywhere.
"""

from datetime import datetime, timezone
from threading import Lock
from time import perf_counter, sleep

import requests
from pynput import keyboard

from features import extract_features

TEST_USER_ID = "e1a556fe-eb9b-405a-9859-dca55c715493"
BACKEND_URL = "http://127.0.0.1:8000/typing/session"

# A session runs for exactly this long, counted from its first keypress.
SESSION_DURATION_S = 120.0
# Maximum number of completed sessions collected per calendar day.
DAILY_SESSION_LIMIT = 10
# How often the main loop re-checks for activity while no session is open.
IDLE_POLL_INTERVAL_S = 0.1

# pynput invokes the press/release callbacks on its own thread while the
# main thread enforces the 2-minute deadline, so all shared state below is
# guarded by this lock.
lock = Lock()

# Maps an in-memory key object to its press time so dwell can be computed.
# The key itself is never printed or written anywhere.
press_times = {}
last_release_time = None
dwell_times = []
flight_times = []
key_press_count = 0
correction_count = 0

# Session lifecycle state.
session_active = False  # a 2-minute collection window is currently open
uploading = False  # a finished session is mid-upload; ignore keypresses
session_start = None  # perf_counter() time the active session began
session_start_at = None  # wall-clock (UTC) time the active session began
sessions_completed_today = 0
limit_day = None  # local calendar date sessions_completed_today applies to
limit_reported = False


def _is_correction(key):
    return key in (keyboard.Key.backspace, keyboard.Key.delete)


def _reset_day_if_needed():
    """Reset the daily completed-session counter when the date changes.

    Caller must hold ``lock``.
    """
    global limit_day, sessions_completed_today, limit_reported

    today = datetime.now().date()
    if limit_day != today:
        limit_day = today
        sessions_completed_today = 0
        limit_reported = False


def _start_session(now):
    """Open a fresh 2-minute collection window.

    Caller must hold ``lock``.
    """
    global session_active, session_start, session_start_at, last_release_time
    global dwell_times, flight_times, key_press_count, correction_count

    session_active = True
    session_start = now
    session_start_at = datetime.now(timezone.utc)
    dwell_times = []
    flight_times = []
    key_press_count = 0
    correction_count = 0
    # No flight time into the first key of a session: the gap since the
    # previous session is idle time, not inter-key delay.
    last_release_time = None


def on_press(key):
    global key_press_count, correction_count, limit_reported

    # Esc is not typing data, and it no longer ends anything.
    if key == keyboard.Key.esc:
        return

    now = perf_counter()

    with lock:
        # Ignore OS key-repeat while the key is already held.
        if key in press_times:
            return

        # Keypresses that arrive while a finished session is being uploaded
        # are skipped; the next session starts on activity after the upload.
        if uploading:
            return

        # Start a session on the first meaningful keypress. Nothing is
        # collected while the listener is idle between sessions.
        if not session_active:
            _reset_day_if_needed()
            if sessions_completed_today >= DAILY_SESSION_LIMIT:
                if not limit_reported:
                    print(
                        "Daily limit of 10 sessions reached; "
                        "no more sessions until tomorrow."
                    )
                    limit_reported = True
                return
            _start_session(now)

        if last_release_time is not None:
            flight_time = now - last_release_time
            flight_times.append(flight_time)

        press_times[key] = now
        key_press_count += 1
        if _is_correction(key):
            correction_count += 1


def on_release(key):
    global last_release_time

    # Esc no longer ends a session or the listener.
    if key == keyboard.Key.esc:
        return

    now = perf_counter()

    with lock:
        if not session_active or uploading:
            return

        press_time = press_times.pop(key, None)
        if press_time is None:
            # Release of a key that was not pressed during this session
            # (e.g. one still held when the previous session ended).
            return

        dwell_time = now - press_time
        dwell_times.append(dwell_time)

        last_release_time = now


def _finalize_session():
    """End the active session at its 2-minute deadline and snapshot its data.

    Called by the main thread once the deadline has passed. Caller must
    hold ``lock``. Returns a dict snapshot of the finished session.
    """
    global session_active, uploading
    global session_start, session_start_at, last_release_time
    global dwell_times, flight_times, key_press_count, correction_count
    global sessions_completed_today

    snapshot = {
        "session_start_at": session_start_at,
        "session_end_at": datetime.now(timezone.utc),
        "dwell_times": list(dwell_times),
        "flight_times": list(flight_times),
        "key_press_count": key_press_count,
        "correction_count": correction_count,
        "session_duration_s": perf_counter() - session_start,
    }

    # A session counts as completed on the calendar day it finishes.
    _reset_day_if_needed()
    sessions_completed_today += 1

    session_active = False
    uploading = True  # block new sessions until this one has been uploaded
    session_start = None
    session_start_at = None
    press_times.clear()
    last_release_time = None
    dwell_times = []
    flight_times = []
    key_press_count = 0
    correction_count = 0

    return snapshot


def _upload_session(snapshot):
    """Extract features from a finished session and POST them as before."""
    features = extract_features(
        dwell_times=snapshot["dwell_times"],
        flight_times=snapshot["flight_times"],
        key_press_count=snapshot["key_press_count"],
        correction_count=snapshot["correction_count"],
        session_duration_s=snapshot["session_duration_s"],
    )

    print("Session features:")
    for name, value in features.items():
        print(f"  {name}: {value}")

    payload = {
        "user_id": TEST_USER_ID,
        "session_start": snapshot["session_start_at"].isoformat(),
        "session_end": snapshot["session_end_at"].isoformat(),
        "dwell_mean": features["dwell_mean"],
        "flight_mean": features["flight_mean"],
        "typing_speed": features["typing_speed"],
        "correction_rate": features["correction_rate"],
        "rhythm_variability": features["rhythm_variability"],
        "pause_count": features["pause_count"],
    }

    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=10)
        print("Backend response:")
        print(f"  status_code: {response.status_code}")
        print(f"  body: {response.text}")
    except requests.RequestException:
        print("Could not send session to the backend.")
        print("Make sure FastAPI is running at http://127.0.0.1:8000")


def _mark_upload_done():
    """Reopen the listener to new sessions once the upload has finished.

    Caller must hold ``lock``.
    """
    global uploading
    uploading = False


def main():
    print(
        "MindKey listener running: a session starts on your first keypress "
        "and ends automatically after 2 minutes. Press Ctrl+C to quit."
    )
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while True:
            # Sleep precisely until the active session's deadline...
            deadline = None
            with lock:
                if session_active:
                    deadline = session_start + SESSION_DURATION_S

            if deadline is not None:
                remaining = deadline - perf_counter()
                if remaining > 0:
                    sleep(remaining)
            else:
                sleep(IDLE_POLL_INTERVAL_S)

            # ...then end the session if its deadline has passed.
            snapshot = None
            with lock:
                if (
                    session_active
                    and perf_counter() - session_start >= SESSION_DURATION_S
                ):
                    snapshot = _finalize_session()

            if snapshot is not None:
                _upload_session(snapshot)
                with lock:
                    _mark_upload_done()
    except KeyboardInterrupt:
        print("\nMindKey listener stopped.")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
