from datetime import datetime, timezone
from time import perf_counter

import requests
from pynput import keyboard

from features import extract_features

TEST_USER_ID = "e1a556fe-eb9b-405a-9859-dca55c715493"
BACKEND_URL = "http://127.0.0.1:8000/typing/session"

# Maps an in-memory key object to its press time so dwell can be computed.
# The key itself is never printed or written anywhere.
press_times = {}
last_release_time = None
dwell_times = []
flight_times = []
key_press_count = 0
correction_count = 0
session_start = None
session_start_at = None


def _is_correction(key):
    return key in (keyboard.Key.backspace, keyboard.Key.delete)


def on_press(key):
    global last_release_time, key_press_count, correction_count

    # Esc ends the session; do not treat it as typing data.
    if key == keyboard.Key.esc:
        return

    now = perf_counter()

    # Ignore OS key-repeat while the key is already held.
    if key in press_times:
        return

    if last_release_time is not None:
        flight_time = now - last_release_time
        flight_times.append(flight_time)
        print(f"flight_time={flight_time:.4f}s")

    press_times[key] = now
    key_press_count += 1
    if _is_correction(key):
        correction_count += 1


def on_release(key):
    global last_release_time

    now = perf_counter()

    if key == keyboard.Key.esc:
        return False

    press_time = press_times.pop(key, None)

    if press_time is not None:
        dwell_time = now - press_time
        dwell_times.append(dwell_time)
        print(f"dwell_time={dwell_time:.4f}s")

    last_release_time = now


def main():
    global session_start, session_start_at

    session_start = perf_counter()
    session_start_at = datetime.now(timezone.utc)
    print("MindKey listener running. Type to see dwell/flight times. Press Esc to stop.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    session_end_at = datetime.now(timezone.utc)
    session_duration_s = perf_counter() - session_start
    features = extract_features(
        dwell_times=dwell_times,
        flight_times=flight_times,
        key_press_count=key_press_count,
        correction_count=correction_count,
        session_duration_s=session_duration_s,
    )

    print("MindKey listener stopped.")
    print("Session features:")
    for name, value in features.items():
        print(f"  {name}: {value}")

    payload = {
        "user_id": TEST_USER_ID,
        "session_start": session_start_at.isoformat(),
        "session_end": session_end_at.isoformat(),
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


if __name__ == "__main__":
    main()
