# MindKey Dashboard (Frontend)

This is the frontend for **MindKey** — a privacy-conscious system that learns a person's normal typing behavior and flags *persistent* deviations from their own baseline. It is **not** a diagnostic tool.

## Files

- `index.html` — dashboard layout (status card, metrics, anomaly score, trend chart, session table, privacy controls).
- `style.css` — dark-themed styling, responsive down to mobile widths.
- `app.js` — rendering logic + a **mock data generator** that simulates the demo story: normal → normal → normal → gradual drift → persistent deviation → flag. This lets the dashboard run standalone before the backend exists.

## Running it now (no backend needed)

Just open `index.html` in a browser, or serve the folder statically:

```bash
cd dashboard
python -m http.server 5500
```

Then visit `http://localhost:5500`. Chart.js is loaded from a CDN, so you need internet access in the browser (not in the hackathon dev sandbox — this only matters for whoever opens the page).

## Switching to the real FastAPI backend

In `app.js`, change:

```js
const API_BASE = null; // mock mode
```

to:

```js
const API_BASE = "http://localhost:8000";
```

The frontend expects these endpoints. Align the backend (Person 3) to return these shapes:

### `GET /api/users/{user_id}/baseline`
```json
{ "wpm": 52, "dwell": 82, "flight": 121, "pauses": 4, "correction": 3, "rhythm": 0.12 }
```

### `GET /api/users/{user_id}/sessions`
Array of session summaries, oldest first:
```json
[
  {
    "session_id": "S001",
    "date": "2026-08-01",
    "wpm": 52,
    "dwell_mean_ms": 82,
    "flight_mean_ms": 121,
    "pause_count": 4,
    "correction_rate": 3.0,
    "rhythm_variability": 0.12,
    "consistency": 94,
    "anomaly_score": 0.05,
    "status": "normal"
  }
]
```
`status` should be one of `"normal"`, `"elevated"`, `"flagged"` — computed server-side by the Isolation Forest / persistence-check logic (Phase 6 in the project plan), not by the frontend.

No endpoint should ever return typed message content — only the timing-derived fields above.

## Design notes

- The **status badge** only shows "PERSISTENT CHANGE DETECTED" when the last several sessions are consistently elevated/flagged — a single unusual session never triggers it (mirrors Phase 12 ML logic).
- The **disclaimer banner** and the flagged-status message intentionally use the exact non-diagnostic language required by the project (`This is not a medical diagnosis...`). Do not remove or soften this when integrating the real backend.
- The **privacy panel** lists the data-minimization commitments; wire "Delete all my data" to a real `DELETE /api/users/{user_id}/data` call once the backend supports it — right now it only clears local mock state.
- Chart metric toggle (WPM / Dwell / Flight / Pauses / Consistency) reuses one Chart.js line chart instance — no extra chart libraries needed.
