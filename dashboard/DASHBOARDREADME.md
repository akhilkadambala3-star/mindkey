# MindKey Dashboard (Frontend)

The MindKey frontend — a calm, privacy-first companion for understanding your
own typing behavior over time. It is **not** a diagnostic tool and never
presents a disease risk.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | App shell: sidebar, topbar, and all views (Home, Trends, Sessions, Check-ins, Symptom check, Health Insights, Medical Assistance, Privacy, Settings). |
| `style.css` | The design system: warm-paper light theme, pine-green primary, restrained status colors, responsive layout (full sidebar → icon rail → top bar). |
| `data.js` | **The data layer.** One centralized source for demo data, localStorage persistence, and the `API_BASE` switch that connects the real backend later. The UI never reads raw data directly. |
| `app.js` | SPA logic: router, per-view renderers, typing-state computation, agent-status simulator, Chart.js helpers, check-in / symptom / care flows. |

## Running it

No backend needed — the dashboard runs on simulated data by default:

```bash
cd dashboard
python -m http.server 5500
# open http://localhost:5500
```

Chart.js and the Inter font are loaded from CDNs, so charts need internet
access in the browser. If Chart.js is unreachable the app still works; the
Trends page shows a small fallback note instead of the chart.

## Demo mode

`data.js` generates deterministic demo data for three scenarios, selectable
from the **Demo** dropdown in the topbar (and in Settings):

- **Consistent pattern** — sessions hold close to the personal baseline.
- **Recent variation** — a few recent sessions differ; the home page invites a
  check-in.
- **Persistent change** — the full narrative: persistent change → symptom
  check → multi-signal insight → professional evaluation recommendation →
  medical assistance flow.

Check-in responses, symptom answers, and the demo appointment persist in
`localStorage` (keys `mindkey.*`) so the product behaves like a real app
across reloads. "Delete all my data" clears them. The agent-status card
simulates the real agent lifecycle (waiting → analyzing with countdown →
uploading → daily limit) so the states the product will show are visible now.

Because the dashboard is simulated, a **Demo data** chip is shown in the
topbar whenever demo mode is active, the Monitoring card is labelled
**Simulated**, and Settings → Data source explains that live read endpoints are
not connected yet. Nothing in demo mode claims to be real collected data.

The agent simulator mirrors the real desktop agent instead of inventing its
own numbers. Two constants at the top of `data.js` are the single frontend
source of truth for the simulator *and* for every piece of daily-limit copy:

```js
const AGENT_SESSION_DURATION_S = 20;   // mirrors SESSION_DURATION_S in agent/listener.py
const AGENT_DAILY_SESSION_LIMIT = 100; // mirrors DAILY_SESSION_LIMIT in agent/listener.py
```

A simulated session therefore lasts **20 seconds** and the daily limit is
**100 sessions**, exactly like the agent. Those two values are never
hard-coded again in the UI.

Everything is labeled honestly: controls that exist are marked **Active**,
demo-only behavior is marked **Demo only**, and unimplemented features are
marked **Planned**.

## Connecting the real backend

In `data.js` set:

```js
const API_BASE = "http://localhost:8000"; // was null
```

Two things must exist before that switch is useful:

1. **The read endpoints below** — the backend currently exposes only POST
   `/typing/session`, POST `/baseline/{user_id}`, GET `/health` and GET
   `/supabase-test`.
2. **Browser CORS access on the backend.** It sends no CORS headers today, so a
   dashboard served from `:5500` cannot call it on `:8000`. Adding CORS
   middleware is a backend change and is outside the scope of the frontend
   work.

The UI never changes — every function in the data layer already branches on
`API_BASE`. The backend currently exposes only `POST /typing/session`,
`POST /baseline/{user_id}`, `GET /health`, and `GET /supabase-test`, so these
read/action endpoints still need to be added (shapes below). Until then the
dashboard stays in demo mode, which is intentional.

### Contract (to be implemented server-side)

`GET /api/users/{user_id}/sessions` — array, oldest first:

```json
[
  {
    "session_id": "S0001",
    "session_start": "2026-08-24T09:12:00Z",
    "session_end": "2026-08-24T09:14:00Z",
    "date": "2026-08-24",
    "typing_speed": 285,
    "wpm": 57,
    "dwell_mean_ms": 120,
    "flight_mean_ms": 80,
    "correction_rate": 0.06,
    "rhythm_variability": 0.19,
    "pause_count": 4,
    "consistency": 96,
    "status": "normal"
  }
]
```

`status` is one of `"normal"` / `"elevated"` / `"flagged"` and is *intended* to
come from the backend's Isolation Forest plus persistence logic, never from the
frontend. **That field does not exist server-side today**:
`POST /typing/session` returns an `anomaly` object containing
`anomaly_score` and `is_anomaly`, and in demo mode the per-session `status` is
produced by the demo generator in `data.js`. Treat the three-state status as
part of the still-to-be-implemented contract, not as something the current
backend produces.

- `GET /api/users/{user_id}/baseline` → `{ typing_speed, dwell_mean,
  flight_mean, correction_rate, rhythm_variability, pause_count, wpm,
  sample_count, updated_at }`
- `GET /api/agent/status` → `{ state: "waiting"|"session"|"uploading"|
  "paused"|"limit", session_remaining_s, sessions_today }` — planned
- `POST /api/users/{user_id}/checkins` → `{ user_id, date, factor, label, note }`
- `POST /api/users/{user_id}/symptoms` → `{ user_id, date, responses }`
- `DELETE /api/users/{user_id}/data` → clears the user's stored data

No endpoint should ever return typed message content — only the timing-derived
fields above. That is a hard privacy boundary of the project.

## Design notes

- **No raw anomaly score anywhere.** The UI speaks in states: *consistent*,
  *recent variation*, *persistent change*, *contextualized variation*,
  *further check-in recommended*, *professional evaluation recommended*.
- **Personal baseline is central.** Charts draw a dashed "your baseline"
  reference; the "What changed?" section compares the user with themselves,
  never with population averages.
- **Multi-signal logic is rule-based and explainable.** Health Insights
  combines typing state + latest check-in + reported symptoms into a calm
  interpretation with a "Why are we recommending this?" section. No fake
  percentages, no disease predictions.
- **Privacy copy is honest.** Controls are labeled Active / Demo only /
  Planned. The agent does extract features locally (see `agent/listener.py`),
  so the "local processing" claim is accurate; on-device anomaly detection is
  labeled as planned.
- **Units are explicit.** Every metric carries its unit in tiles, table cells
  and chart axes (`wpm`, `ms`, `%`, seconds for rhythm variability), and the
  chart subtitle spells the unit out in words instead of a bare symbol.
- **Accessibility is part of the same UI, not a separate mode.** Skip link;
  `aria-current` navigation; `aria-pressed` segmented controls; radio groups for
  check-ins and the symptom questionnaire; `aria-live` status, interpretation
  and tile counters; a text summary of the chart for screen readers; a
  focusable, scrollable sessions table with a sticky header that becomes cards
  on small screens; keyboard-reachable controls everywhere; and full
  `prefers-reduced-motion` support (including Chart.js animation, the agent
  progress bar and the toast).
- **Rendering is defensive.** The dashboard never assumes a baseline, a session
  field, or chart data exists — an empty history, an unconnected backend or a
  partial payload degrade to "no baseline yet" / "no sessions today yet" /
  fallback notes instead of throwing.