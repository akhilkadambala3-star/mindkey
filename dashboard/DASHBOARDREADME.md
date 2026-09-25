# MindKey Dashboard (Frontend)

The MindKey frontend — a calm, privacy-first companion for understanding your
own typing behavior over time. It is **not** a diagnostic tool and never
presents a disease risk.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | App shell: sidebar, topbar, and all views (Home, Trends, Sessions, Check-ins, Symptom check, Health Insights, Medical Assistance, Privacy, Settings). |
| `style.css` | The design system: warm-paper light theme, pine-green primary, restrained status colors, responsive layout (full sidebar → icon rail → top bar). |
| `data.js` | **The data layer.** One centralized source for demo data, localStorage persistence, and the runtime configuration that connects the real backend. The UI never reads raw data directly. |
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

Because the dashboard is simulated by default, a **Demo data** chip is shown in
the topbar whenever demo mode is active, the Monitoring card is labelled
**Simulated**, and Settings → Data source states whether the dashboard is on
simulated data or a live backend. Nothing in demo mode claims to be real
collected data, and live mode never shows simulated rows.

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

Live mode is a **runtime** setting — no file edit is required:

```
http://localhost:5500/?api=http://localhost:8000&user=<your-user-uuid>
```

Values resolve in this order (highest first):

1. **URL parameters** — `?api=<base>` and `?user=<id>`.
2. **localStorage** — keys `mindkey.apiBase` / `mindkey.userId`, so a base you
   pass once is remembered on later visits.
3. **Built-in defaults** — `null` (demo mode) and `"demo-user-1"`.

Pass `?api=demo` to return to demo mode and forget a remembered API base. The
dashboard stays in demo mode by default; live mode only activates once an API
base is configured.

Running it:

```bash
# terminal 1 — backend (from backend/)
./.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# terminal 2 — dashboard (from dashboard/)
python -m http.server 5500
```

Then open `http://localhost:5500/?api=http://localhost:8000&user=<uuid>`.

What is already wired up:

1. **Read endpoints.** `GET /api/users/{user_id}/sessions`,
   `GET /api/users/{user_id}/baseline` and
   `GET /api/users/{user_id}/investigation` (Phase 4) are implemented. The
   sessions and baseline endpoints are what the dashboard reads today; the
   investigation endpoint exists but has no dashboard view yet.
2. **Browser CORS access.** The backend sends CORS headers for the dashboard
   origin (`http://localhost:5500` and `http://127.0.0.1:5500`), `GET` only,
   with credentials disabled.

`user` must be a **UUID**, because the backend stores `user_id` as a uuid
column: a malformed value is rejected by the database and the dashboard
degrades to empty states instead of crashing. Agent status, check-ins, symptoms
and data deletion are still **planned** — the dashboard has no live path for
them, so those controls remain local/demo only.

The UI never changes between modes — every function in the data layer branches
on the resolved `API_BASE`, and live mode starts from an empty state and loads
real data before rendering (it never shows simulated rows while claiming to be
live).

### Read contract

`GET /api/users/{user_id}/sessions` — array, oldest first (implemented, Phase 4):

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
    "pause_count": 4
  }
]
```

The live contract carries timing-derived fields only. Two fields the demo
generator adds for its simulated story are **not** part of it and stay absent
in live mode: `status` (`"normal"` / `"elevated"` / `"flagged"`) and
`consistency`. The Sessions table therefore shows a neutral status, and the
Home/quality consistency figure shows "–" for live data. `duration_s` is
also absent from the contract; the dashboard derives it from
`session_start`/`session_end` for display only.

- `GET /api/users/{user_id}/baseline` → `{ typing_speed, dwell_mean,
  flight_mean, correction_rate, rhythm_variability, pause_count, wpm,
  sample_count, updated_at }` — implemented (Phase 4); a user with no baseline
  returns `sample_count: 0` with null fields
- `GET /api/users/{user_id}/investigation?session_id=&as_of=` → grounded
  investigation report (evidence, timeline, limitations, disclaimer) —
  implemented (Phase 4); no dashboard view yet
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