# MindKey UI Upgrade: Phase 1 Audit

Branch: `ui-upgrade` · Spec: `docs/SPEC.md`

## What exists today

| Area | Finding |
|---|---|
| Framework | Vanilla HTML/CSS/JS SPA (`dashboard/`), no build step. Deployed as static files on Vercel. Live site matches the repo (same nav, same copy). |
| Pages / routing | One `index.html` with 10 `<section data-view>` views; `setView()` in `app.js` toggles them. No URL hash routing, so views can't be deep-linked. |
| State | Global `App` object in `app.js`; `data.js` is a clean data layer (demo generator + `?api=` live mode + localStorage). |
| API layer | `data.js` reads `GET /sessions` and `GET /baseline` in live mode. Check-ins, symptoms, agent status, delete are stubbed (labelled Planned). |
| Charts | Chart.js 4 from CDN; one line chart on Trends with a dashed baseline. No change points, no context overlay, no uncertainty band. |
| Demo data | 3 scenarios generated in `data.js` (consistent / recent variation / persistent change). |
| Styling | 1,600-line `style.css`, warm paper theme, pine-green primary, some CSS variables. Light theme only. |
| Responsive | Sidebar → icon rail → top bar. Sessions table becomes cards on mobile. Works. |
| Accessibility | Already strong: skip link, aria-current, aria-live, reduced motion, chart text summary. |
| Loading / errors | Defensive empty states exist; no distinct ML-down / agent-down / insufficient-data states. |
| **Backend** | FastAPI + Supabase. **601 tests pass.** Isolation Forest per-user model (`ml/`), and a complete **deterministic investigation agent** (LangGraph, hypotheses H1–H4, 11 alternative explanations, critic, grounded report, trace). `GET /api/users/{id}/investigation` is implemented but **the dashboard never calls it.** |
| Agent demo | `backend/investigation/agent/demo.py` runs the *real* engine on 5 in-memory scenarios offline. |

**Main finding:** the ML and agent are much further along than the UI shows. The largest improvement is to put the real agent output on screen, not to invent new AI.

## KEEP
- Vanilla JS, no build step (Vercel deploy stays trivial, and there's no rebuild risk)
- `data.js` data layer, runtime `?api=` live mode, demo/live labelling
- Calm light palette, Inter font, existing accessibility work
- Trends, Sessions and Check-ins logic, and the honest "Planned / Demo only" labels
- All backend code and tests (additive changes only)

## MODIFY
- **Design tokens:** consolidate into one `:root` token block (color, type scale, spacing, radius, shadow, motion, status colors, chart colors) and add a dark theme.
- **Home → Dashboard:** status hero → 4 metric cards → baseline trend → AI Investigation card → "What changed?" → Context.
- **Trends:** add 7/30/90/All ranges, change-point markers, a baseline band (±1 SD), a wellbeing-event overlay, and a composite view.
- **Sessions:** add a click-through session detail drawer (vs baseline, deviation, data quality).
- **Check-ins → Wellbeing:** add a context timeline, with the same events overlaid on charts.
- **Privacy → Privacy Center:** collects vs. does-not-collect, architecture diagram, data controls labelled Active / Demo / Planned.
- **Routing:** hash routes (`#/investigation`) so the demo can deep-link.
- **Status vocabulary:** map everything to the spec's 8 data states (NO_DATA … INSUFFICIENT_EVIDENCE).

## REPLACE
- Topbar demo dropdown → **Demo Lab** page (7 scenarios, Run Scenario, live walkthrough).
- **Medical Assistance** page (mock doctors and appointment booking) → **needs team decision.** It pushes the "medical dashboard" feel the spec says to avoid, and the fake providers are "pretend it works" UI. Proposal: remove it from nav and keep a single "Talk to a professional" resource link in Insights.

## ADD
- **AI Investigation page** (centerpiece): conclusion, animated timeline built from the real trace, evidence explorer, qualitative hypotheses (Supported / Uncertain / Weakened / Unavailable, taken directly from the engine, no percentages), evidence for/against, alternative explanations, limitations, and critic result.
- **"Why was this flagged?"** card generated from `report.observations` + `summary`.
- **Evidence Explorer:** per-signal current vs. baseline, % change, persistence, and detectors (statistical window drift, Isolation Forest, temporal persistence), each linked to evidence IDs.
- **Demo Lab + Live AI flow:** DATA → ML → AGENT → EVIDENCE → INSIGHT stepper.
- **Landing / first impression** hero with Explore Demo / View Insights.
- Loading, error and empty states for ML-unavailable, agent-unavailable, insufficient data, and no sessions.

## Backend work (additive)
1. `GET /api/demo/scenarios` and `GET /api/demo/scenarios/{key}`: run the real engine on in-memory demo data (no DB needed).
2. Add 2 scenarios to `demo.py` to match the spec's 7 (sudden change, recovery; "noisy data" maps to the existing data-quality scenario), with tests.
3. `scripts/export_demo.py` writes the real engine output to `dashboard/demo/*.json`, so **the static Vercel site shows genuine agent traces without a running backend.** The label is "Demo data · real agent output."
4. CORS: allow the Vercel origin via an env var.
5. Make `routes/supabase_test.py` import lazily so the backend and tests start without Supabase env vars.

## Honesty rules applied throughout
- No risk score, no probabilities, and no disease names outside disclaimers.
- Hypothesis bars show qualitative states only.
- Timeline animation replays **real** trace events. Replay pacing is cosmetic and labelled "replay".
- Context alternatives (sleep, stress) show **Unavailable** when the backend has no check-in store, because the engine says so. In Demo Lab, check-ins come from the demo scenario and are labelled that way.

## Build order
1. Tokens + components → 2. Backend demo endpoints + export → 3. Dashboard → 4. AI Investigation + Demo Lab → 5. Trends/Evidence + context overlay → 6. Privacy Center → 7. Sessions/Wellbeing → 8. Live wiring (`/investigation`) → 9. Responsive, accessibility and performance pass.
