# MindKey — working rules

- Spec: `docs/SPEC.md`. Audit and plan: `docs/AUDIT.md`.
- Dashboard is vanilla HTML/CSS/JS in `dashboard/` with no build step (static Vercel deploy). Do not add a framework.
- Honesty rules: no risk scores, no probabilities, no disease names outside disclaimers. Demo data is always labelled. Never fake agent output: the investigation UI renders only engine output (`dashboard/demo/*.json` or the API).
- After any change to `backend/investigation/agent/**`, run `python backend/scripts/export_demo.py`. The test `tests/test_demo_routes.py` fails if the static export drifts from the engine.
- Backend tests: `cd backend && SUPABASE_URL=http://x.invalid SUPABASE_SECRET_KEY=x python -m pytest -q tests`.
- Before committing UI work, check desktop (1440) and mobile (390) in light and dark, with no console errors.
- Offline mode: `MINDKEY_STORE=local` replaces Supabase with a SQLite file (`backend/.local/`). Seed it with `python backend/scripts/seed_local.py --scenario persistent_change`.
- Check-ins are agent input (day + factor only; notes never reach the investigation). Supabase needs `backend/sql/001_checkins.sql` applied once.
- The API never exposes stored column names; the investigation `engine` block uses neutral signal ids (`speed`, `dwell`, …). Privacy tests enforce this.
