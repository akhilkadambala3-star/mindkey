# mindkey
MindKey is an AI-powered behavioral monitoring system that analyzes keystroke dynamics such as typing rhythm, dwell time, flight time, pauses, and correction patterns to detect persistent changes from a user’s personal baseline. It is a privacy-conscious, non-diagnostic research prototype for studying potential cognitive and neurological changes.


## Running MindKey

**Dashboard only (demo data):** `cd dashboard && python -m http.server 5500`, then open http://localhost:5500. The Demo Lab runs seven scenarios through the real investigation engine (precomputed into `dashboard/demo/`).

**Full pipeline with Supabase:**
1. Apply `backend/sql/001_checkins.sql` once in the Supabase SQL editor.
2. Put `SUPABASE_URL` and `SUPABASE_SECRET_KEY` in `backend/.env`.
3. `cd backend && pip install -r requirements.txt && uvicorn main:app --port 8000`
4. Open `http://localhost:5500/?api=http://localhost:8000&user=<user uuid>`.

**Full pipeline offline (no Supabase):**
```bash
cd backend
export MINDKEY_STORE=local              # SQLite at backend/.local/mindkey.db
python scripts/seed_local.py --scenario persistent_change   # optional demo history
uvicorn main:app --port 8000
```
Then open the dashboard with the `?api=…&user=…` link the seed script prints. The desktop agent (`agent/listener.py`) can post real sessions to this backend too.

**Deploy the API (Render, free):** Render → New → Blueprint → this repo. `render.yaml` sets everything; Render asks for `SUPABASE_URL` and `SUPABASE_SECRET_KEY` (Supabase → Project Settings → API). CORS already admits this project's Vercel URLs via `MINDKEY_CORS_ORIGIN_REGEX`. Free instances sleep after ~15 minutes idle and take ~50 seconds to wake, so open `/health` a minute before a demo.

**Point the dashboard at it:** Settings → *Connect a live backend* → paste the Render URL and a user ID. **Point the desktop agent at it:** `MINDKEY_BACKEND_URL=https://<api>.onrender.com MINDKEY_USER_ID=<uuid> python agent/listener.py`.

**Tests:** `cd backend && SUPABASE_URL=http://x.invalid SUPABASE_SECRET_KEY=x python -m pytest -q tests`
