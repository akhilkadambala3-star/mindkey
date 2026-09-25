/*
  MindKey — data layer (dashboard/data.js)

  Everything the UI renders comes through this file. The dashboard runs in
  DEMO MODE by default: a deterministic generator produces three scenarios
  (consistent / recent variation / persistent change) so the full product
  flow can be demonstrated without a backend.

  LIVE MODE reads real sessions and the personal baseline from the Phase 4
  read API. Point the dashboard at a running backend at runtime — no edit to
  this file needed:

      http://localhost:5500/?api=http://localhost:8000&user=<uuid>

  Resolution order for both values (highest first):
    1. URL parameters  ?api=<base>  &user=<id>   (?api=demo forces demo mode)
    2. localStorage    "mindkey.apiBase" / "mindkey.userId"
    3. built-in defaults (null = demo mode, "demo-user-1")

  Endpoint status:
    GET    /api/users/{user_id}/sessions         implemented (Phase 4)
    GET    /api/users/{user_id}/baseline         implemented (Phase 4)
    GET    /api/users/{user_id}/investigation    implemented (Phase 4)
    GET    /api/agent/status                     planned
    POST   /api/users/{user_id}/checkins         planned
    POST   /api/users/{user_id}/symptoms         planned
    DELETE /api/users/{user_id}/data             planned
*/

// Runtime configuration. Resolved at load time by resolveRuntimeConfig() below
// from URL parameters, then localStorage, then these built-in defaults.
// API_BASE = null keeps the dashboard in demo mode with simulated data.
let API_BASE = null; // e.g. "http://localhost:8000" to read from a live backend.
let USER_ID = "demo-user-1";

/* --------------------------------------------------------------------------
   Agent parameters mirrored for the demo (single frontend source)
   --------------------------------------------------------------------------
   These are NOT new settings invented by the frontend. The single source of
   truth remains the real desktop agent in agent/listener.py
   (SESSION_DURATION_S = 20.0, DAILY_SESSION_LIMIT = 100).

   They are mirrored here so the demo agent simulator, the "Sessions today"
   tile and every piece of daily-limit copy in the UI stay consistent with the
   real agent and with each other. If the agent's values ever change, update
   these two numbers and the whole frontend follows.
   -------------------------------------------------------------------------- */

const AGENT_SESSION_DURATION_S = 20; // mirrors SESSION_DURATION_S in agent/listener.py
const AGENT_DAILY_SESSION_LIMIT = 100; // mirrors DAILY_SESSION_LIMIT in agent/listener.py

/** True when the dashboard is showing simulated data instead of a live backend. */
const demoMode = () => !API_BASE;

/* --------------------------------------------------------------------------
   Local storage (tiny safe wrapper)
   -------------------------------------------------------------------------- */

const store = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage unavailable (private mode etc.) — demo still works in memory */
    }
  },
  remove(key) {
    try {
      localStorage.removeItem(key);
    } catch {
      /* ignore */
    }
  },
  clear() {
    try {
      Object.keys(localStorage)
        .filter((k) => k.startsWith("mindkey."))
        .forEach((k) => localStorage.removeItem(k));
    } catch {
      /* ignore */
    }
  },
};

const KEYS = {
  checkins: "mindkey.checkins",
  symptoms: "mindkey.symptoms",
  appointment: "mindkey.appointment",
  apiBase: "mindkey.apiBase",
  userId: "mindkey.userId",
};

/* --------------------------------------------------------------------------
   Runtime configuration
   --------------------------------------------------------------------------
   The dashboard can be pointed at a live backend without editing this file.
   Precedence for API_BASE and USER_ID (highest first):
     1. URL query parameters   ?api=<base>&user=<id>
     2. localStorage           "mindkey.apiBase" / "mindkey.userId"
     3. built-in defaults      (null = demo mode, "demo-user-1")
   ?api=demo (or ?api=) forces demo mode and forgets a remembered API base.
   -------------------------------------------------------------------------- */

function resolveRuntimeConfig() {
  const params = new URLSearchParams(window.location.search);
  const apiParam = params.get("api");
  const userParam = params.get("user");

  if (apiParam !== null) {
    const value = apiParam.trim();
    if (value === "" || value.toLowerCase() === "demo") {
      // Explicitly return to demo mode and drop any remembered backend base.
      store.remove(KEYS.apiBase);
      API_BASE = null;
    } else {
      const base = value.replace(/\/+$/, "");
      store.set(KEYS.apiBase, base);
      API_BASE = base;
    }
  } else {
    const stored = store.get(KEYS.apiBase, null);
    API_BASE =
      typeof stored === "string" && stored.trim() !== ""
        ? stored.trim().replace(/\/+$/, "")
        : null;
  }

  if (userParam !== null && userParam.trim() !== "") {
    store.set(KEYS.userId, userParam.trim());
    USER_ID = userParam.trim();
  } else {
    const stored = store.get(KEYS.userId, null);
    USER_ID = typeof stored === "string" && stored.trim() !== "" ? stored.trim() : "demo-user-1";
  }
}

resolveRuntimeConfig();

/* --------------------------------------------------------------------------
   Deterministic mock data
   -------------------------------------------------------------------------- */

// Small seeded PRNG so a scenario always produces the same story.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// Personal baseline — roughly the "normal" range of a typical user in the
// MindKey pipeline (agent/features.py + ml/features.py). These are timing
// aggregates only, never typed content.
const BASE_PROFILE = {
  typing_speed: 285, // characters per minute
  dwell_mean: 0.12, // seconds key held down
  flight_mean: 0.08, // seconds between keys
  correction_rate: 0.06, // 0..1
  rhythm_variability: 0.19,
  pause_count: 4,
};

const SCENARIOS = {
  consistent: {
    label: "Consistent pattern",
    driftDays: 0,
    driftPeak: 0,
  },
  variation: {
    label: "Recent variation",
    driftDays: 5,
    driftPeak: 0.5,
  },
  persistent: {
    label: "Persistent change",
    driftDays: 16,
    driftPeak: 1.0,
  },
};

function daysAgoISO(days, rand, hour) {
  const d = new Date();
  d.setDate(d.getDate() - days);
  d.setHours(hour, Math.floor(rand() * 60), 0, 0);
  return d;
}

/**
 * Generate the session history for a scenario.
 * Sessions are returned oldest → newest. Today always has 4 sessions so the
 * "Sessions today: 4 / 10" snapshot reads naturally in every scenario.
 */
function generateSessions(scenarioKey) {
  const scenario = SCENARIOS[scenarioKey] || SCENARIOS.consistent;
  const rand = mulberry32(hashString(USER_ID + ":" + scenarioKey + ":v2"));

  const sessions = [];
  const DAYS = 90;
  const today = new Date();

  const isToday = (d) =>
    d.getFullYear() === today.getFullYear() &&
    d.getMonth() === today.getMonth() &&
    d.getDate() === today.getDate();

  for (let daysBack = DAYS - 1; daysBack >= 0; daysBack--) {
    const base = new Date();
    base.setDate(base.getDate() - daysBack);

    // Weekend effect: fewer sessions on Saturdays and Sundays.
    const weekend = base.getDay() === 0 || base.getDay() === 6;
    let count = weekend ? (rand() < 0.5 ? 1 : 2) : 2 + Math.floor(rand() * 2);
    if (rand() < 0.08) count = 0; // some quiet days
    if (isToday(base)) count = 4; // the demo story needs a "today"

    for (let i = 0; i < count; i++) {
      // Drift factor: 0 = exactly at the personal baseline.
      let drift = 0;
      if (scenario.driftDays > 0 && daysBack < scenario.driftDays) {
        const progress = (scenario.driftDays - daysBack) / scenario.driftDays;
        drift = Math.min(scenario.driftPeak, progress * scenario.driftPeak);
      }
      const noise = () => (rand() - 0.5) * 0.08;

      const typing_speed = Math.max(120, BASE_PROFILE.typing_speed * (1 - drift * 0.16 + noise()));
      const dwell_mean = Math.max(0.05, BASE_PROFILE.dwell_mean * (1 + drift * 0.2 + noise()));
      const flight_mean = Math.max(0.03, BASE_PROFILE.flight_mean * (1 + drift * 0.24 + noise()));
      const correction_rate = Math.min(0.2, Math.max(0.02, BASE_PROFILE.correction_rate * (1 + drift * 1.5 + noise())));
      const rhythm_variability = Math.max(0.08, BASE_PROFILE.rhythm_variability * (1 + drift * 0.8 + noise()));
      const pause_count = Math.max(1, Math.round(BASE_PROFILE.pause_count * (1 + drift * 1.1 + noise())));

      const consistency = Math.max(55, Math.min(99, Math.round(97 - drift * 30 + (rand() - 0.5) * 4)));

      let status = "normal";
      if (drift > 0.35 && drift <= 0.75) status = "elevated";
      if (drift > 0.75) status = "flagged";

      const start = daysAgoISO(daysBack, rand, 9 + Math.floor(rand() * 13));
      const durationS = 120 + Math.round((rand() - 0.5) * 12);
      const end = new Date(start.getTime() + durationS * 1000);

      sessions.push({
        session_id: "S" + String(sessions.length + 1).padStart(4, "0"),
        session_start: start.toISOString(),
        session_end: end.toISOString(),
        date: start.toISOString().slice(0, 10),
        typing_speed: Math.round(typing_speed),
        wpm: Math.round(typing_speed / 5),
        dwell_mean_ms: Math.round(dwell_mean * 1000),
        flight_mean_ms: Math.round(flight_mean * 1000),
        correction_rate: Math.round(correction_rate * 1000) / 1000,
        rhythm_variability: Math.round(rhythm_variability * 1000) / 1000,
        pause_count,
        consistency,
        status,
        duration_s: durationS,
      });
    }
  }

  const baseline = computeBaseline(sessions);
  return { sessions, baseline, scenario: scenarioKey };
}

/** Personal baseline = average of the user's own stable period (first ~70% of history). */
function computeBaseline(sessions) {
  const stable = sessions.slice(0, Math.floor(sessions.length * 0.7));
  if (stable.length === 0) return { ...BASE_PROFILE, wpm: Math.round(BASE_PROFILE.typing_speed / 5), sample_count: 0, updated_at: null };

  const mean = (key) =>
    stable.reduce((sum, s) => sum + s[key], 0) / stable.length;

  return {
    typing_speed: Math.round(mean("typing_speed")),
    dwell_mean: mean("dwell_mean_ms") / 1000,
    flight_mean: mean("flight_mean_ms") / 1000,
    correction_rate: mean("correction_rate"),
    rhythm_variability: mean("rhythm_variability"),
    pause_count: Math.round(mean("pause_count")),
    wpm: Math.round(mean("typing_speed") / 5),
    sample_count: stable.length,
    updated_at: new Date().toISOString(),
  };
}

/* --------------------------------------------------------------------------
   App state (demo mode)
   -------------------------------------------------------------------------- */

const DemoState = {
  scenario: "variation",
  sessions: [],
  baseline: null,
};

function setScenario(key) {
  DemoState.scenario = SCENARIOS[key] ? key : "variation";
  const { sessions, baseline } = generateSessions(DemoState.scenario);
  DemoState.sessions = sessions;
  DemoState.baseline = baseline;
}

/* --------------------------------------------------------------------------
   API layer — one function per backend capability.
   Demo mode returns local data; API mode talks to the backend.
   -------------------------------------------------------------------------- */

/**
 * The session read contract omits duration_s, so derive it for display from the
 * start/end timestamps. Rows with missing or unusable timestamps are returned
 * untouched, so a partial payload can never produce a NaN duration.
 */
function withDerivedDuration(session) {
  if (!session || typeof session !== "object") return session;
  if (typeof session.duration_s === "number") return session;
  const start = Date.parse(session.session_start);
  const end = Date.parse(session.session_end);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return session;
  return { ...session, duration_s: Math.round((end - start) / 1000) };
}

async function getSessions() {
  if (API_BASE) {
    const res = await fetch(`${API_BASE}/api/users/${USER_ID}/sessions`);
    if (!res.ok) return [];
    const data = await res.json().catch(() => null);
    // Never let an error body or a non-array payload reach the views.
    return Array.isArray(data) ? data.map(withDerivedDuration) : [];
  }
  return DemoState.sessions;
}

async function getBaseline() {
  if (API_BASE) {
    const res = await fetch(`${API_BASE}/api/users/${USER_ID}/baseline`);
    if (!res.ok) return null;
    const data = await res.json().catch(() => null);
    // An error body ({"detail": ...}) or any non-baseline shape must never be
    // passed downstream as if it were a real baseline.
    return data && typeof data === "object" && !Array.isArray(data) && "sample_count" in data
      ? data
      : null;
  }
  return DemoState.baseline;
}

/** Agent status. Planned endpoint — in demo mode the UI runs a local simulator. */
async function getAgentStatus() {
  if (API_BASE) {
    // const res = await fetch(`${API_BASE}/api/agent/status`);
    // return res.json();
  }
  return null;
}

async function submitCheckin(payload) {
  if (API_BASE) {
    // await fetch(`${API_BASE}/api/users/${USER_ID}/checkins`, {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify(payload),
    // });
    return payload;
  }
  const checkins = store.get(KEYS.checkins, []);
  checkins.push(payload);
  store.set(KEYS.checkins, checkins);
  return payload;
}

async function submitSymptoms(payload) {
  if (API_BASE) {
    // await fetch(`${API_BASE}/api/users/${USER_ID}/symptoms`, {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify(payload),
    // });
    return payload;
  }
  store.set(KEYS.symptoms, payload);
  return payload;
}

function getCheckins() {
  return store.get(KEYS.checkins, []);
}

function getSymptoms() {
  return store.get(KEYS.symptoms, null);
}

async function deleteAllData() {
  if (API_BASE) {
    // await fetch(`${API_BASE}/api/users/${USER_ID}/data`, { method: "DELETE" });
  }
  store.clear();
}

/* --------------------------------------------------------------------------
   Scenario metadata + deterministic helper exports
   -------------------------------------------------------------------------- */

const CHECKIN_OPTIONS = [
  { id: "feeling_well", label: "Feeling well" },
  { id: "tired", label: "Tired" },
  { id: "stressed", label: "Stressed" },
  { id: "poor_sleep", label: "Poor sleep" },
  { id: "unwell", label: "Feeling unwell" },
  { id: "distracted", label: "Distracted / busy" },
  { id: "other", label: "Something else" },
];

const SYMPTOM_GROUPS = [
  {
    group: "Motor",
    items: [
      { id: "tremor", label: "Shaking or tremor in hands or fingers" },
      { id: "stiffness", label: "Stiffness in hands, arms, or elsewhere" },
      { id: "slow_movement", label: "Unusually slow movements" },
      { id: "coordination", label: "Difficulty with coordination or fine motor tasks" },
    ],
  },
  {
    group: "Cognitive",
    items: [
      { id: "memory", label: "Memory difficulties" },
      { id: "concentration", label: "Concentration difficulties" },
      { id: "word_finding", label: "Difficulty finding the right word" },
    ],
  },
  {
    group: "General",
    items: [
      { id: "fatigue", label: "Fatigue" },
      { id: "sleep_changes", label: "Changes in sleep" },
    ],
  },
];