/*
  MindKey — frontend logic (dashboard/app.js)

  A small vanilla-JS single-page app. All data comes through the layer in
  data.js (demo mode by default; swap API_BASE there to connect the backend).

  Structure:
    - Router / views
    - Typing-state computation (personal baseline → status)
    - Agent status simulator (demo)
    - Per-view renderers
    - Chart helpers (Chart.js, graceful fallback offline)
*/

"use strict";

const $ = (id) => document.getElementById(id);

const VIEW_META = {
  home: ["Dashboard", "Your personal behavioral overview"],
  investigation: ["AI Investigation", "How the agent reached its conclusion — step by step"],
  lab: ["Demo Lab", "Run a scenario and watch Data → ML → Agent → Insight"],
  trends: ["Trends", "How your typing pattern has changed over time"],
  sessions: ["Sessions", "A record of your analyzed typing sessions"],
  checkins: ["Wellbeing", "Context that helps explain variation"],
  symptoms: ["Symptom check", "A few additional questions"],
  insights: ["Insights", "What your typing pattern shows, in plain words"],
  privacy: ["Privacy Center", "What we collect — and what we never see"],
  settings: ["Profile & settings", "Profile, data, and preferences"],
};

const METRICS = {
  speed: {
    label: "Typing speed",
    unit: "wpm",
    get: (s) => (isNumber(s.wpm) ? Math.round(s.wpm * 10) / 10 : s.wpm),
    base: (b) => b.wpm,
  },
  dwell: {
    label: "Dwell time",
    unit: "ms",
    get: (s) => (isNumber(s.dwell_mean_ms) ? Math.round(s.dwell_mean_ms) : s.dwell_mean_ms),
    base: (b) => Math.round(b.dwell_mean * 1000),
  },
  flight: {
    label: "Flight time",
    unit: "ms",
    get: (s) => (isNumber(s.flight_mean_ms) ? Math.round(s.flight_mean_ms) : s.flight_mean_ms),
    base: (b) => Math.round(b.flight_mean * 1000),
  },
  corrections: {
    label: "Correction rate",
    unit: "%",
    get: (s) => Math.round(s.correction_rate * 1000) / 10,
    base: (b) => Math.round(b.correction_rate * 1000) / 10,
  },
  pauses: {
    label: "Pauses",
    unit: "",
    get: (s) => s.pause_count,
    base: (b) => b.pause_count,
  },
  rhythm: {
    label: "Rhythm variability",
    unit: "s",
    get: (s) => s.rhythm_variability,
    base: (b) => Math.round(b.rhythm_variability * 100) / 100,
  },
};

/* --------------------------------------------------------------------------
   App state
   -------------------------------------------------------------------------- */

const App = {
  view: "home",
  lastView: "home",
  range: 30,
  metric: "speed",
  sessionRange: "all",
  sessions: [],
  baseline: null,
  chart: null,
  agent: {
    state: "waiting", // waiting | session | uploading | paused | limit
    sessionCountToday: 4,
    timers: [],
  },
  checkin: { selected: null },
  symptoms: {},
};

/* --------------------------------------------------------------------------
   Small helpers
   -------------------------------------------------------------------------- */

function showToast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => t.classList.remove("show"), 3200);
}

function fmtTime(iso) {
  const d = new Date(iso);
  let h = d.getHours();
  const m = String(d.getMinutes()).padStart(2, "0");
  const ap = h >= 12 ? "PM" : "AM";
  h = h % 12 || 12;
  return `${h}:${m} ${ap}`;
}

function fmtDay(iso) {
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function fmtFull(iso) {
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function fmtDuration(sec) {
  const m = Math.floor(sec / 60);
  const s = String(Math.round(sec % 60)).padStart(2, "0");
  return `${m}:${s}`;
}

function isTodayISO(iso) {
  const d = new Date(iso);
  const now = new Date();
  return (
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate()
  );
}

function mean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

const isNumber = (v) => typeof v === "number" && Number.isFinite(v);

/**
 * Baseline accessors. The dashboard must never assume a baseline (or a given
 * baseline metric) exists: an empty history, an unconnected backend or a
 * partial payload must degrade gracefully instead of throwing.
 */
function baselineValue(key) {
  return App.baseline && isNumber(App.baseline[key]) ? App.baseline[key] : null;
}

const prefersReducedMotion = () =>
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Loading affordance for refreshes. Demo data resolves instantly, so this is
 * deliberately cheap (a body class plus a busy refresh button); it never
 * blocks rendering if something goes wrong.
 */
function setLoading(on) {
  const busy = !!on;
  document.body.classList.toggle("is-loading", busy);
  const btn = $("refreshBtn");
  if (btn) {
    btn.disabled = busy;
    btn.setAttribute("aria-busy", busy ? "true" : "false");
  }
}

// Human noun forms for check-in factors, so sentences read naturally.
const CONTEXT_NOUNS = {
  feeling_well: "feeling well",
  tired: "tiredness",
  stressed: "stress",
  poor_sleep: "poor sleep",
  unwell: "feeling unwell",
  distracted: "being busy or distracted",
  other: "something else",
};

const STATUS_META = {
  recorded: {
    label: "Recorded",
    pill: "pill-neutral",
    help: "Stored session. Individual sessions are not classified — MindKey only draws conclusions across many sessions.",
  },
  ml_flag: {
    label: "ML flag",
    pill: "pill-flag",
    help: "The anomaly model, trained only on your own earlier sessions, found this session unusual for you. A flag is not a diagnosis.",
  },
  invalid: {
    label: "Invalid",
    pill: "pill-invalid",
    help: "This session failed feature validation and is excluded from the analysis.",
  },
  normal: {
    label: "Consistent",
    pill: "pill-ok",
    help: "This session stayed close to your personal baseline.",
  },
  elevated: {
    label: "Variation",
    pill: "pill-warn",
    help: "This session differed from your personal baseline, but not enough to conclude anything.",
  },
  flagged: {
    label: "Change",
    pill: "pill-alert",
    help: "This session contributed to a persistent change — several indicators shifted together.",
  },
};

/* --------------------------------------------------------------------------
   Typing-state computation (personal baseline logic, mirrored from the ML
   persistence rule: a single unusual session never triggers anything)
   -------------------------------------------------------------------------- */

function typingState(sessions) {
  if (typeof MK !== "undefined" && MK.state.payload) return MK.legacyTypingState();
  const tail = sessions.slice(-8);
  const flagged = tail.filter((s) => s.status === "flagged").length;
  const elevated = tail.filter((s) => s.status === "elevated").length;
  if (flagged >= 2 && flagged + elevated >= 4) return "alert";
  if (elevated + flagged >= 2) return "warn";
  return "ok";
}

/** Relative change of the most recent chunk vs. the rest of the window. */
function recentChange(sessions, metric) {
  if (sessions.length < 6) return 0;
  const recent = sessions.slice(-5);
  const earlier = sessions.slice(0, -5);
  const r = mean(recent.map(metric.get));
  const e = mean(earlier.map(metric.get));
  if (e === 0) return 0;
  return (r - e) / e;
}

function describeChanges(sessions, rangeDays) {
  const window = sessions.slice(-rangeDays);
  if (window.length < 6) return [];
  const changes = [];

  const speedDelta = recentChange(window, METRICS.speed);
  if (Math.abs(speedDelta) >= 0.05) {
    changes.push({
      metric: "speed",
      dir: speedDelta > 0 ? "increased" : "decreased",
      text: `Your average typing speed has ${speedDelta > 0 ? "increased" : "decreased"} compared with earlier in this period.`,
    });
  }

  const dwellDelta = recentChange(window, METRICS.dwell);
  if (Math.abs(dwellDelta) >= 0.05) {
    changes.push({
      metric: "dwell",
      text: `Keys are held ${dwellDelta > 0 ? "longer" : "more briefly"} on average than your recent baseline.`,
    });
  }

  const flightDelta = recentChange(window, METRICS.flight);
  if (Math.abs(flightDelta) >= 0.05) {
    changes.push({
      metric: "flight",
      text: `Gaps between keys have ${flightDelta > 0 ? "grown" : "shrunk"} compared with your recent baseline.`,
    });
  }

  const corrDelta = recentChange(window, METRICS.corrections);
  if (Math.abs(corrDelta) >= 0.05) {
    changes.push({
      metric: "corrections",
      text: `Correction rate has ${corrDelta > 0 ? "increased" : "decreased"} compared with your recent baseline.`,
    });
  }

  const pauseDelta = recentChange(window, METRICS.pauses);
  if (Math.abs(pauseDelta) >= 0.05) {
    changes.push({
      metric: "pauses",
      text: `Pauses have ${pauseDelta > 0 ? "become more frequent" : "become less frequent"} compared with your recent baseline.`,
    });
  }

  const rhythmDelta = recentChange(window, METRICS.rhythm);
  if (Math.abs(rhythmDelta) >= 0.05) {
    changes.push({
      metric: "rhythm",
      text: `Your typing rhythm is ${rhythmDelta > 0 ? "more variable" : "more even"} than usual — ${rhythmDelta > 0 ? "less" : "more"} consistent timing between keys.`,
    });
  }

  return changes;
}

/* --------------------------------------------------------------------------
   Router
   -------------------------------------------------------------------------- */

let firstRender = true;

const VIEW_RENDERERS = () => ({
  home: renderHome,
  investigation: () => MK.renderInvestigation(),
  lab: () => MK.renderLab(),
  trends: renderTrends,
  sessions: renderSessions,
  checkins: renderCheckins,
  symptoms: renderSymptoms,
  insights: renderInsights,
  privacy: renderPrivacy,
  settings: renderSettings,
});

function rerenderCurrentView() {
  VIEW_RENDERERS()[App.view]();
}

function setView(name, { fromHash = false } = {}) {
  if (!VIEW_META[name]) name = "home";
  if (!fromHash && location.hash !== "#/" + name) {
    history.pushState(null, "", "#/" + name);
  }
  App.lastView = App.view;
  App.view = name;

  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === name));
  document.querySelectorAll(".nav-item").forEach((n) => {
    const active = n.dataset.view === name;
    n.classList.toggle("active", active);
    // aria-current keeps screen readers on the same page as the visuals.
    if (active) {
      n.setAttribute("aria-current", "page");
    } else {
      n.removeAttribute("aria-current");
    }
  });

  const [title, subtitle] = VIEW_META[name];
  $("pageTitle").textContent = title;
  $("pageSubtitle").textContent = subtitle;

  VIEW_RENDERERS()[name]();
  window.scrollTo({ top: 0 });

  // Move focus to the new view's heading so keyboard and screen-reader users
  // are not left behind in the old view. Skipped on first paint.
  if (!firstRender) {
    const heading = $("pageTitle");
    if (heading && typeof heading.focus === "function") {
      heading.focus({ preventScroll: true });
    }
  }
  firstRender = false;
}

/* --------------------------------------------------------------------------
   Agent simulator (demo mode only — the backend agent reports no live status)
   -------------------------------------------------------------------------- */

const AGENT_STATES = {
  waiting: { chip: "Monitoring", dot: "ok live", state: "Waiting for typing", dotClass: "ok" },
  session: { chip: "Analyzing…", dot: "ok live", state: "Analyzing typing session", dotClass: "ok" },
  uploading: { chip: "Uploading…", dot: "ok", state: "Uploading session", dotClass: "ok" },
  paused: { chip: "Paused", dot: "paused", state: "Monitoring paused", dotClass: "paused" },
  limit: { chip: "Limit reached", dot: "warn", state: "Daily limit reached", dotClass: "warn" },
};

function agentLine() {
  const a = App.agent;
  switch (a.state) {
    case "session":
      return `Analyzing typing session — ${fmtDuration(a.sessionRemaining || 0)} remaining`;
    case "uploading":
      return "Uploading the session to your secure store…";
    case "paused":
      return "Monitoring is paused. Nothing is being collected right now.";
    case "limit":
      return `You've reached the simulated daily limit of ${AGENT_DAILY_SESSION_LIMIT} sessions. Monitoring resumes tomorrow.`;
    default:
      return "MindKey listens for your next typing activity.";
  }
}

function clearAgentTimers() {
  App.agent.timers.forEach((t) => {
    clearTimeout(t);
    clearInterval(t);
  });
  App.agent.timers = [];
}

function renderAgent() {
  const a = App.agent;
  const meta = AGENT_STATES[a.state];

  // Chips (sidebar + topbar)
  [["sidebarAgentDot", "sidebarAgentText"], ["topbarAgentDot", "topbarAgentText"]].forEach(([dotId, textId]) => {
    $(dotId).className = "agent-dot " + meta.dot;
    $(textId).textContent = meta.chip;
  });

  // Home card
  const progress = $("agentProgress");
  const stateText = $("agentStateText");
  stateText.textContent = meta.state;
  stateText.className = "agent-state " + meta.dotClass;
  $("agentLine").textContent = agentLine();
  $("agentSessionsToday").textContent = a.sessionCountToday;

  if (a.state === "session") {
    const pct = Math.max(
      0,
      Math.min(100, (1 - a.sessionRemaining / a.sessionDuration) * 100)
    );
    progress.hidden = false;
    progress.setAttribute("role", "progressbar");
    progress.setAttribute("aria-label", "Simulated typing session progress");
    progress.setAttribute("aria-valuemin", "0");
    progress.setAttribute("aria-valuemax", "100");
    progress.setAttribute("aria-valuenow", String(Math.round(pct)));
    $("agentProgressFill").style.width = pct.toFixed(1) + "%";
  } else {
    progress.hidden = true;
    progress.removeAttribute("aria-valuenow");
  }

  // Pause button (home + privacy)
  const pauseLabel = a.state === "paused" ? "Resume monitoring" : "Pause monitoring";
  $("agentPauseBtn").textContent = pauseLabel;
  $("privacyPauseBtn").textContent = pauseLabel;
  $("privacyPausePill").textContent = a.state === "paused" ? "Paused" : "Active";

  // Today tile + the daily-limit copy. Both read the one mirrored constant so
  // the tile, the agent card and the limit message can never drift apart.
  if ($("todaySessions")) $("todaySessions").textContent = a.sessionCountToday;
  if ($("todaySessionsLimit")) $("todaySessionsLimit").textContent = AGENT_DAILY_SESSION_LIMIT;
  $("agentSessionsLimit").textContent = AGENT_DAILY_SESSION_LIMIT;
}

function agentPauseToggle() {
  const a = App.agent;
  if (a.state === "paused") {
    a.state = "waiting";
    renderAgent();
    agentSchedule();
  } else {
    clearAgentTimers();
    a.state = "paused";
    renderAgent();
  }
}

function agentSchedule() {
  const a = App.agent;
  if (a.state === "paused" || a.state === "limit") return;
  const wait = 22000 + Math.random() * 28000;
  a.timers.push(setTimeout(agentBeginSession, wait));
}

function agentBeginSession() {
  const a = App.agent;
  if (a.state === "paused" || a.state === "limit") return;
  a.state = "session";
  // Mirrors the real agent's session length (agent/listener.py) so the demo
  // shows the true 20-second window instead of an invented duration.
  a.sessionDuration = AGENT_SESSION_DURATION_S;
  a.sessionRemaining = AGENT_SESSION_DURATION_S;
  renderAgent();

  const tick = setInterval(() => {
    a.sessionRemaining -= 0.25;
    if (a.sessionRemaining <= 0) {
      clearInterval(tick);
      agentEndSession();
    } else {
      renderAgent();
    }
  }, 250);
  a.timers.push(tick);
}

function agentEndSession() {
  const a = App.agent;
  a.sessionCountToday += 1;
  if (a.sessionCountToday >= AGENT_DAILY_SESSION_LIMIT) {
    a.state = "limit";
    renderAgent();
    showToast("Daily analysis limit reached for the demo — monitoring resumes tomorrow.");
    return;
  }
  a.state = "uploading";
  renderAgent();
  a.timers.push(
    setTimeout(() => {
      if (a.state === "paused") return;
      a.state = "waiting";
      renderAgent();
      agentSchedule();
    }, 2600)
  );
}

function agentResetDay() {
  clearAgentTimers();
  App.agent.sessionCountToday = 4;
  App.agent.state = "waiting";
  renderAgent();
  agentSchedule();
}


/* --------------------------------------------------------------------------
   Home
   -------------------------------------------------------------------------- */

function greeting() {
  const h = new Date().getHours();
  if (h < 5) return "Good evening";
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

function renderHome() {
  MK.renderDashboard();
  MK.scenarioChip();
  renderAgent();
}

function vsBaseline(value, base, unit) {
  const delta = value - base;
  if (Math.abs(delta) < 0.5) return `about your baseline (${base} ${unit})`;
  const dir = delta > 0 ? "above" : "below";
  return `${Math.abs(Math.round(delta))} ${unit} ${dir} your baseline (${base} ${unit})`;
}

/** Human list of what changed vs. the personal baseline (persistent state). */
function observedChanges() {
  const recent = App.sessions.slice(-5);
  if (!App.baseline || !recent.length) {
    return ["Several typing metrics have shifted from your personal baseline"];
  }

  // mean() of an empty/filtered list is 0, which would read as "slower than
  // usual", so every value is filtered to real numbers first and every
  // comparison is skipped when its baseline metric is missing.
  const m = (key) => mean(recent.map((s) => s[key]).filter(isNumber));
  const items = [];

  const bWpm = baselineValue("wpm");
  if (bWpm != null && m("wpm") < bWpm * 0.94) items.push("Slower typing than your usual pace");

  const bDwell = baselineValue("dwell_mean");
  if (bDwell != null && m("dwell_mean_ms") > bDwell * 1000 * 1.06)
    items.push("Keys held slightly longer than usual");

  const bFlight = baselineValue("flight_mean");
  if (bFlight != null && m("flight_mean_ms") > bFlight * 1000 * 1.06)
    items.push("Longer gaps between keys");

  const bPauses = baselineValue("pause_count");
  if (bPauses != null && m("pause_count") > bPauses * 1.2) items.push("More pauses while typing");

  const bCorr = baselineValue("correction_rate");
  if (bCorr != null && m("correction_rate") > bCorr * 1.15)
    items.push("More corrections than usual");

  const bRhythm = baselineValue("rhythm_variability");
  if (bRhythm != null && m("rhythm_variability") > bRhythm * 1.12)
    items.push("Less even typing rhythm");

  return items.length ? items : ["Several typing metrics have shifted from your personal baseline"];
}

/* --------------------------------------------------------------------------
   Trends
   -------------------------------------------------------------------------- */

function metricSeries(sessions, metric) {
  return sessions.map(metric.get);
}

/*
  Chart teardown helper. Every exit path calls it, so a chart from a previous
  range/scenario can never be left behind on the canvas when the current
  window has too little data or Chart.js is unavailable.
*/
function destroyChart() {
  if (App.chart) {
    App.chart.destroy();
    App.chart = null;
  }
}

/* Human names for the axis units, so the chart subtitle never reads as a bare
   symbol ("ms" → "milliseconds"). */
const UNIT_LABELS = {
  wpm: "words per minute",
  ms: "milliseconds",
  "%": "percent of keystrokes",
  s: "seconds (standard deviation)",
  "": "count per session",
};

/* Text equivalent of the chart, announced to screen readers. */
function setChartSummary(sessions, metric, base) {
  const el = $("chartSummary");
  if (!el) return;
  if (!sessions || !sessions.length) {
    el.textContent = "No chart data for this period.";
    return;
  }
  const unit = metric.unit ? ` ${metric.unit}` : "";
  const latest = metric.get(sessions[sessions.length - 1]);
  const parts = [
    `${sessions.length} sessions, ${rangeLabel().toLowerCase()}.`,
    `Latest ${metric.label.toLowerCase()}: ${latest}${unit}.`,
  ];
  if (isNumber(base)) parts.push(`Your baseline: ${base}${unit}.`);
  el.textContent = parts.join(" ");
}

function rangeLabel() {
  return App.range >= 9999 ? "All time" : `Last ${App.range} days`;
}

const cssToken = (name, fallback) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

function renderTrends() {
  const metric = METRICS[App.metric];
  $("chartTitle").textContent = metric.label;
  $("chartSub").textContent = `${rangeLabel()} · ${UNIT_LABELS[metric.unit] || metric.label.toLowerCase()}`;
  $("whatChangedRange").textContent = rangeLabel();

  // What changed? — data-driven, independent of charts.
  const changes = describeChanges(App.sessions, App.range);
  const list = $("changeList");
  if (!changes.length) {
    list.innerHTML = `<li class="change-empty">No meaningful change detected in this period — your pattern is holding close to your personal baseline.</li>`;
  } else {
    list.innerHTML = changes.map((c) => `<li class="change-warn">${c.text}</li>`).join("");
  }

  // Chart. Every exit path tears the previous chart down first.
  const chartWindow = App.sessions.filter((s) => s.status !== "invalid").slice(-App.range);
  const fallback = $("chartFallback");

  if (!App.chartOk) {
    destroyChart();
    fallback.textContent =
      "The chart couldn't be drawn. The same numbers are in \"What changed?\" below.";
    fallback.hidden = false;
    setChartSummary([], metric, null);
    return;
  }

  if (chartWindow.length < 2) {
    destroyChart();
    fallback.textContent = "Not enough sessions yet to draw a chart.";
    fallback.hidden = false;
    setChartSummary([], metric, null);
    return;
  }

  fallback.hidden = true;

  const labels = chartWindow.map((s) => fmtDay(s.session_start));
  const data = metricSeries(chartWindow, metric);
  // A missing/partial baseline simply means no dashed reference line.
  const base = App.baseline ? metric.base(App.baseline) : null;
  const baseline = isNumber(base) ? Array(chartWindow.length).fill(base) : null;

  App.chart = buildChart({
    sessions: chartWindow,
    labels,
    data,
    baseline,
    unit: metric.unit,
    color: cssToken("--chart-1", "#2F6B4F"),
  });
  setChartSummary(chartWindow, metric, base);
}

function buildChart({ sessions, labels, data, baseline, unit, color }) {
  destroyChart();
  const ctx = $("trendChart");
  ctx.setAttribute("role", "img");
  ctx.setAttribute(
    "aria-label",
    `${METRICS[App.metric].label}, ${rangeLabel().toLowerCase()}, compared with your personal baseline.`
  );

  const datasets = [
    {
      label: "Sessions",
      data,
      borderColor: color,
      backgroundColor: cssToken("--chart-1-fill", "rgba(47, 107, 79, 0.05)"),
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: color,
      tension: 0.3,
      fill: true,
    },
  ];
  // The dashed baseline reference exists only when a baseline exists.
  if (baseline) {
    datasets.push({
      label: "Your baseline",
      data: baseline,
      borderColor: cssToken("--chart-baseline", "#B9B4A9"),
      borderWidth: 1.5,
      borderDash: [5, 5],
      pointRadius: 0,
      fill: false,
    });
  }

  const markers = typeof MK !== "undefined" ? MK.markersFor(sessions) : [];
  return new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    plugins: typeof MK !== "undefined" ? [MK.markerPlugin] : [],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: prefersReducedMotion() ? false : undefined,
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: { display: false },
        mkMarkers: { items: markers },
        tooltip: {
          backgroundColor: cssToken("--surface", "#FFFFFF"),
          titleColor: cssToken("--text", "#26231D"),
          bodyColor: cssToken("--text-2", "#5C584F"),
          borderColor: cssToken("--border", "#E6E3DC"),
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          displayColors: false,
          callbacks: {
            title: (items) => {
              // Read the exact window that was rendered, not a fresh slice of
              // application state.
              const s = sessions[items[0].dataIndex];
              return s ? `${fmtFull(s.session_start)} · ${fmtTime(s.session_start)}` : "";
            },
            label: (item) => {
              const suffix = unit ? ` ${unit}` : "";
              return `${item.dataset.label === "Your baseline" ? "Your baseline" : "Sessions"}: ${Number(item.parsed.y.toFixed(2))}${suffix}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { color: cssToken("--border", "#E6E3DC") },
          ticks: { color: cssToken("--chart-axis", "#8B857A"), font: { family: "Inter", size: 11 }, maxTicksLimit: 9, maxRotation: 0 },
        },
        y: {
          grid: { color: cssToken("--chart-grid", "#EFEDE7") },
          border: { display: false },
          ticks: {
            color: cssToken("--chart-axis", "#8B857A"),
            font: { family: "Inter", size: 11 },
            callback: (v) => (unit ? `${v} ${unit}` : v),
          },
        },
      },
    },
  });
}

/* --------------------------------------------------------------------------
   Sessions
   -------------------------------------------------------------------------- */

function renderSessions() {
  let list = [...App.sessions].reverse(); // newest first
  const days = { all: 0, 30: 30, 7: 7 }[App.sessionRange];
  if (days) {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - days);
    list = list.filter((s) => new Date(s.session_start) >= cutoff);
  }

  const count = `${list.length} session${list.length === 1 ? "" : "s"}`;
  $("sessionsCount").textContent =
    days === 0
      ? `${count} recorded${demoMode() ? " (simulated demo data)" : ""}`
      : `${count} in the last ${days} days`;

  const tbody = $("sessionsBody");
  tbody.innerHTML = "";
  $("sessionsEmpty").hidden = list.length > 0;

  for (const s of list) {
    const status = STATUS_META[s.status] || STATUS_META.recorded;
    // Separate ML indicator — distinct from the status pill and never folded
    // into it. A score is a number, not a diagnosis.
    const ml = s.anomaly || (s.is_anomaly != null ? { is_anomaly: s.is_anomaly, anomaly_score: s.anomaly_score } : null);
    const anomalyBadge =
      ml && ml.is_anomaly === true
        ? ` <span class="pill pill-flag" title="The anomaly model found this session unusual for you${
            isNumber(ml.anomaly_score) ? ` (score ${Number(ml.anomaly_score).toFixed(2)})` : ""
          }. A score is not a diagnosis.">ML flag</span>`
        : "";
    const tr = document.createElement("tr");
    tr.className = "mk-session-row";
    tr.tabIndex = 0;
    tr.setAttribute("role", "button");
    tr.setAttribute("aria-label", `Session on ${fmtFull(s.session_start)} at ${fmtTime(s.session_start)}: open details`);
    tr.addEventListener("click", () => openSessionDialog(s));
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openSessionDialog(s);
      }
    });
    tr.innerHTML = `
      <td class="td-main" data-label="Date">${fmtFull(s.session_start)}<br><span class="opt">${fmtTime(s.session_start)}</span></td>
      <td data-label="Duration">${isNumber(s.duration_s) ? fmtDuration(s.duration_s) : "—"}</td>
      <td class="num" data-label="Speed">${isNumber(s.wpm) ? `${Math.round(s.wpm)} <span class="opt">wpm</span>` : "—"}</td>
      <td class="num" data-label="Dwell">${isNumber(s.dwell_mean_ms) ? `${Math.round(s.dwell_mean_ms)} <span class="opt">ms</span>` : "—"}</td>
      <td class="num" data-label="Flight">${isNumber(s.flight_mean_ms) ? `${Math.round(s.flight_mean_ms)} <span class="opt">ms</span>` : "—"}</td>
      <td class="num" data-label="Corrections">${isNumber(s.correction_rate) ? `${Math.round(s.correction_rate * 100)}%` : "—"}</td>
      <td class="num" data-label="Pauses">${isNumber(s.pause_count) ? s.pause_count : "—"}</td>
      <td data-label="Status"><span class="pill ${status.pill}" title="${status.help}">${status.label}</span>${anomalyBadge}</td>`;
    tbody.appendChild(tr);
  }
}

/* --------------------------------------------------------------------------
   Check-ins
   -------------------------------------------------------------------------- */

function renderCheckins() {
  MK.renderWellbeingContext();
  const wrap = $("checkinOptions");
  wrap.innerHTML = CHECKIN_OPTIONS.map(
    (o) => `
      <button class="checkin-opt" data-factor="${o.id}">
        <span class="opt-dot"></span>${o.label}
      </button>`
  ).join("");

  wrap.querySelectorAll(".checkin-opt").forEach((btn) => {
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", "false");
    btn.addEventListener("click", () => {
      wrap.querySelectorAll(".checkin-opt").forEach((b) => {
        b.classList.remove("selected");
        b.setAttribute("aria-checked", "false");
      });
      btn.classList.add("selected");
      btn.setAttribute("aria-checked", "true");
      App.checkin.selected = btn.dataset.factor;
    });
  });

  renderCheckinHistory();
  $("checkinContext").classList.remove("visible");
}

function renderCheckinHistory() {
  const history = getCheckins().slice().reverse();
  const list = $("checkinHistory");
  if (!history.length) {
    list.innerHTML = `<li class="empty-state" style="padding:10px 0">No check-ins yet.</li>`;
    return;
  }
  list.innerHTML = history
    .map((c) => {
      const opt = CHECKIN_OPTIONS.find((o) => o.id === c.factor);
      const label = opt ? opt.label : c.factor;
      return `<li><span class="hist-factor">${label}</span><span class="hist-date">${fmtFull(c.date)}</span></li>`;
    })
    .join("");
}

async function submitCheckinFlow() {
  const factor = App.checkin.selected;
  if (!factor) {
    showToast("Choose an option first — or skip if you'd rather not say.");
    return;
  }
  const opt = CHECKIN_OPTIONS.find((o) => o.id === factor);
  const note = $("checkinNote").value.trim();
  const now = new Date();
  const localDay = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  try {
    await submitCheckin({
      user_id: USER_ID,
      date: localDay,
      factor,
      label: opt.label,
      note,
    });
  } catch (err) {
    showToast("Your check-in wasn't saved — MindKey couldn't reach the server. Try again in a moment.");
    return;
  }
  // Live mode: the investigation reads check-ins, so re-run it now.
  if (!demoMode()) MK.loadLive().then(() => MK.renderWellbeingContext());

  const ctx = $("checkinContext");
  const state = typingState(App.sessions);
  const contextual = ["tired", "stressed", "poor_sleep", "unwell", "distracted"].includes(factor);

  const noun = CONTEXT_NOUNS[factor] || opt.label.toLowerCase();

  if (state === "warn" || state === "alert") {
    if (contextual) {
      ctx.innerHTML = `Today's variation may have been influenced by reported <strong>${noun}</strong>. It stays in your history — MindKey simply understands it with that context in mind.`;
      ctx.className = "context-note visible";
    } else {
      ctx.innerHTML = `Thanks — noted as <strong>${opt.label.toLowerCase()}</strong>. MindKey will keep observing whether the variation persists over the coming sessions.`;
      ctx.className = "context-note visible";
    }
  } else {
    ctx.innerHTML = "Thanks — your check-in has been saved. It will be considered alongside your typing pattern.";
    ctx.className = "context-note visible";
  }

  App.checkin.selected = null;
  $("checkinNote").value = "";
  wrapClearSelection();
  renderCheckinHistory();
  showToast("Check-in saved.");
}

function wrapClearSelection() {
  $("checkinOptions")
    .querySelectorAll(".checkin-opt")
    .forEach((b) => {
      b.classList.remove("selected");
      b.setAttribute("aria-checked", "false");
    });
}

/* --------------------------------------------------------------------------
   Symptom check
   -------------------------------------------------------------------------- */

function renderSymptoms() {
  const container = $("symptomGroups");
  if (!container.dataset.built) {
    container.innerHTML = SYMPTOM_GROUPS.map(
      (g) => `
        <div class="card symptom-group">
          <h4>${g.group}</h4>
          ${g.items
            .map(
              (item) => `
              <div class="symptom-row">
                <span class="symptom-label">${item.label}</span>
                <div class="freq-seg" data-symptom="${item.id}" role="radiogroup" aria-label="${item.label}">
                  <button class="freq-btn selected" data-freq="none" role="radio" aria-checked="true">Not at all</button>
                  <button class="freq-btn" data-freq="sometimes" role="radio" aria-checked="false">Sometimes</button>
                  <button class="freq-btn" data-freq="often" role="radio" aria-checked="false">Often</button>
                </div>
              </div>`
            )
            .join("")}
        </div>`
    ).join("");
    container.dataset.built = "1";
  }

  container.querySelectorAll(".freq-seg").forEach((seg) => {
    seg.querySelectorAll(".freq-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        seg.querySelectorAll(".freq-btn").forEach((b) => {
          b.classList.remove("selected", "sel-warn", "sel-alert");
          b.setAttribute("aria-checked", "false");
        });
        btn.classList.add("selected");
        btn.setAttribute("aria-checked", "true");
        if (btn.dataset.freq === "sometimes") btn.classList.add("sel-warn");
        if (btn.dataset.freq === "often") btn.classList.add("sel-alert");
      });
    });
  });
}

async function submitSymptomsFlow() {
  const responses = {};
  document.querySelectorAll(".freq-seg").forEach((seg) => {
    const selected = seg.querySelector(".freq-btn.selected");
    responses[seg.dataset.symptom] = selected ? selected.dataset.freq : "none";
  });
  App.symptoms = responses;
  await submitSymptoms({
    user_id: USER_ID,
    date: new Date().toISOString().slice(0, 10),
    responses,
  });
  showToast("Answers saved — thank you.");
  setView("insights");
}

function reportedSymptoms() {
  const stored = getSymptoms();
  if (!stored || !stored.responses) return [];
  const list = [];
  for (const g of SYMPTOM_GROUPS) {
    for (const item of g.items) {
      const freq = stored.responses[item.id];
      if (freq === "sometimes" || freq === "often") {
        list.push(`${item.label} (${freq === "often" ? "often" : "sometimes"})`);
      }
    }
  }
  return list;
}

/* --------------------------------------------------------------------------
   Health insights
   -------------------------------------------------------------------------- */

function renderInsights() {
  // Every statement here comes from the agent's own output (MK.insightModel);
  // symptoms are the one input the agent does not read, and say so.
  const m = MK.insightModel();
  const symptoms = reportedSymptoms();
  const k = m.key;
  const persistent = k === "PERSISTENT_CHANGE";
  const changed = k === "CHANGE_DETECTED";
  const tooEarly = k === "NO_DATA" || k === "BASELINE_FORMING" || k === "INSUFFICIENT_EVIDENCE";

  // Signal: typing pattern (the agent's conclusion)
  $("signalTypingText").textContent = m.state.head + ".";
  $("signalTypingDetail").textContent = m.conclusion ? m.conclusion.statement : m.state.text({});

  // Signal: wellbeing context (what the agent made of the check-ins)
  const lastEvent = m.events.length ? m.events[m.events.length - 1] : null;
  $("signalWellbeingText").textContent = lastEvent
    ? `Latest check-in: ${lastEvent.label.toLowerCase()} (${fmtFull(lastEvent.date)}).`
    : "No check-ins yet.";
  $("signalWellbeingDetail").textContent = m.contextLines.length
    ? m.contextLines.join(" ")
    : "A quick check-in when you notice a change helps MindKey separate everyday causes from unexplained patterns.";

  // Signal: symptoms (not agent input)
  if (symptoms.length) {
    $("signalSymptomsText").textContent = symptoms.slice(0, 2).join("; ") + (symptoms.length > 2 ? ` +${symptoms.length - 2} more` : "") + ".";
    $("signalSymptomsDetail").textContent = "Kept alongside your typing history for your own reference. The AI investigation does not use them, and they are never treated as a diagnosis.";
  } else {
    $("signalSymptomsText").textContent = "No symptoms reported.";
    $("signalSymptomsDetail").textContent = "The symptom check is optional. It is most useful when a persistent change isn't explained by anything you reported.";
  }

  // Interpretation
  let level, pillClass, headline, body, action = "";
  const ctxNames = m.supportedContext.join(" and ");
  if (tooEarly) {
    level = "ok"; pillClass = "pill-ok"; headline = "Too early to say";
    body = m.state.text({ validSessions: null });
  } else if (!persistent && !changed) {
    level = "ok"; pillClass = "pill-ok"; headline = "Your typing pattern appears consistent with your personal baseline";
    body = "Recent sessions stay close to the baseline MindKey learned from your own typing. Keep typing normally; MindKey keeps learning in the background.";
  } else if (changed && m.contextExplains) {
    level = "warn"; pillClass = "pill-warn"; headline = "A recent change with an everyday explanation";
    body = `The change hasn't persisted, and the agent treats ${ctxNames} as a plausible explanation. MindKey will keep watching to confirm things settle.`;
  } else if (changed) {
    level = "warn"; pillClass = "pill-warn"; headline = "Worth watching";
    body = "Some signals moved, but not for long enough to draw a conclusion. A short check-in helps the agent understand the context.";
    action = `<button class="btn btn-ghost btn-sm" data-nav="checkins" style="margin-top:10px">Add a check-in</button>`;
  } else if (m.contextExplains) {
    level = "warn"; pillClass = "pill-warn"; headline = "A persistent change, with reported context";
    body = `The change has lasted across recent sessions. You also reported ${ctxNames}, which the agent treats as a plausible explanation for part of it. Check in again once things settle. If the change continues after that, talking to your usual doctor is a sensible next step.`;
  } else if (symptoms.length) {
    level = "alert"; pillClass = "pill-alert"; headline = "Worth discussing with a professional";
    body = "The change has lasted across recent sessions, nothing you reported explains it, and you noted symptoms. Bringing this investigation to your usual doctor is a sensible next step. MindKey cannot tell what is behind the change.";
  } else {
    level = "warn"; pillClass = "pill-warn"; headline = "A persistent change, not explained by context";
    body = `The change has lasted across recent sessions${m.contextRuledOut ? ", and what you reported (feeling well) makes sleep, stress or fatigue less likely" : ""}. MindKey will keep monitoring. The optional symptom check can round out the picture.`;
    action = `<button class="btn btn-ghost btn-sm" data-nav="symptoms" style="margin-top:10px">Take the optional symptom check</button>`;
  }

  $("interpretationCard").className = "card interpretation interpret-" + level;
  $("interpretationPill").className = "pill " + pillClass;
  $("interpretationPill").textContent = m.state.label;
  $("interpretationTitle").textContent = headline;
  $("interpretationText").innerHTML = `${escapeHtml(body)}${action ? `<br>${action}` : ""}`;

  // Why? — the agent's own reasons
  const why = [];
  if (m.conclusion) why.push(`Agent conclusion: ${m.conclusion.statement}`);
  if (m.depth && m.depth.statement) why.push(m.depth.statement);
  m.contextLines.forEach((line) => why.push(line));
  why.push(
    symptoms.length
      ? `You reported ${symptoms.length} symptom${symptoms.length > 1 ? "s" : ""}: ${symptoms.join("; ")}. These are not agent input.`
      : "No symptoms have been reported."
  );
  why.push("MindKey compares you with your own baseline, never with population averages.");
  $("whyList").innerHTML = why.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
}

/* --------------------------------------------------------------------------
   Session details — one session against the personal baseline
   -------------------------------------------------------------------------- */

const SESSION_SIGNALS = [
  { label: "Typing speed", unit: "wpm", get: (s) => s.wpm, base: (b) => b.wpm, fmt: (v) => Math.round(v) },
  { label: "Dwell time", unit: "ms", get: (s) => s.dwell_mean_ms, base: (b) => (isNumber(b.dwell_mean) ? b.dwell_mean * 1000 : null), fmt: (v) => Math.round(v) },
  { label: "Flight time", unit: "ms", get: (s) => s.flight_mean_ms, base: (b) => (isNumber(b.flight_mean) ? b.flight_mean * 1000 : null), fmt: (v) => Math.round(v) },
  { label: "Correction rate", unit: "%", get: (s) => (isNumber(s.correction_rate) ? s.correction_rate * 100 : null), base: (b) => (isNumber(b.correction_rate) ? b.correction_rate * 100 : null), fmt: (v) => v.toFixed(1) },
  { label: "Rhythm variability", unit: "s", get: (s) => s.rhythm_variability, base: (b) => b.rhythm_variability, fmt: (v) => v.toFixed(2) },
  { label: "Pauses", unit: "", get: (s) => s.pause_count, base: (b) => b.pause_count, fmt: (v) => (Math.round(v * 10) / 10).toString() },
];

function openSessionDialog(s) {
  const dlg = $("sessionDialog");
  const b = App.baseline || {};
  const threshold = (MK.state.payload && MK.state.payload.movedThreshold) || 0.1;
  const invalid = s.status === "invalid" || s.is_valid === false;

  const rows = SESSION_SIGNALS.map((sig) => {
    const v = sig.get(s);
    const base = sig.base(b);
    const rel = isNumber(v) && isNumber(base) && base !== 0 ? (v - base) / base : null;
    const moved = isNumber(rel) && Math.abs(rel) >= threshold;
    const change = isNumber(rel) ? `${rel > 0 ? "+" : rel < 0 ? "−" : ""}${Math.abs(Math.round(rel * 1000) / 10)}%` : "—";
    return `<tr${moved ? ' class="moved"' : ""}>
      <th scope="row">${sig.label}</th>
      <td class="num">${isNumber(v) ? `${sig.fmt(v)} <span class="opt">${sig.unit}</span>` : "—"}</td>
      <td class="num">${isNumber(base) ? `${sig.fmt(base)} <span class="opt">${sig.unit}</span>` : "—"}</td>
      <td class="num">${change}${moved ? ' <span class="mk-moved-tag">beyond ' + Math.round(threshold * 100) + "%</span>" : ""}</td>
    </tr>`;
  }).join("");
  const movedCount = SESSION_SIGNALS.filter((sig) => {
    const v = sig.get(s), base = sig.base(b);
    return isNumber(v) && isNumber(base) && base !== 0 && Math.abs((v - base) / base) >= threshold;
  }).length;

  const ml = s.anomaly;
  const mlLine = invalid
    ? "Not scored: this session's data was incomplete."
    : ml
      ? `Anomaly score ${Number(ml.anomaly_score).toFixed(2)} from the model trained on your earlier sessions (higher means more unusual for you). ${ml.is_anomaly ? "Flagged as unusual for you." : "Within your normal range."}`
      : "Not scored: the model trains once there are at least 10 earlier valid sessions.";

  const day = String(s.session_start || s.date || "").slice(0, 10);
  const dayCheckins = MK.state.payload && MK.state.payload.source === "demo" ? MK.state.payload.checkins : getCheckins();
  const sameDay = (dayCheckins || []).filter((c) => String(c.date).slice(0, 10) === day);

  $("sessionDialogEyebrow").textContent = `Session ${s.session_id}${demoMode() ? " · demo data" : ""}`;
  $("sessionDialogTitle").textContent = `${fmtFull(s.session_start)} · ${fmtTime(s.session_start)}`;
  $("sessionDialogBody").innerHTML = `
    <div class="mk-session-facts">
      <div class="mk-stat"><b>${isNumber(s.duration_s) ? fmtDuration(s.duration_s) : "—"}</b><span>duration</span></div>
      <div class="mk-stat"><b>${invalid ? "Invalid" : "Valid"}</b><span>data quality</span></div>
      <div class="mk-stat"><b>${invalid ? "—" : `${movedCount} of 6`}</b><span>signals beyond ${Math.round(threshold * 100)}%</span></div>
    </div>
    ${invalid ? `<p class="mk-session-note">This session's data was incomplete (for example, no typing speed was recorded), so it is left out of your baseline, the charts and the investigation.</p>` : ""}
    <div class="mk-session-table-wrap">
      <table class="mk-session-table">
        <caption class="sr-only">Signals for this session compared with the personal baseline</caption>
        <thead><tr><th scope="col">Signal</th><th scope="col">This session</th><th scope="col">Your baseline</th><th scope="col">Change</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="mk-session-meta">
      <p><strong>ML:</strong> ${escapeHtml(mlLine)}</p>
      <p><strong>Check-in that day:</strong> ${sameDay.length ? escapeHtml(sameDay.map((c) => (CHECKIN_OPTIONS.find((o) => o.id === c.factor) || { label: c.factor }).label).join(", ")) : "none"}</p>
      <p class="mk-muted">A single session never decides anything. The investigation looks for changes that persist across sessions and signals.</p>
    </div>`;
  if (typeof dlg.showModal === "function") dlg.showModal();
  else dlg.setAttribute("open", "");
  $("sessionDialogClose").focus();
}

function escapeHtml(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderPrivacy() {
  // Pause button state is kept in sync by renderAgent().
  $("privacyPauseBtn").textContent = App.agent.state === "paused" ? "Resume monitoring" : "Pause monitoring";
}

function openDataDialog() {
  const dump = {
    user_id: USER_ID,
    scenario: demoMode() ? DemoState.scenario : "live",
    sessions: App.sessions,
    baseline: App.baseline,
    checkins: getCheckins(),
    symptoms: getSymptoms(),
  };
  $("dataDump").textContent = JSON.stringify(dump, null, 2);
  $("dataDialog").showModal();
}

async function deleteDataFlow() {
  if (!confirm("This permanently removes every stored session, check-in, and symptom response from this device. Continue?")) return;
  await deleteAllData();
  const current = DemoState.scenario;
  setScenario(current);
  App.sessions = DemoState.sessions;
  App.baseline = DemoState.baseline;
  App.symptoms = {};
  agentResetDay();
  renderCheckinHistory();
  showToast("Your check-ins and answers were cleared from this browser.");
}

/* --------------------------------------------------------------------------
   Settings
   -------------------------------------------------------------------------- */

function renderSettings() {
  $("profileName").textContent = "Demo user";
  $("profileId").textContent = USER_ID;
  const sel = $("settingsScenarioSelect");
  if (sel && !sel.options.length) {
    sel.innerHTML = MK.SCENARIOS.map((s) => `<option value="${s.id}">${s.title}</option>`).join("");
  }
  if (sel) sel.value = MK.state.scenarioId || "persistent";
  $("demoCard").hidden = !demoMode();
  MK.scenarioChip();
  renderDemoBadge();
  renderDataSource();
}

/** "Demo data" indicator — visible exactly when the dashboard is simulated. */
function renderDemoBadge() {
  const badge = $("demoBadge");
  if (badge) badge.hidden = !demoMode();
}

/**
 * Honest data-source status. The dashboard runs on simulated data until the
 * backend exposes read endpoints, so the Settings card must say so and must
 * not imply that live sessions are being displayed.
 */
function renderDataSource() {
  const live = !demoMode();
  const demoDot = $("dataSourceDemoDot");
  const apiDot = $("dataSourceApiDot");
  if (!demoDot || !apiDot) return;

  demoDot.className = "dot " + (live ? "dot-muted" : "dot-ok");
  $("dataSourceDemoOpt").textContent = live ? "(inactive)" : "(active)";
  apiDot.className = "dot " + (live ? "dot-ok" : "dot-muted");
  $("dataSourceApiOpt").textContent = live ? `(active · ${API_BASE})` : "(not connected)";
  if ($("connectApi") && live) {
    $("connectApi").value = API_BASE;
    $("connectUser").value = USER_ID;
  }
  if ($("disconnectBtn")) $("disconnectBtn").hidden = !live;
}

/**
 * Connect to a live backend: check /health first (a sleeping free host can
 * take ~50 s to wake), then reload with ?api=&user= so the data layer's
 * normal precedence stores and applies them.
 */
async function connectBackend(e) {
  e.preventDefault();
  const status = $("connectStatus");
  const api = $("connectApi").value.trim().replace(/\/+$/, "");
  const user = $("connectUser").value.trim();
  if (!/^https?:\/\//.test(api)) {
    status.textContent = "Enter the API URL, starting with https://";
    return;
  }
  if (!user) {
    status.textContent = "Enter the user ID whose data you want to see.";
    return;
  }
  $("connectBtn").disabled = true;
  status.textContent = "Checking the backend… a free host can take up to a minute to wake up.";
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 75000);
    const res = await fetch(`${api}/health`, { signal: ctrl.signal });
    clearTimeout(timer);
    if (!res.ok) throw new Error(String(res.status));
    const url = new URL(location.href);
    url.search = `?api=${encodeURIComponent(api)}&user=${encodeURIComponent(user)}`;
    url.hash = "#/home";
    location.assign(url.toString());
  } catch (err) {
    status.textContent =
      "Couldn't reach the backend. Check the URL, and that it allows this site's address (CORS).";
    $("connectBtn").disabled = false;
  }
}

function disconnectBackend() {
  const url = new URL(location.href);
  url.search = "?api=demo";
  url.hash = "#/home";
  location.assign(url.toString());
}

/* --------------------------------------------------------------------------
   Data refresh / scenario switching
   -------------------------------------------------------------------------- */

async function applyScenario(id) {
  await MK.loadScenario(id);
  MK.applyToDashboard();
  App.symptoms = {};
  agentResetDay();
  onScenarioApplied();
  setView("home");
}

/** Called whenever the Demo Lab loads a scenario into the dashboard. */
function onScenarioApplied() {
  App.sessions = DemoState.sessions;
  App.baseline = DemoState.baseline;
  MK.scenarioChip();
  const sel = $("settingsScenarioSelect");
  if (sel && sel.options.length) sel.value = MK.state.scenarioId;
}

async function refreshData({ silent = false } = {}) {
  setLoading(true);
  try {
    const [sessions, baseline, anomalies] = await Promise.all([
      getSessions(),
      getBaseline(),
      getAnomalies(),
    ]);
    // Attach the stored ML anomaly result to each session by id. It is shown
    // as a separate indicator only; it never changes the baseline status.
    App.sessions = (Array.isArray(sessions) ? sessions : []).map((s) => {
      const anomaly = anomalies && anomalies[String(s.session_id)];
      return anomaly
        ? {
            ...s,
            is_anomaly: anomaly.is_anomaly,
            anomaly_score: anomaly.anomaly_score,
            anomaly: { is_anomaly: anomaly.is_anomaly, anomaly_score: anomaly.anomaly_score },
          }
        : s;
    });
    App.baseline = baseline || null;
    if (!demoMode()) await Promise.all([MK.loadLive(), refreshCheckins()]);
    rerenderCurrentView();
    renderDemoBadge();
    if (!silent) showToast(demoMode() ? "Demo data refreshed." : "Data refreshed.");
  } catch (err) {
    // A failed refresh must never leave the dashboard in a broken state.
    showToast("Could not load new data. The current view is unchanged — try again.");
  } finally {
    setLoading(false);
  }
}

/* --------------------------------------------------------------------------
   Wiring
   -------------------------------------------------------------------------- */

/** Keep aria-pressed in sync with the visual .active state of a segmented control. */
function syncSegPressed(segId) {
  const seg = $(segId);
  if (!seg) return;
  seg.querySelectorAll(".seg-btn").forEach((b) => {
    b.setAttribute("aria-pressed", b.classList.contains("active") ? "true" : "false");
  });
}

function wire() {
  // Navigation
  document.querySelectorAll(".nav-item[data-view]").forEach((item) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      setView(item.dataset.view);
    });
  });

  // In-content navigation buttons (CTAs rendered dynamically)
  document.addEventListener("click", (e) => {
    const nav = e.target.closest("[data-nav]");
    if (nav) {
      e.preventDefault();
      setView(nav.dataset.nav);
    }
  });

  // Topbar
  $("themeBtn").addEventListener("click", MK.toggleTheme);
  $("connectForm").addEventListener("submit", connectBackend);
  $("disconnectBtn").addEventListener("click", disconnectBackend);
  window.addEventListener("popstate", () => setView(viewFromHash(), { fromHash: true }));
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-dash-metric]");
    if (b) MK.setDashMetric(b.dataset.dashMetric);
    const run = e.target.closest("[data-run-scenario]");
    if (run && !run.disabled) MK.runFromDashboard(run.dataset.runScenario);
    const jump = e.target.closest("[data-scroll]");
    if (jump) {
      const target = $(jump.dataset.scroll);
      if (target) {
        target.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
        const first = document.querySelector("[data-run-scenario]");
        if (first) first.focus({ preventScroll: true });
      }
    }
  });
  $("refreshBtn").addEventListener("click", refreshData);

  // Trends controls
  document.querySelectorAll("#rangeSeg .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      App.range = Number(btn.dataset.range);
      document.querySelectorAll("#rangeSeg .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      syncSegPressed("rangeSeg");
      renderTrends();
    });
  });
  document.querySelectorAll("#metricSeg .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      App.metric = btn.dataset.metric;
      document.querySelectorAll("#metricSeg .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      syncSegPressed("metricSeg");
      renderTrends();
    });
  });

  // Sessions filter
  document.querySelectorAll("#sessionRangeSeg .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      App.sessionRange = btn.dataset.srange;
      document.querySelectorAll("#sessionRangeSeg .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      syncSegPressed("sessionRangeSeg");
      renderSessions();
    });
  });

  // Segmented controls start with aria-pressed matching their .active button.
  syncSegPressed("rangeSeg");
  syncSegPressed("metricSeg");
  syncSegPressed("sessionRangeSeg");

  // Agent
  $("agentPauseBtn").addEventListener("click", agentPauseToggle);
  $("privacyPauseBtn").addEventListener("click", agentPauseToggle);

  // Check-ins
  $("checkinSubmit").addEventListener("click", submitCheckinFlow);

  // Symptoms
  $("symptomSubmit").addEventListener("click", submitSymptomsFlow);
  $("symptomBack").addEventListener("click", () => setView(App.lastView === "symptoms" ? "home" : App.lastView));

  // Insights
  $("insightTrendsBtn").addEventListener("click", () => setView("trends"));

  // Privacy
  $("viewDataBtn").addEventListener("click", openDataDialog);
  $("dataDialogClose").addEventListener("click", () => $("dataDialog").close());
  $("sessionDialogClose").addEventListener("click", () => $("sessionDialog").close());
  $("sessionDialog").addEventListener("click", (e) => {
    if (e.target === $("sessionDialog")) $("sessionDialog").close();
  });
  $("dataDialog").addEventListener("click", (e) => {
    if (e.target === $("dataDialog")) $("dataDialog").close();
  });
  $("deleteDataBtn").addEventListener("click", deleteDataFlow);
  $("exportDataBtn").addEventListener("click", exportDataFlow);
  $("exclusionsBtn").addEventListener("click", () =>
    showToast("Application exclusions are planned — the desktop agent doesn't support per-app filtering yet.")
  );

  // Settings
  $("settingsScenarioSelect").addEventListener("change", (e) => applyScenario(e.target.value));
  $("resetDemoBtn").addEventListener("click", async () => {
    await deleteAllData();
    await applyScenario("persistent");
    showToast("Demo data reset.");
  });
}

/* --------------------------------------------------------------------------
   Init
   -------------------------------------------------------------------------- */

function viewFromHash() {
  const v = (location.hash || "").replace(/^#\/?/, "");
  return VIEW_META[v] ? v : "home";
}

/** Export: downloads what the dashboard currently holds (demo: local data). */
function exportDataFlow() {
  const payload = {
    exported_at: new Date().toISOString(),
    data_source: demoMode() ? "demo" : "live",
    user_id: USER_ID,
    sessions: App.sessions,
    baseline: App.baseline,
    checkins: getCheckins(),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "mindkey-export.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  showToast("Export downloaded — timing features only, no typed content.");
}

async function init() {
  App.chartOk = typeof Chart !== "undefined";
  wire();
  renderCheckinHistory();

  if (demoMode()) {
    renderSettings();
    MK.state.loading = true;
    setView(viewFromHash(), { fromHash: true });
    await MK.loadScenario(MK.savedScenario());
    MK.applyToDashboard();
    onScenarioApplied();
    agentResetDay();
    renderSettings();
    rerenderCurrentView();
  } else {
    // Live mode. Start from an honest empty state and load real data — never
    // seed simulated rows here, because they would be shown labelled "live".
    App.sessions = [];
    App.baseline = null;
    App.agent.state = "waiting";
    renderSettings();
    setView(viewFromHash(), { fromHash: true });
    await refreshData({ silent: true });
  }
}

init();
