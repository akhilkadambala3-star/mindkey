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
  home: ["Home", "Your personal behavioral overview"],
  trends: ["Trends", "How your typing pattern has changed over time"],
  sessions: ["Sessions", "A record of your analyzed typing sessions"],
  checkins: ["Check-ins", "Context that helps explain variation"],
  symptoms: ["Symptom check", "A few additional questions"],
  insights: ["Health Insights", "What multiple signals are telling us"],
  care: ["Medical Assistance", "Moving from insight to professional help"],
  privacy: ["Privacy", "What we collect — and what we never see"],
  settings: ["Settings", "Profile, data, and preferences"],
};

const METRICS = {
  speed: {
    label: "Typing speed",
    unit: "wpm",
    get: (s) => s.wpm,
    base: (b) => b.wpm,
  },
  dwell: {
    label: "Dwell time",
    unit: "ms",
    get: (s) => s.dwell_mean_ms,
    base: (b) => Math.round(b.dwell_mean * 1000),
  },
  flight: {
    label: "Flight time",
    unit: "ms",
    get: (s) => s.flight_mean_ms,
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
  care: { step: 1, specialty: null, provider: null, slot: null },
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

function setView(name) {
  if (!VIEW_META[name]) name = "home";
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

  const renderers = {
    home: renderHome,
    trends: renderTrends,
    sessions: renderSessions,
    checkins: renderCheckins,
    symptoms: renderSymptoms,
    insights: renderInsights,
    care: renderCare,
    privacy: renderPrivacy,
    settings: renderSettings,
  };
  renderers[name]();
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
  $("todaySessions").textContent = a.sessionCountToday;
  $("todaySessionsLimit").textContent = AGENT_DAILY_SESSION_LIMIT;
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
  $("greeting").textContent = greeting();
  const state = typingState(App.sessions);
  const panel = $("statusPanel");
  const icon = $("statusIcon");
  const headline = $("statusHeadline");
  const text = $("statusText");
  const observed = $("statusObserved");
  const actions = $("statusActions");

  panel.className = "status-panel state-" + state;

  if (state === "ok") {
    icon.innerHTML = '<svg viewBox="0 0 24 24"><path d="M4.5 13 8 9.5l3 3 8.5-8.5"/></svg>';
    $("statusEyebrow").textContent = "Behavioral status";
    headline.textContent = "Your typing pattern is consistent";
    text.textContent = "No meaningful change has been detected compared with your recent baseline.";
    observed.hidden = true;
    actions.innerHTML = `
      <button class="btn btn-primary" data-nav="trends">View your trends</button>
      <button class="btn btn-ghost" data-nav="checkins">Complete a check-in</button>`;
  } else if (state === "warn") {
    icon.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 4 3.5 19.5h17L12 4Z"/><path d="M12 10v4.5M12 17.2v.1"/></svg>';
    $("statusEyebrow").textContent = "Behavioral status";
    headline.textContent = "Some recent variation was detected";
    text.textContent =
      "A few recent sessions differ from your usual pattern. MindKey will keep observing before drawing any conclusions — everyday factors like tiredness or stress can cause short-term variation.";
    observed.hidden = true;
    actions.innerHTML = `
      <button class="btn btn-primary" data-nav="checkins">Complete a short check-in</button>
      <button class="btn btn-ghost" data-nav="trends">View trends</button>`;
  } else {
    icon.innerHTML = '<svg viewBox="0 0 24 24"><path d="M4.5 13 8 9.5l3 3 8.5-8.5"/></svg>';
    $("statusEyebrow").textContent = "Persistent change detected";
    headline.textContent = "We've noticed a persistent change";
    text.textContent =
      "Your typing pattern has remained different from your usual baseline across multiple sessions. Persistent changes can have many possible explanations, including temporary factors — we'd like to understand whether anything else may explain it.";
    observed.hidden = false;
    $("statusObservedList").innerHTML = observedChanges()
      .map((c) => `<li>${c}</li>`)
      .join("");
    actions.innerHTML = `
      <button class="btn btn-primary" data-nav="symptoms">Continue to symptom check</button>
      <button class="btn btn-ghost" data-nav="insights">View health insight</button>`;
  }

  // Today snapshot — tolerant of an empty day, a missing baseline, or a
  // payload that omits a feature. None of these may break the dashboard.
  const today = App.sessions.filter((s) => isTodayISO(s.session_start));
  const consistencyValues = today.map((s) => s.consistency).filter(isNumber);
  const speedValues = today.map((s) => s.wpm).filter(isNumber);
  const dwellValues = today.map((s) => s.dwell_mean_ms).filter(isNumber);
  const speed = speedValues.length ? Math.round(mean(speedValues)) : null;
  const dwell = dwellValues.length ? Math.round(mean(dwellValues)) : null;

  const baseWpm = baselineValue("wpm");
  const baseDwellSeconds = baselineValue("dwell_mean");
  const baseDwell = baseDwellSeconds == null ? null : Math.round(baseDwellSeconds * 1000);

  $("todayConsistency").textContent = consistencyValues.length
    ? Math.round(mean(consistencyValues))
    : "–";
  $("todaySpeed").textContent = speed ?? "–";
  $("todayDwell").textContent = dwell ?? "–";
  $("todaySpeedSub").textContent =
    speed == null
      ? "no sessions today yet"
      : baseWpm == null
        ? "no baseline yet"
        : vsBaseline(speed, baseWpm, "wpm");
  $("todayDwellSub").textContent =
    dwell == null
      ? "no sessions today yet"
      : baseDwell == null
        ? "no baseline yet"
        : vsBaseline(dwell, baseDwell, "ms");

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
    `${sessions.length} sessions in the last ${App.range} days.`,
    `Latest ${metric.label.toLowerCase()}: ${latest}${unit}.`,
  ];
  if (isNumber(base)) parts.push(`Your baseline: ${base}${unit}.`);
  el.textContent = parts.join(" ");
}

function renderTrends() {
  const metric = METRICS[App.metric];
  $("chartTitle").textContent = metric.label;
  $("chartSub").textContent = `Last ${App.range} days · ${UNIT_LABELS[metric.unit] || metric.label.toLowerCase()}`;
  $("whatChangedRange").textContent = `Last ${App.range} days`;

  // What changed? — data-driven, independent of charts.
  const changes = describeChanges(App.sessions, App.range);
  const list = $("changeList");
  if (!changes.length) {
    list.innerHTML = `<li class="change-empty">No meaningful change detected in this period — your pattern is holding close to your personal baseline.</li>`;
  } else {
    list.innerHTML = changes.map((c) => `<li class="change-warn">${c.text}</li>`).join("");
  }

  // Chart. Every exit path tears the previous chart down first.
  const chartWindow = App.sessions.slice(-App.range);
  const fallback = $("chartFallback");

  if (!App.chartOk) {
    destroyChart();
    fallback.textContent =
      "Charts need internet access to load Chart.js. Everything else works offline.";
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
    color: "#2F6B4F",
  });
  setChartSummary(chartWindow, metric, base);
}

function buildChart({ sessions, labels, data, baseline, unit, color }) {
  destroyChart();
  const ctx = $("trendChart");
  ctx.setAttribute("role", "img");
  ctx.setAttribute(
    "aria-label",
    `${METRICS[App.metric].label} over the last ${App.range} days, compared with your personal baseline.`
  );

  const datasets = [
    {
      label: "Sessions",
      data,
      borderColor: color,
      backgroundColor: "rgba(47, 107, 79, 0.05)",
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
      borderColor: "#B9B4A9",
      borderWidth: 1.5,
      borderDash: [5, 5],
      pointRadius: 0,
      fill: false,
    });
  }

  return new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: prefersReducedMotion() ? false : undefined,
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#FFFFFF",
          titleColor: "#26231D",
          bodyColor: "#5C584F",
          borderColor: "#E6E3DC",
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
              return `${item.dataset.label === "Your baseline" ? "Your baseline" : "Sessions"}: ${item.parsed.y}${suffix}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { color: "#E6E3DC" },
          ticks: { color: "#8B857A", font: { family: "Inter", size: 11 }, maxTicksLimit: 9, maxRotation: 0 },
        },
        y: {
          grid: { color: "#EFEDE7" },
          border: { display: false },
          ticks: {
            color: "#8B857A",
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
    const status = STATUS_META[s.status] || STATUS_META.normal;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="td-main" data-label="Date">${fmtFull(s.session_start)}<br><span class="opt">${fmtTime(s.session_start)}</span></td>
      <td data-label="Duration">${isNumber(s.duration_s) ? fmtDuration(s.duration_s) : "—"}</td>
      <td class="num" data-label="Speed">${isNumber(s.wpm) ? `${s.wpm} <span class="opt">wpm</span>` : "—"}</td>
      <td class="num" data-label="Dwell">${isNumber(s.dwell_mean_ms) ? `${s.dwell_mean_ms} <span class="opt">ms</span>` : "—"}</td>
      <td class="num" data-label="Flight">${isNumber(s.flight_mean_ms) ? `${s.flight_mean_ms} <span class="opt">ms</span>` : "—"}</td>
      <td class="num" data-label="Corrections">${isNumber(s.correction_rate) ? `${Math.round(s.correction_rate * 100)}%` : "—"}</td>
      <td class="num" data-label="Pauses">${isNumber(s.pause_count) ? s.pause_count : "—"}</td>
      <td data-label="Status"><span class="pill ${status.pill}" title="${status.help}">${status.label}</span></td>`;
    tbody.appendChild(tr);
  }
}

/* --------------------------------------------------------------------------
   Check-ins
   -------------------------------------------------------------------------- */

function renderCheckins() {
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
  await submitCheckin({
    user_id: USER_ID,
    date: new Date().toISOString().slice(0, 10),
    factor,
    label: opt.label,
    note,
  });

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
  const state = typingState(App.sessions);
  const checkins = getCheckins();
  const latest = checkins.length ? checkins[checkins.length - 1] : null;
  const contextual = latest && ["tired", "stressed", "poor_sleep", "unwell", "distracted"].includes(latest.factor);
  const symptoms = reportedSymptoms();

  // Signal: typing
  const typingTexts = {
    ok: "Your recent sessions are consistent with your personal baseline.",
    warn: "Several recent sessions have varied from your personal baseline.",
    alert: "Several typing metrics have shifted from your personal baseline and stayed shifted.",
  };
  const typingDetail = {
    ok: "No persistent change is present. MindKey compares you with your own baseline, learned from your own typing.",
    warn: "The variation is recent and has not yet persisted long enough to be conclusive.",
    alert: "The change has persisted across multiple sessions, which is why MindKey is paying closer attention.",
  };
  $("signalTypingText").textContent = typingTexts[state];
  $("signalTypingDetail").textContent = typingDetail[state];

  // Signal: wellbeing
  if (latest) {
    $("signalWellbeingText").textContent = `You reported "${latest.label}"${latest.note ? ` — "${latest.note}"` : ""}.`;
    $("signalWellbeingDetail").textContent = `Check-in on ${fmtFull(latest.date)}.${contextual ? " This kind of factor can explain short-term typing variation." : ""}`;
  } else {
    $("signalWellbeingText").textContent = "No check-ins yet.";
    $("signalWellbeingDetail").textContent = "Completing a check-in when you notice variation helps MindKey separate everyday causes from unexplained patterns.";
  }

  // Signal: symptoms
  if (symptoms.length) {
    $("signalSymptomsText").textContent = symptoms.slice(0, 2).join("; ") + (symptoms.length > 2 ? ` +${symptoms.length - 2} more` : "") + ".";
    $("signalSymptomsDetail").textContent = "Reported in the symptom questionnaire — kept alongside your typing history, never treated as a diagnosis.";
  } else {
    $("signalSymptomsText").textContent = "No symptoms reported.";
    $("signalSymptomsDetail").textContent = "The symptom questionnaire is optional and only asked when a persistent change remains unexplained.";
  }

  // Interpretation
  const interp = $("interpretationCard");
  const pill = $("interpretationPill");
  const title = $("interpretationTitle");
  const text = $("interpretationText");

  let level, pillClass, headline, body;
  if (state === "ok") {
    level = "ok";
    pillClass = "pill-ok";
    headline = "Your pattern looks consistent";
    body = "Your recent typing behavior remains close to your personal baseline, and no additional signals suggest anything to follow up on. Keep typing normally — MindKey continues learning quietly in the background.";
  } else if (state === "warn") {
    if (contextual) {
      level = "warn";
      pillClass = "pill-warn";
      headline = "Variation with a likely everyday explanation";
      body = `Recent variation overlaps with what you told us — ${CONTEXT_NOUNS[latest.factor] || latest.label.toLowerCase()}. MindKey treats this as contextualized: it stays in your history, and it is understood with that context in mind. We'll keep observing to confirm things settle back.`;
    } else {
      level = "warn";
      pillClass = "pill-warn";
      headline = "Worth monitoring";
      body = "A few recent sessions differ from your usual pattern, but the change has not persisted long enough to draw conclusions. If you can, complete a short check-in — it helps MindKey understand the context.";
    }
  } else {
    if (symptoms.length) {
      level = "alert";
      pillClass = "pill-alert";
      headline = "Professional evaluation recommended";
      body = "Several signals have persisted: your typing pattern has stayed different from your baseline, and you've reported symptoms you don't consider normal. Because multiple signals line up over time, MindKey recommends discussing these changes with a healthcare professional.";
    } else {
      level = "warn";
      pillClass = "pill-warn";
      headline = "Further check-in recommended";
      body = "Your typing pattern has remained different from your usual baseline, but nothing you've shared yet explains it. A few more days of observation, a short check-in, and — if the change persists — a conversation with a professional are the sensible next steps.";
    }
  }

  interp.className = "card interpretation interpret-" + level;
  pill.className = "pill " + pillClass;
  pill.textContent = headline === "Your pattern looks consistent" ? "No notable concern" : headline;
  title.textContent = headline;
  text.textContent = body;

  // Why?
  const why = [];
  why.push({
    ok: "Your recent sessions remain consistent with the baseline MindKey learned from your own typing.",
    warn: "Your recent sessions have varied from your personal baseline — enough to notice, not enough to conclude anything yet.",
    alert: "Your typing pattern has shifted from your personal baseline and stayed shifted across several sessions.",
  }[state]);

  if (latest) {
    why.push(
      contextual
        ? `You reported ${CONTEXT_NOUNS[latest.factor] || latest.label.toLowerCase()} on ${fmtFull(latest.date)} — an everyday explanation that can account for some of the variation.`
        : `Your most recent check-in (${latest.label.toLowerCase()}, ${fmtFull(latest.date)}) provides context for this period.`
    );
  } else {
    why.push("You haven't completed a check-in recently, so there's no wellbeing context for this period.");
  }

  if (symptoms.length) {
    why.push(`You reported ${symptoms.length} symptom${symptoms.length > 1 ? "s" : ""} in the questionnaire: ${symptoms.join("; ")}.`);
  } else {
    why.push("No symptoms have been reported, so MindKey has no symptom signal to add.");
  }

  why.push("MindKey compares you with your own baseline, not with population averages — everyone types differently.");
  if (state === "alert") {
    why.push("MindKey only recommends professional evaluation after the change persists across multiple sessions and across multiple signals, never from a single session or a single metric.");
  }

  $("whyList").innerHTML = why.map((w) => `<li>${w}</li>`).join("");
}

/* --------------------------------------------------------------------------
   Medical assistance
   -------------------------------------------------------------------------- */

const CARE_SPECIALTIES = [
  { id: "neurology", name: "Neurology", desc: "Doctors who specialise in the brain and nervous system." },
  { id: "general", name: "General Medicine", desc: "A family doctor or general practitioner who knows your overall health." },
  { id: "psychiatry", name: "Psychiatry", desc: "Medical doctors focused on mental and emotional health." },
  { id: "psychology", name: "Psychology", desc: "Professionals supporting emotional, cognitive, and behavioural wellbeing." },
];

const PROVIDER_POOL = {
  neurology: ["Dr. A. Reyes", "Dr. M. Okafor", "Dr. L. Tran"],
  general: ["Dr. S. Novak", "Dr. P. Andersson", "Dr. K. Patel"],
  psychiatry: ["Dr. J. Meyer", "Dr. R. Haddad", "Dr. C. Lindqvist"],
  psychology: ["Dr. N. Bergström", "Dr. T. Yoshida", "Dr. E. Marchetti"],
};

function mockProviders(specialtyId) {
  return (PROVIDER_POOL[specialtyId] || []).map((name, i) => ({
    id: `${specialtyId}-${i}`,
    name,
    specialty: CARE_SPECIALTIES.find((s) => s.id === specialtyId).name,
    note: `Example provider · ${i === 0 ? "usually available within days" : "usually available within 1–2 weeks"}`,
  }));
}

function mockSlots() {
  const slots = [];
  const now = new Date();
  const times = ["09:00", "11:30", "14:00", "16:30"];
  for (let d = 1; d <= 5; d++) {
    const day = new Date(now);
    day.setDate(day.getDate() + d);
    const weekday = day.toLocaleDateString("en-US", { weekday: "short" });
    const dateLabel = day.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    for (const t of times) {
      // Every ~5th slot is unavailable, so the demo shows a realistic mix.
      const taken = (d * times.length + times.indexOf(t)) % 5 === 0;
      slots.push({ day: dateLabel, weekday, time: t, taken });
    }
  }
  return slots;
}

function renderCare() {
  const steps = $("careSteps");
  steps.querySelectorAll(".care-step").forEach((el, i) => {
    el.className = "care-step" + (i + 1 < App.care.step ? " done" : "") + (i + 1 === App.care.step ? " active" : "");
  });

  const content = $("careContent");
  const c = App.care;

  if (c.step === 1) {
    content.innerHTML = `
      <h3>Which kind of professional would you like to see?</h3>
      <div class="specialty-grid">
        ${CARE_SPECIALTIES.map(
          (s) => `
            <button class="specialty-card ${c.specialty === s.id ? "selected" : ""}" data-specialty="${s.id}">
              <h4>${s.name}</h4>
              <p>${s.desc}</p>
              <span class="specialty-tag">${s.id === "neurology" ? "Relevant to typing changes" : "General option"}</span>
            </button>`
        ).join("")}
      </div>
      <p class="card-note">MindKey never chooses for you — this is about making it easy to start a conversation. Your usual doctor is always a good first step.</p>`;

    content.querySelectorAll(".specialty-card").forEach((card) => {
      card.addEventListener("click", () => {
        App.care.specialty = card.dataset.specialty;
        App.care.provider = null;
        App.care.slot = null;
        App.care.step = 2;
        renderCare();
      });
    });
  } else if (c.step === 2) {
    const providers = mockProviders(c.specialty);
    content.innerHTML = `
      <h3>Choose a provider <span class="opt">— ${CARE_SPECIALTIES.find((s) => s.id === c.specialty).name}</span></h3>
      ${providers
        .map(
          (p) => `
            <div class="provider-row">
              <div>
                <p class="provider-name">${p.name}<span class="provider-tag">Example</span></p>
                <p class="provider-meta">${p.note}</p>
              </div>
              <button class="btn btn-ghost btn-sm" data-provider="${p.id}">${c.provider === p.id ? "Selected" : "Choose"}</button>
            </div>`
        )
        .join("")}
      <div class="care-actions">
        <button class="btn btn-ghost" id="careBackBtn">Back</button>
      </div>`;

    content.querySelectorAll("[data-provider]").forEach((btn) => {
      btn.addEventListener("click", () => {
        App.care.provider = btn.dataset.provider;
        App.care.slot = null;
        App.care.step = 3;
        renderCare();
      });
    });
    $("careBackBtn").addEventListener("click", () => {
      App.care.step = 1;
      renderCare();
    });
  } else if (c.step === 3) {
    const slots = mockSlots();
    const provider = mockProviders(c.specialty).find((p) => p.id === c.provider);
    content.innerHTML = `
      <h3>Pick a time <span class="opt">— ${provider.name}</span></h3>
      <div class="slot-grid">
        ${slots
          .map(
            (s, i) => `
              <button class="slot-btn ${c.slot === i ? "selected" : ""}" data-slot="${i}" ${s.taken ? "disabled" : ""}>
                <span class="slot-date">${s.weekday} ${s.day}</span>${s.time}
              </button>`
          )
          .join("")}
      </div>
      <p class="card-note">All times shown are simulated for the prototype.</p>
      <div class="care-actions">
        <button class="btn btn-ghost" id="careBackBtn">Back</button>
      </div>`;

    content.querySelectorAll("[data-slot]").forEach((btn) => {
      if (btn.disabled) return;
      btn.addEventListener("click", () => {
        App.care.slot = Number(btn.dataset.slot);
        App.care.step = 4;
        renderCare();
      });
    });
    $("careBackBtn").addEventListener("click", () => {
      App.care.step = 2;
      renderCare();
    });
  } else {
    const provider = mockProviders(c.specialty).find((p) => p.id === c.provider);
    const slot = mockSlots()[c.slot];
    const booked = store.get(KEYS.appointment, null);
    content.innerHTML = `
      <h3>Confirm your appointment</h3>
      <div class="card care-summary">
        <div class="care-summary-row"><strong>Provider</strong><span>${provider.name} — ${provider.specialty}</span></div>
        <div class="care-summary-row"><strong>When</strong><span>${slot.weekday}, ${slot.day} at ${slot.time}</span></div>
        <div class="care-summary-row"><strong>How</strong><span>In-person or video call — confirmed by the clinic</span></div>
      </div>
      ${
        booked
          ? `<div class="care-confirm-note">Appointment saved locally for this demo (${booked.date}). A real booking system would send this to the clinic and confirm by email.</div>`
          : `<div class="care-actions">
              <button class="btn btn-primary" id="careConfirmBtn">Confirm appointment</button>
              <button class="btn btn-ghost" id="careBackBtn">Back</button>
            </div>`
      }`;

    $("careBackBtn")?.addEventListener("click", () => {
      App.care.step = 3;
      renderCare();
    });
    $("careConfirmBtn")?.addEventListener("click", () => {
      store.set(KEYS.appointment, {
        user_id: USER_ID,
        specialty: provider.specialty,
        provider: provider.name,
        slot: `${slot.weekday}, ${slot.day} at ${slot.time}`,
        date: new Date().toISOString(),
      });
      renderCare();
      showToast("Appointment noted (demo). A real integration would confirm with the clinic.");
    });
  }
}

/* --------------------------------------------------------------------------
   Privacy
   -------------------------------------------------------------------------- */

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
  showToast("Demo data cleared. (Production: wire this to DELETE /api/users/{id}/data.)");
}

/* --------------------------------------------------------------------------
   Settings
   -------------------------------------------------------------------------- */

function renderSettings() {
  $("profileName").textContent = "Demo user";
  $("profileId").textContent = USER_ID;
  $("settingsScenarioSelect").value = DemoState.scenario;
  $("scenarioSelect").value = DemoState.scenario;
  $("demoCard").hidden = !demoMode();
  $("demoSelectWrap").style.display = demoMode() ? "inline-flex" : "none";
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
  $("dataSourceApiOpt").textContent = live ? "(active)" : "(not connected)";
}

/* --------------------------------------------------------------------------
   Data refresh / scenario switching
   -------------------------------------------------------------------------- */

function applyScenario(key) {
  setScenario(key);
  App.sessions = DemoState.sessions;
  App.baseline = DemoState.baseline;
  App.symptoms = {};
  agentResetDay();
  setView("home");
}

async function refreshData() {
  setLoading(true);
  try {
    const [sessions, baseline] = await Promise.all([getSessions(), getBaseline()]);
    App.sessions = Array.isArray(sessions) ? sessions : [];
    App.baseline = baseline || null;
    const renderers = {
      home: renderHome,
      trends: renderTrends,
      sessions: renderSessions,
      checkins: renderCheckins,
      symptoms: renderSymptoms,
      insights: renderInsights,
      care: renderCare,
      privacy: renderPrivacy,
      settings: renderSettings,
    };
    renderers[App.view]();
    renderDemoBadge();
    showToast(demoMode() ? "Demo data refreshed." : "Data refreshed.");
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
  $("scenarioSelect").addEventListener("change", (e) => applyScenario(e.target.value));
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
  $("insightCareBtn").addEventListener("click", () => setView("care"));
  $("insightTrendsBtn").addEventListener("click", () => setView("trends"));

  // Privacy
  $("viewDataBtn").addEventListener("click", openDataDialog);
  $("dataDialogClose").addEventListener("click", () => $("dataDialog").close());
  $("dataDialog").addEventListener("click", (e) => {
    if (e.target === $("dataDialog")) $("dataDialog").close();
  });
  $("deleteDataBtn").addEventListener("click", deleteDataFlow);
  $("exclusionsBtn").addEventListener("click", () =>
    showToast("Application exclusions are planned — the desktop agent doesn't support per-app filtering yet.")
  );

  // Settings
  $("settingsScenarioSelect").addEventListener("change", (e) => applyScenario(e.target.value));
  $("regenerateBtn").addEventListener("click", () => {
    setScenario(DemoState.scenario);
    App.sessions = DemoState.sessions;
    App.baseline = DemoState.baseline;
    agentResetDay();
    renderSettings();
    showToast("Demo data regenerated.");
  });
  $("resetDemoBtn").addEventListener("click", async () => {
    await deleteAllData();
    applyScenario("variation");
    showToast("Demo data reset.");
  });
}

/* --------------------------------------------------------------------------
   Init
   -------------------------------------------------------------------------- */

async function init() {
  App.chartOk = typeof Chart !== "undefined";
  wire();
  renderCheckinHistory();

  if (demoMode()) {
    setScenario("variation");
    App.sessions = DemoState.sessions;
    App.baseline = DemoState.baseline;
    agentResetDay();
    renderSettings();
    setView("home");
  } else {
    // Live mode. Start from an honest empty state and load real data — never
    // seed simulated rows here, because they would be shown labelled "live".
    App.sessions = [];
    App.baseline = null;
    App.agent.state = "waiting";
    renderSettings();
    setView("home");
    await refreshData();
  }
}

init();