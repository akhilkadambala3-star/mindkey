/*
  MindKey Dashboard — frontend logic.

  This file currently uses a local MOCK_DATA generator that simulates:
    normal -> normal -> normal -> gradual drift -> persistent deviation -> flag
  This mirrors Phase 9 (Demo simulation) since real longitudinal cognitive
  decline cannot be produced during a 24-hour hackathon.

  When the FastAPI backend is ready, replace fetchSessions() / fetchSummary()
  with real fetch() calls to:
    GET /api/users/{user_id}/sessions
    GET /api/users/{user_id}/summary
    GET /api/users/{user_id}/baseline
  The rest of the rendering code does not need to change, as long as the
  API responses match the shapes documented below.
*/

const API_BASE = null; // e.g. "http://localhost:8000" once FastAPI is running. null = use mock data.

// ---------- Mock data generator (Phase 9 demo simulation) ----------

function generateMockSessions(numSessions = 24) {
  const sessions = [];
  const baseline = {
    wpm: 52,
    dwell: 82,
    flight: 121,
    pauses: 4,
    correction: 3,
    rhythm: 0.12,
  };

  // Sessions 0..11: normal, close to baseline with small noise.
  // Sessions 12..17: gradual drift begins.
  // Sessions 18..23: persistent deviation, consistently far from baseline.
  for (let i = 0; i < numSessions; i++) {
    let driftFactor = 0;
    if (i >= 12 && i < 18) {
      driftFactor = (i - 11) / 6; // ramps 0 -> ~1
    } else if (i >= 18) {
      driftFactor = 1;
    }

    const noise = () => (Math.random() - 0.5) * 0.06;

    const wpm = Math.round(baseline.wpm * (1 - driftFactor * 0.18 + noise()));
    const dwell = Math.round(baseline.dwell * (1 + driftFactor * 0.22 + noise()));
    const flight = Math.round(baseline.flight * (1 + driftFactor * 0.26 + noise()));
    const pauses = Math.round(baseline.pauses * (1 + driftFactor * 1.3 + noise()));
    const correction = Math.round(baseline.correction * (1 + driftFactor * 1.6 + noise()) * 10) / 10;
    const rhythm = Math.round(baseline.rhythm * (1 + driftFactor * 1.1 + noise()) * 100) / 100;

    const consistency = Math.max(55, Math.round(96 - driftFactor * 30 + (Math.random() - 0.5) * 4));

    let anomalyScore = Math.max(0, Math.min(1, driftFactor * 0.75 + Math.random() * 0.12));
    anomalyScore = Math.round(anomalyScore * 100) / 100;

    let status = "normal";
    if (driftFactor > 0.35 && driftFactor <= 0.75) status = "elevated";
    if (driftFactor > 0.75) status = "flagged";

    const date = new Date();
    date.setDate(date.getDate() - (numSessions - i));

    sessions.push({
      session_id: `S${(i + 1).toString().padStart(3, "0")}`,
      date: date.toISOString().slice(0, 10),
      wpm,
      dwell_mean_ms: dwell,
      flight_mean_ms: flight,
      pause_count: pauses,
      correction_rate: correction,
      rhythm_variability: rhythm,
      consistency,
      anomaly_score: anomalyScore,
      status,
    });
  }

  return { baseline, sessions };
}

let STATE = generateMockSessions();

// ---------- Data access layer (swap for real API later) ----------

async function fetchSessions(userId) {
  if (API_BASE) {
    const res = await fetch(`${API_BASE}/api/users/${userId}/sessions`);
    return await res.json();
  }
  return STATE.sessions;
}

async function fetchBaseline(userId) {
  if (API_BASE) {
    const res = await fetch(`${API_BASE}/api/users/${userId}/baseline`);
    return await res.json();
  }
  return STATE.baseline;
}

// ---------- Rendering ----------

let trendChart = null;
let currentMetric = "consistency";

function isPersistentlyFlagged(sessions, windowSize = 4) {
  if (sessions.length < windowSize) return false;
  const tail = sessions.slice(-windowSize);
  return tail.every((s) => s.status === "flagged" || s.status === "elevated") &&
         tail.filter((s) => s.status === "flagged").length >= 2;
}

function renderStatus(sessions) {
  const recent = sessions.slice(-6);
  const avgConsistency = Math.round(
    recent.reduce((sum, s) => sum + s.consistency, 0) / recent.length
  );
  const flagged = isPersistentlyFlagged(sessions);
  const elevatedTail = recent.filter((s) => s.status !== "normal").length;

  const badge = document.getElementById("statusBadge");
  const note = document.getElementById("statusNote");

  if (flagged) {
    badge.textContent = "PERSISTENT CHANGE DETECTED";
    badge.className = "status-badge flagged";
    note.textContent =
      "A persistent change in typing behavior has been detected across multiple recent sessions. This is not a medical diagnosis. Consider discussing persistent changes with a qualified professional.";
  } else if (elevatedTail >= 2) {
    badge.textContent = "MONITORING";
    badge.className = "status-badge deviation";
    note.textContent =
      "Some recent sessions differ from your personal baseline. MindKey will continue monitoring to see if this pattern persists.";
  } else {
    badge.textContent = "STABLE";
    badge.className = "status-badge";
    note.textContent =
      "Your recent typing sessions are consistent with your personal baseline.";
  }

  document.getElementById("baselineConsistency").textContent = `${avgConsistency}%`;

  const latestScore = sessions[sessions.length - 1].anomaly_score;
  document.getElementById("anomalyScore").textContent = latestScore.toFixed(2);
  document.getElementById("anomalyScaleFill").style.width = `${Math.round(latestScore * 100)}%`;
}

function renderMetricsCard(baseline, sessions) {
  const latest = sessions[sessions.length - 1];
  document.getElementById("wpmBase").textContent = baseline.wpm;
  document.getElementById("wpmNow").textContent = latest.wpm;
  document.getElementById("dwellBase").textContent = baseline.dwell;
  document.getElementById("dwellNow").textContent = latest.dwell_mean_ms;
  document.getElementById("flightBase").textContent = baseline.flight;
  document.getElementById("flightNow").textContent = latest.flight_mean_ms;
  document.getElementById("pauseBase").textContent = baseline.pauses;
  document.getElementById("pauseNow").textContent = latest.pause_count;
  document.getElementById("correctionBase").textContent = `${baseline.correction}%`;
  document.getElementById("correctionNow").textContent = `${latest.correction_rate}%`;
  document.getElementById("rhythmBase").textContent = baseline.rhythm;
  document.getElementById("rhythmNow").textContent = latest.rhythm_variability;
}

function metricSeries(sessions, metric) {
  switch (metric) {
    case "wpm":
      return sessions.map((s) => s.wpm);
    case "dwell":
      return sessions.map((s) => s.dwell_mean_ms);
    case "flight":
      return sessions.map((s) => s.flight_mean_ms);
    case "pauses":
      return sessions.map((s) => s.pause_count);
    default:
      return sessions.map((s) => s.consistency);
  }
}

function renderChart(sessions) {
  const ctx = document.getElementById("trendChart");
  const labels = sessions.map((s) => s.date.slice(5)); // MM-DD
  const data = metricSeries(sessions, currentMetric);

  const flaggedIndices = sessions
    .map((s, i) => (s.status === "flagged" ? i : null))
    .filter((i) => i !== null);

  const pointColors = sessions.map((s) =>
    s.status === "flagged" ? "#f27272" : s.status === "elevated" ? "#f6b94d" : "#7ee3c4"
  );

  if (trendChart) trendChart.destroy();

  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: currentMetric,
          data,
          borderColor: "#6ea8ff",
          backgroundColor: "rgba(110,168,255,0.08)",
          pointBackgroundColor: pointColors,
          pointBorderColor: pointColors,
          pointRadius: 4,
          tension: 0.35,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: (item) => {
              const s = sessions[item.dataIndex];
              return `Status: ${s.status}`;
            },
          },
        },
      },
      scales: {
        x: { grid: { color: "#2a3348" }, ticks: { color: "#8b93a7" } },
        y: { grid: { color: "#2a3348" }, ticks: { color: "#8b93a7" } },
      },
    },
  });
}

function renderSessionsTable(sessions) {
  const tbody = document.getElementById("sessionsTableBody");
  tbody.innerHTML = "";
  const recent = sessions.slice(-10).reverse();

  for (const s of recent) {
    const tr = document.createElement("tr");
    const badgeClass =
      s.status === "flagged" ? "badge-flagged" : s.status === "elevated" ? "badge-elevated" : "badge-normal";

    tr.innerHTML = `
      <td>${s.session_id}</td>
      <td>${s.date}</td>
      <td>${s.wpm}</td>
      <td>${s.dwell_mean_ms}</td>
      <td>${s.flight_mean_ms}</td>
      <td>${s.pause_count}</td>
      <td>${s.correction_rate}%</td>
      <td class="${badgeClass}">${s.anomaly_score.toFixed(2)}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function renderAll() {
  const userId = document.getElementById("userSelect").value;
  const [sessions, baseline] = await Promise.all([
    fetchSessions(userId),
    fetchBaseline(userId),
  ]);

  renderStatus(sessions);
  renderMetricsCard(baseline, sessions);
  renderChart(sessions);
  renderSessionsTable(sessions);

  document.getElementById("lastUpdated").textContent = new Date().toLocaleTimeString();
}

// ---------- Event wiring ----------

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    currentMetric = chip.dataset.metric;
    renderChart(STATE.sessions);
  });
});

document.getElementById("refreshBtn").addEventListener("click", () => {
  STATE = generateMockSessions();
  renderAll();
});

document.getElementById("userSelect").addEventListener("change", renderAll);

function togglePause(button) {
  const isPaused = button.dataset.paused === "true";
  button.dataset.paused = (!isPaused).toString();
  button.textContent = isPaused ? "Pause monitoring" : "Resume monitoring";
  const dot = document.querySelector(".agent-status .dot");
  dot.className = isPaused ? "dot dot-active" : "dot dot-paused";
  document.querySelector(".agent-status span:last-child").textContent = isPaused
    ? "Agent running"
    : "Agent paused";
}

document.getElementById("pauseAgentBtn").addEventListener("click", (e) => togglePause(e.target));
document.getElementById("pauseAgentBtn2").addEventListener("click", (e) => togglePause(e.target));

document.getElementById("deleteDataBtn").addEventListener("click", () => {
  if (confirm("This will permanently delete all stored behavioral data for this user. Continue?")) {
    STATE = generateMockSessions(0);
    STATE.sessions = [];
    alert("All stored behavioral data has been deleted (demo action — wire this to DELETE /api/users/{id}/data on the backend).");
  }
});

// Initial render
renderAll();
