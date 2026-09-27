/* ==========================================================================
   MindKey v2 — investigation layer, Demo Lab and dashboard views.

   Honesty contract (see docs/SPEC.md §29):
   - Everything shown about the investigation comes from the real MindKey
     investigation engine (backend/investigation/agent). In demo mode it is
     read from dashboard/demo/<key>.json, written by
     backend/scripts/export_demo.py (a test keeps it identical to the engine).
     With ?api= set it is fetched live from the backend.
   - Demo sessions are synthetic and always labelled "Demo data".
   - Hypotheses and alternatives are shown as the engine's qualitative states.
     No probabilities, no risk score.
   - The timeline "replay" paces recorded trace events for readability; the
     events and their order are exactly the engine's.
   ========================================================================== */

const MK = (() => {
  const esc = (v) =>
    String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduced = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const el = (id) => document.getElementById(id);
  const num = (v) => typeof v === "number" && Number.isFinite(v);

  /* ------------------------------------------------------------------------
     Scenario catalog — spec names mapped onto the engine's demo datasets.
     Each scenario's check-ins come with its engine output and are real agent
     input: the investigation reads the ones in its recent 7-day window.
     ------------------------------------------------------------------------ */
  const SCENARIOS = [
    { id: "stable", key: "consistent", title: "Stable baseline",
      blurb: "Thirty days on the personal baseline. The user checks in feeling well." },
    { id: "fatigue", key: "recent_variation", title: "Temporary fatigue",
      blurb: "Typing speed dips for a few days while the user reports poor sleep and tiredness." },
    { id: "persistent", key: "persistent_change", title: "Persistent change",
      blurb: "Six signals shift and stay shifted. The user reports feeling well, so context is ruled out." },
    { id: "sudden", key: "sudden_change", title: "Sudden change",
      blurb: "A sharp shift in only the two latest sessions, on a day the user was distracted." },
    { id: "recovery", key: "recovery", title: "Recovery",
      blurb: "A week of change while unwell, now back on the baseline." },
    { id: "noisy", key: "invalid_data", title: "Noisy data",
      blurb: "A dense history with one captured session that fails validation." },
    { id: "insufficient", key: "insufficient_history", title: "Insufficient data",
      blurb: "Only three sessions — below the model's minimum of ten." },
  ];
  const byId = (id) => SCENARIOS.find((s) => s.id === id) || SCENARIOS[2];

  /* ------------------------------------------------------------------------
     Data states (spec §24) and behavioral change level (spec §11)
     ------------------------------------------------------------------------ */
  const STATES = {
    NO_DATA: { label: "No data", tone: "neutral", level: null,
      head: "Start monitoring to build your baseline",
      text: () => "Start monitoring to begin building your personal baseline. MindKey needs a few sessions before it can compare you with yourself." },
    BASELINE_FORMING: { label: "Baseline forming", tone: "info", level: null,
      head: "Building your personal baseline",
      text: (p) => `MindKey needs more sessions before it can establish a reliable personal baseline${p && p.validSessions != null ? ` — ${p.validSessions} of ${p.minimum || 10} so far` : ""}.` },
    STABLE: { label: "Stable", tone: "stable", level: "Minimal",
      head: "Your behavioral baseline is stable",
      text: () => "Your recent interaction patterns remain within your personal baseline." },
    MONITORING: { label: "Monitoring", tone: "stable", level: "Minimal",
      head: "Stable — with a data-quality note",
      text: (p) => `Recent patterns match your baseline. ${p && p.invalid ? `${p.invalid} captured session${p.invalid === 1 ? "" : "s"} failed validation and ${p.invalid === 1 ? "was" : "were"} excluded from the analysis.` : "MindKey keeps monitoring."}` },
    CHANGE_DETECTED: { label: "Change detected", tone: "change", level: "Moderate",
      head: "A recent change was detected",
      text: () => "Some signals moved away from your baseline, but the change has not persisted long enough to draw a conclusion. MindKey will keep monitoring." },
    INVESTIGATING: { label: "Investigating", tone: "info", level: null, live: true,
      head: "AI investigation in progress",
      text: () => "The agent is checking persistence, data quality and alternative explanations." },
    PERSISTENT_CHANGE: { label: "Persistent change", tone: "persist", level: "Persistent",
      head: "A persistent behavioral change was detected",
      text: () => "Several signals have stayed away from your personal baseline across recent sessions. The investigation shows what was checked — behavioral data alone cannot determine a cause." },
    INSUFFICIENT_EVIDENCE: { label: "Insufficient evidence", tone: "neutral", level: null,
      head: "Evidence is currently insufficient",
      text: () => "The available evidence is not enough to reach a conclusion. MindKey will continue monitoring." },
  };

  const state = {
    scenarioId: null,
    payload: null,       // normalized investigation payload
    source: null,        // demo | live
    error: null,         // null | "agent_unavailable" | "ml_unavailable"
    loading: false,
    replaying: false,
    dashMetric: "speed",
    charts: {},
  };

  function dataState(p = state.payload) {
    if (state.replaying) return "INVESTIGATING";
    if (!p || !p.sessionCount) return "NO_DATA";
    const c = p.conclusion || {};
    if (c.status === "inconclusive") return c.basis === "insufficient_history" ? "BASELINE_FORMING" : "INSUFFICIENT_EVIDENCE";
    if (c.status === "no_deviation") return p.invalidSessions > 0 ? "MONITORING" : "STABLE";
    if (c.status === "preliminary") return "CHANGE_DETECTED";
    if (c.status === "grounded") return "PERSISTENT_CHANGE";
    return "INSUFFICIENT_EVIDENCE";
  }
  function stateCtx(p = state.payload) {
    const dq = (p && p.dataQuality) || {};
    return { validSessions: dq.valid_sessions, minimum: p && p.minimum, invalid: p ? p.invalidSessions : 0 };
  }

  /* ------------------------------------------------------------------------
     Loading + normalization
     ------------------------------------------------------------------------ */
  async function fetchJSON(url) {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error(String(res.status));
    return res.json();
  }

  function normalize(raw, source) {
    const report = raw.report || null;
    // Live payloads nest the engine read model under ``engine``; the demo
    // export carries the same fields at the top level.
    const eng = raw.engine || raw;
    const evidence = Array.isArray(eng.evidence) ? eng.evidence : [];
    const sessions = Array.isArray(raw.sessions) ? raw.sessions : [];
    const fa = eng.final_assessment || (report && report.engine_assessment) || {};
    const e8 = evidence.find((e) => e.kind === "session_count");
    return {
      raw, source,
      key: raw.key || null,
      title: raw.title || null,
      sessions,
      checkins: Array.isArray(raw.checkins) ? raw.checkins : [],
      // The engine counts a signal as "moved" at this relative change
      // (investigation/adapter.py MOVED_RELATIVE_THRESHOLD).
      movedThreshold: typeof eng.moved_threshold === "number" ? eng.moved_threshold : 0.1,
      sessionCount: sessions.length || (fa.data_quality && fa.data_quality.total_sessions) || (report ? 1 : 0),
      invalidSessions: sessions.filter((s) => s.is_valid === false).length ||
        Math.max(0, ((fa.data_quality || {}).total_sessions || 0) - ((fa.data_quality || {}).valid_sessions || 0)),
      report,
      conclusion: report ? report.conclusion : { status: "inconclusive", basis: "data_unavailable", statement: "No stored sessions to investigate yet." },
      evidence,
      evidenceById: Object.fromEntries(evidence.map((e) => [e.id, e])),
      trace: Array.isArray(eng.trace) ? eng.trace : [],
      timelineText: Array.isArray(raw.timeline) ? raw.timeline : [],
      finalAssessment: fa,
      dataQuality: fa.data_quality || {},
      persistence: fa.persistence || {},
      mlEvidence: eng.ml || null,
      toolsCalled: eng.tools_called || [],
      iterations: eng.iterations ?? fa.iterations ?? null,
      stopReason: raw.stop_reason || fa.stop_reason || null,
      minimum: e8 ? e8.threshold : 10,
      stopSeq: report && report.link ? report.link.stop_event_seq : null,
      traceCount: report && report.link ? report.link.trace_event_count : (eng.trace || []).length,
      critic: report && report.critic_events && report.critic_events[0] ? report.critic_events[0] : null,
    };
  }

  async function loadScenario(id) {
    const sc = byId(id);
    state.loading = true;
    state.error = null;
    try {
      const url = typeof API_BASE === "string" && API_BASE
        ? `${API_BASE}/api/demo/scenarios/${sc.key}`
        : `demo/${sc.key}.json`;
      let raw;
      try {
        raw = await fetchJSON(url);
      } catch (e) {
        // A live backend without the demo routes falls back to the static export.
        raw = await fetchJSON(`demo/${sc.key}.json`);
      }
      state.scenarioId = sc.id;
      state.payload = normalize(raw, "demo");
      state.source = "demo";
      try { localStorage.setItem("mindkey.labScenario", sc.id); } catch (e) { /* storage optional */ }
      return state.payload;
    } catch (e) {
      state.error = "agent_unavailable";
      state.payload = null;
      return null;
    } finally {
      state.loading = false;
    }
  }

  async function loadLive() {
    state.loading = true;
    state.error = null;
    try {
      const raw = await fetchJSON(`${API_BASE}/api/users/${encodeURIComponent(USER_ID)}/investigation`);
      state.payload = normalize({ ...raw, sessions: (typeof App !== "undefined" ? App.sessions : []) }, "live");
      state.source = "live";
    } catch (e) {
      state.error = "agent_unavailable";
      state.payload = null;
    } finally {
      state.loading = false;
    }
  }

  function savedScenario() {
    try { return localStorage.getItem("mindkey.labScenario") || "persistent"; } catch (e) { return "persistent"; }
  }

  /* Hand the scenario's sessions to the existing data layer, so Trends,
     Sessions and Insights show exactly the data the agent investigated. */
  function applyToDashboard() {
    const p = state.payload;
    if (!p || p.source !== "demo" || typeof DemoState === "undefined") return;
    const valid = p.sessions.filter((s) => s.is_valid !== false);
    const stat = (key) => {
      const e = p.evidence.find((x) => x.kind === "window_stat" && x.window_days === 30 && x.signal === key);
      return e && num(e.value) ? e.value : null;
    };
    const speed = stat("speed");
    DemoState.sessions = p.sessions.map((s) => ({
      ...s,
      duration_s: 20,
      status: s.is_valid === false ? "invalid" : s.anomaly && s.anomaly.is_anomaly ? "ml_flag" : undefined,
    }));
    DemoState.baseline = {
      typing_speed: speed,
      wpm: num(speed) ? Math.round(speed / 5) : null,
      dwell_mean: stat("dwell"),
      flight_mean: stat("flight"),
      correction_rate: stat("corrections"),
      rhythm_variability: stat("rhythm"),
      pause_count: stat("pauses"),
      sample_count: valid.length,
      updated_at: p.raw.as_of || null,
    };
    DemoState.scenario = state.scenarioId;
    if (typeof App !== "undefined") {
      App.sessions = DemoState.sessions;
      App.baseline = DemoState.baseline;
    }
  }

  /* ------------------------------------------------------------------------
     Vocabulary
     ------------------------------------------------------------------------ */
  const TOOL_NAMES = {
    build_behavioral_evidence: "Built behavioral evidence package",
    get_recent_sessions: "Retrieved recent sessions",
    calculate_behavioral_drift: "Calculated behavioral drift across time windows",
    get_ml_evidence: "Retrieved ML anomaly result",
    get_historical_baseline: "Retrieved 30-day personal baseline",
    compare_time_windows: "Compared time windows",
  };
  const STEP_NAMES = {
    recent_sessions: "recent sessions",
    window_robustness: "window robustness",
    ml_evidence: "ML anomaly result",
    baseline: "personal baseline",
  };
  const STOP_NAMES = {
    evidence_sufficient: "Evidence sufficient to conclude",
    evidence_insufficient: "Evidence insufficient to conclude",
    no_further_evidence: "No further evidence available",
    only_unanswerable_questions_remain: "Only unanswerable questions remain",
    max_iterations: "Iteration limit reached",
  };
  const REASONS = {
    persistence_not_established: "Persistence has not been established",
    window_robustness_not_assessed: "Window robustness was not assessed",
    window_robustness_disagrees: "Comparison windows disagree",
    "unanswered:context_factors": "No recent check-ins on sleep, stress or fatigue",
    "unanswered:capture_change": "Keyboard or device changes cannot be checked",
    cause_not_established: "Behavioral data alone cannot establish a cause",
    insufficient_history: "Not enough history for a reliable baseline",
  };
  const HYP_NOTES = {
    H1: "Short-lived variation that returns to baseline",
    H2: "Sleep, fatigue, stress or workload",
    H3: "A sustained shift from the personal baseline",
    H4: "Capture or data problems",
  };
  const STATUS_WORD = { supported: "Supported", uncertain: "Uncertain", weakened: "Weakened", unavailable: "Unavailable", possible: "Possible", partially_evaluated: "Partially evaluated" };
  const human = (s) => String(s || "").replace(/_/g, " ");

  /* Signals — display units */
  const SIGNALS = {
    speed: { label: "Typing speed", unit: "wpm", fmt: (v) => Math.round(v / 5), metric: "speed" },
    dwell: { label: "Dwell time", unit: "ms", fmt: (v) => Math.round(v * 1000), metric: "dwell" },
    flight: { label: "Flight time", unit: "ms", fmt: (v) => Math.round(v * 1000), metric: "flight" },
    corrections: { label: "Correction rate", unit: "%", fmt: (v) => (Math.round(v * 1000) / 10).toFixed(1), metric: "corrections" },
    rhythm: { label: "Rhythm variability", unit: "s", fmt: (v) => v.toFixed(2), metric: "rhythm" },
    pauses: { label: "Pauses", unit: "/ session", fmt: (v) => Math.round(v * 10) / 10, metric: "pauses" },
  };
  const pct = (r) => (num(r) ? `${r > 0 ? "+" : r < 0 ? "−" : ""}${Math.abs(Math.round(r * 1000) / 10)}%` : "—");
  const arrow = (r) => (!num(r) || r === 0 ? "→" : r > 0 ? "↑" : "↓");

  function signalRows(p = state.payload) {
    if (!p) return [];
    return p.evidence
      .filter((e) => e.kind === "signal_change" && SIGNALS[e.signal])
      .map((e) => {
        const stat7 = p.evidence.find((x) => x.kind === "window_stat" && x.window_days === 7 && x.signal === e.signal);
        return { ...e, sig: SIGNALS[e.signal], recent7: stat7 ? stat7.value : null, recent7Id: stat7 ? stat7.id : null };
      });
  }

  /* ------------------------------------------------------------------------
     Trace → readable investigation steps
     ------------------------------------------------------------------------ */
  function traceSteps(p = state.payload) {
    if (!p) return [];
    if (!p.trace.length) {
      // Live endpoint: only rendered text lines are available.
      return p.timelineText.map((line, i) => {
        const m = String(line).match(/^(\S+)\s+[—-]\s+(.*)$/);
        return { seq: i + 1, ts: m ? m[1] : "", kind: "node", title: m ? m[2] : line, detail: "", raw: "", evidence: [] };
      });
    }
    const out = [];
    let hypGroup = null;
    let altGroup = null;
    for (const e of p.trace) {
      const d = e.detail || "";
      const base = { seq: e.seq, ts: e.ts, raw: `${e.node} · ${e.event_type}${e.tool ? ` · ${e.tool}` : ""}${d ? ` · ${d}` : ""}`, evidence: e.evidence_ids || [] };
      if (e.event_type === "node_enter" && e.node === "ingest_signal") {
        out.push({ ...base, kind: "node", title: "Signal received", detail: `Triggering session ${p.finalAssessment.trigger_session_id || ""}` });
      } else if (e.event_type === "tool_call") {
        out.push({ ...base, kind: "tool", title: TOOL_NAMES[e.tool] || human(e.tool), detail: `Tool call · ${e.tool}` });
      } else if (e.event_type === "tool_result") {
        const last = out[out.length - 1];
        if (last && last.kind === "tool") {
          last.detail = `${e.tool} → ${d || "result"}`;
          last.evidence = [...last.evidence, ...(e.evidence_ids || [])];
          last.raw += `\n${base.raw}`;
        }
      } else if (e.event_type === "decision") {
        let title = human(d);
        let kind = "decision";
        if (d.startsWith("plan_required_evidence:")) {
          title = "Planned evidence to collect";
          base.detailText = d.split(":")[1].split("+").map((s) => STEP_NAMES[s] || human(s)).join(", ");
        } else if (d.startsWith("collect:")) {
          const step = (d.match(/step=(\S+)/) || [])[1];
          title = `Next step: collect ${STEP_NAMES[step] || human(step)}`;
        } else if (d === "awaiting_required_evidence") {
          title = "Persistence check waiting for more evidence";
        } else if (d.startsWith("persistence_assessed")) {
          const v = d.split(" ")[1] || "";
          const [cls, rob] = v.split("/");
          title = `Persistence assessed: ${human(cls)}${rob && rob !== "not_assessed" ? ` · windows ${human(rob)}` : ""}`;
        } else if (d === "no_collectible_evidence") {
          title = "No further evidence can be collected";
        }
        out.push({ ...base, kind, title, detail: base.detailText || "" });
      } else if (e.event_type === "evidence_added") {
        out.push({ ...base, kind: "decision", title: "Registered persistence evidence", detail: human(d) });
      } else if (e.event_type === "hypothesis_updated") {
        if (!hypGroup) {
          hypGroup = { ...base, kind: "hyp", title: "Evaluated behavioral hypotheses", detail: "", items: [] };
          out.push(hypGroup);
        }
        const [id, st] = d.split(":");
        hypGroup.items.push({ id, st });
        hypGroup.evidence = [...new Set([...hypGroup.evidence, ...(e.evidence_ids || [])])];
        hypGroup.raw += `\n${base.raw}`;
      } else if (e.event_type === "state_change" && e.node === "assess_alternatives") {
        if (!altGroup) {
          altGroup = { ...base, kind: "alt", title: "Checked alternative explanations", detail: "", items: [] };
          out.push(altGroup);
        }
        const [id, st] = d.split(":");
        altGroup.items.push({ id, st });
      } else if (e.event_type === "stop") {
        out.push({ ...base, kind: "stop", title: STOP_NAMES[d] || `Stopped: ${human(d)}`, detail: "Stop condition met" });
      }
    }
    if (p.critic) {
      const c = p.critic;
      const m = (c.detail || "").match(/is_sound=(\w+) rejections=(\d+) contradictions=(\d+)/);
      out.push({
        seq: c.seq, ts: c.ts, kind: "critic", evidence: [], raw: `critic · decision · ${c.detail}`,
        title: m && m[1] === "True" ? "Critic verified the report" : "Critic reviewed the report",
        detail: m ? `${m[2]} rejected claims · ${m[3]} contradictions — every claim is linked to evidence` : c.detail,
      });
    }
    out.push({ seq: (p.traceCount || 0) + 1, ts: "", kind: "stop", title: "Investigation complete", detail: STOP_NAMES[p.stopReason] || human(p.stopReason), raw: "", evidence: [] });
    return out;
  }

  /* ------------------------------------------------------------------------
     Components
     ------------------------------------------------------------------------ */
  const CHECK = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 6.2 5 8.5l4.5-5"/></svg>';

  function StatusBadge(key) {
    const s = STATES[key];
    return `<span class="mk-status tone-${s.tone}${s.live ? " is-live" : ""}"><span class="dot" aria-hidden="true"></span>${esc(s.label)}</span>`;
  }

  function SourceChip(p = state.payload) {
    if (!p) return "";
    return p.source === "live"
      ? `<span class="mk-chip mk-chip-real"><span class="dot"></span>Live data · engine output</span>`
      : `<span class="mk-chip mk-chip-demo" title="Synthetic sessions. The investigation itself is the real MindKey engine run on them."><span class="dot"></span>Demo data · real engine output</span>`;
  }

  function MetricCard(row) {
    const s = row.sig;
    return `<div class="mk-metric">
      <p class="mk-metric-label">${esc(s.label)}</p>
      <p class="mk-metric-value">${num(row.value) ? esc(s.fmt(row.value)) : "—"}<small>${esc(s.unit)}</small></p>
      <p class="mk-metric-delta${!num(row.relative_change) || row.relative_change === 0 ? " flat" : ""}">${arrow(row.relative_change)} ${pct(row.relative_change)}</p>
      <p class="mk-metric-base">Baseline ${num(row.baseline_value) ? esc(s.fmt(row.baseline_value)) : "—"} ${esc(s.unit)} · <span class="mk-evid">${esc(row.id)}</span></p>
    </div>`;
  }

  function HypothesisCard(h) {
    const st = h.status || "unavailable";
    return `<div class="mk-hyp-row">
      <div class="name">${esc(h.statement)}<small>${esc(HYP_NOTES[h.id] || "")}${h.evidence_ids && h.evidence_ids.length ? ` · ${h.evidence_ids.map((i) => `<span class="mk-evid">${esc(i)}</span>`).join(" ")}` : ""}</small></div>
      <div class="mk-meter st-${esc(st)}" role="img" aria-label="${esc(STATUS_WORD[st] || st)}"><i></i><i></i><i></i><i></i></div>
      <div class="mk-hyp-status st-text-${esc(st)}">${esc(STATUS_WORD[st] || human(st))}</div>
    </div>`;
  }

  const FACTOR_LABELS = {
    feeling_well: "Feeling well", tired: "Tired", stressed: "Stressed", poor_sleep: "Poor sleep",
    unwell: "Feeling unwell", distracted: "Distracted / busy", other: "Something else",
  };
  const NEUTRAL_FACTORS = new Set(["feeling_well", "other"]);

  /** Check-ins for the loaded data: the scenario's (demo) or the user's (live). */
  function contextEvents(p = state.payload) {
    if (!p) return [];
    const rows = p.source === "demo"
      ? p.checkins
      : (typeof getCheckins === "function" ? getCheckins() : []);
    return (rows || [])
      .filter((c) => c && c.date && c.factor)
      .map((c) => ({ date: String(c.date).slice(0, 10), factor: c.factor, label: FACTOR_LABELS[c.factor] || c.label || c.factor, neutral: NEUTRAL_FACTORS.has(c.factor) }))
      .sort((a, b) => a.date.localeCompare(b.date));
  }

  function ContextList(events) {
    if (!events.length) return `<p class="mk-muted" style="font-size:13px">No check-ins yet.</p>`;
    return `<ul class="mk-context">${events
      .map((c) => `<li><span class="when">${esc(fmtShort(c.date))}</span><span class="pip ${c.neutral ? "neutral" : ""}" aria-hidden="true"></span><span>${esc(c.label)}</span></li>`)
      .join("")}</ul>`;
  }

  function fmtShort(iso) {
    const d = new Date(iso.length === 10 ? iso + "T12:00:00" : iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }

  function ErrorState(kind) {
    const map = {
      agent_unavailable: ["AI investigation could not be completed", "View the underlying behavioral evidence on the Trends and Sessions pages.", "trends", "View trends"],
      ml_unavailable: ["Behavioral analysis is temporarily unavailable", "Your collected data remains safe.", "sessions", "View sessions"],
      NO_DATA: ["No sessions yet", "Start monitoring to begin building your personal baseline.", "lab", "Explore the Demo Lab"],
    };
    const [h, t, nav, cta] = map[kind] || map.agent_unavailable;
    return `<div class="mk-empty" role="status"><h3>${esc(h)}</h3><p>${esc(t)}</p><button class="btn btn-ghost" data-nav="${nav}">${esc(cta)}</button></div>`;
  }

  function LoadingState() {
    return `<div class="mk-empty" role="status" aria-live="polite"><h3>Loading investigation…</h3>
      <div class="mk-skeleton-lines"><span>Retrieving personal baseline…</span><span>Reading engine trace…</span></div></div>`;
  }

  /* ------------------------------------------------------------------------
     Why was this flagged? — built only from engine evidence
     ------------------------------------------------------------------------ */
  function whyFlagged(p = state.payload) {
    if (!p) return { reasons: [], context: [] };
    const reasons = [];
    const rows = signalRows(p).filter((r) => num(r.relative_change) && Math.abs(r.relative_change) >= p.movedThreshold)
      .sort((a, b) => Math.abs(b.relative_change) - Math.abs(a.relative_change));
    for (const r of rows.slice(0, 4)) {
      reasons.push({ text: `${r.sig.label} ${r.relative_change < 0 ? "decreased" : "increased"} ${Math.abs(Math.round(r.relative_change * 1000) / 10)}% vs. your 30-day baseline`, ids: [r.id] });
    }
    const depth = p.evidence.find((e) => e.kind === "persistence_depth");
    if (depth && num(depth.value) && depth.value > 0) {
      reasons.push({ text: `The pattern persisted across ${depth.value} consecutive recent session${depth.value === 1 ? "" : "s"}${depth.onset ? ` (since ${fmtShort(depth.onset)})` : ""}`, ids: [depth.id] });
    }
    const anomaly = p.evidence.find((e) => e.kind === "anomaly" && e.status === "true");
    if (anomaly) reasons.push({ text: `The per-user Isolation Forest flagged the triggering session as unusual`, ids: [anomaly.id] });
    if (!reasons.length && p.report) {
      (p.report.summary || []).slice(0, 3).forEach((s) => reasons.push({ text: s.replace(/\s*\[[^\]]+\]$/, ""), ids: (s.match(/E\d+/g) || []) }));
    }
    return { reasons, context: contextFindings(p) };
  }

  /* What the agent concluded about context, in plain words, from its own
     evidence items and alternative findings. */
  function contextFindings(p = state.payload) {
    if (!p) return [];
    const out = [];
    const r = p.report;
    const alts = r ? r.alternatives : [];
    const ctxIds = new Set(["poor_sleep", "fatigue", "stress", "illness_or_mood", "distraction"]);
    const ctxAlts = alts.filter((a) => ctxIds.has(a.id));
    const supported = ctxAlts.filter((a) => a.status === "supported").map((a) => a.statement.toLowerCase());
    const weakened = ctxAlts.filter((a) => a.status === "weakened");
    const factors = p.evidence.filter((e) => e.kind === "context_factor");
    const stable = p.conclusion && p.conclusion.status === "no_deviation";

    if (factors.length) {
      const parts = factors.map((e) => `${e.label} on ${e.session_count} day${e.session_count === 1 ? "" : "s"}`);
      out.push(`In the last ${factors[0].window_days} days you reported ${parts.join(", ")}.`);
    }
    if (supported.length) {
      out.push(`The agent treats ${supported.join(" and ")} as a plausible explanation for part of the change.`);
    } else if (weakened.length && factors.length) {
      out.push("Because you reported feeling well, the agent weakened sleep, stress and fatigue as explanations.");
    } else if (stable && factors.length) {
      out.push("There is no measured change for this context to explain.");
    }
    if (p.evidence.some((e) => e.kind === "context_not_reported")) {
      out.push("No check-ins in the last 7 days, so the agent cannot confirm or rule out sleep, stress or fatigue. A quick check-in helps.");
    }
    if (p.evidence.some((e) => e.kind === "context_absence")) {
      out.push("No check-in store is connected, so the agent cannot confirm or rule out sleep, stress or fatigue.");
    }
    out.push("Behavioral data alone cannot determine the cause of a change.");
    return out;
  }

  /* ------------------------------------------------------------------------
     Charts (Chart.js). One series + dashed baseline + event markers.
     ------------------------------------------------------------------------ */
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  const markerPlugin = {
    id: "mkMarkers",
    afterDatasetsDraw(chart, args, opts) {
      const markers = (opts && opts.items) || [];
      if (!markers.length) return;
      const { ctx, chartArea, scales } = chart;
      ctx.save();
      for (const m of markers) {
        const x = scales.x.getPixelForValue(m.index);
        if (!Number.isFinite(x)) continue;
        if (m.type === "change") {
          ctx.strokeStyle = cssVar("--chart-change");
          ctx.lineWidth = 1.5;
          ctx.setLineDash([]);
          ctx.beginPath(); ctx.moveTo(x, chartArea.top + 14); ctx.lineTo(x, chartArea.bottom); ctx.stroke();
          ctx.fillStyle = cssVar("--chart-change");
          ctx.font = "600 11px Inter, sans-serif";
          ctx.textAlign = x > chartArea.right - 80 ? "right" : "left";
          ctx.fillText(m.label, x + (ctx.textAlign === "left" ? 5 : -5), chartArea.top + 10);
        } else if (m.type === "context") {
          ctx.fillStyle = cssVar("--chart-context");
          ctx.globalAlpha = 0.9;
          ctx.fillRect(x - 4, chartArea.bottom - 9, 8, 8);
          ctx.globalAlpha = 1;
        }
      }
      ctx.restore();
    },
  };

  function markersFor(sessions) {
    const p = state.payload;
    const items = [];
    if (!p || !sessions.length) return items;
    const dayIdx = (iso) => {
      const day = String(iso).slice(0, 10);
      let idx = sessions.findIndex((s) => String(s.session_start || s.date).slice(0, 10) >= day);
      return idx;
    };
    const onset = (p.persistence && p.persistence.onset) || (p.evidence.find((e) => e.kind === "persistence_depth" && e.onset) || {}).onset;
    if (onset && p.conclusion && p.conclusion.status !== "no_deviation") {
      const i = dayIdx(onset);
      if (i > 0) items.push({ type: "change", index: i, label: p.conclusion.status === "grounded" ? "Change point · persistent" : "Change point" });
    }
    for (const c of contextEvents(p)) {
      const i = sessions.findIndex((s) => String(s.session_start || s.date).slice(0, 10) >= c.date);
      if (i >= 0) items.push({ type: "context", index: i, label: c.label });
    }
    return items;
  }

  function lineChart(canvas, { sessions, get, base, unit, label, markers }) {
    if (typeof Chart === "undefined" || !canvas) return null;
    const labels = sessions.map((s) => fmtShort(s.session_start || s.date));
    const data = sessions.map(get);
    const ctxEvents = {};
    for (const m of (markers || []).filter((x) => x.type === "context")) {
      ctxEvents[m.index] = ctxEvents[m.index] ? `${ctxEvents[m.index]}, ${m.label}` : m.label;
    }
    const flagged = sessions.map((s) => !!(s.anomaly && s.anomaly.is_anomaly));
    const datasets = [{
      label, data,
      borderColor: cssVar("--chart-1"), backgroundColor: cssVar("--chart-1-fill"),
      borderWidth: 2, tension: 0.25, fill: true,
      pointRadius: flagged.map((f) => (f ? 5 : 0)),
      pointBorderWidth: flagged.map((f) => (f ? 2 : 0)),
      pointBackgroundColor: cssVar("--surface"),
      pointBorderColor: cssVar("--chart-change"),
      pointHoverRadius: 5, pointHoverBackgroundColor: cssVar("--chart-1"),
    }];
    if (num(base)) {
      datasets.push({ label: "Personal baseline", data: sessions.map(() => base), borderColor: cssVar("--chart-baseline"), borderWidth: 1.5, borderDash: [5, 5], pointRadius: 0, fill: false });
    }
    return new Chart(canvas, {
      type: "line",
      data: { labels, datasets },
      plugins: [markerPlugin],
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: reduced() ? false : { duration: 500 },
        interaction: { intersect: false, mode: "index" },
        plugins: {
          legend: { display: false },
          mkMarkers: { items: markers || [] },
          tooltip: {
            backgroundColor: cssVar("--surface"), titleColor: cssVar("--text"), bodyColor: cssVar("--text-2"),
            borderColor: cssVar("--border"), borderWidth: 1, padding: 10, cornerRadius: 8, displayColors: false,
            callbacks: {
              title: (items) => {
                const s = sessions[items[0].dataIndex];
                return s ? `${fmtShort(s.session_start || s.date)} · session ${s.session_id}` : "";
              },
              label: (item) => `${item.dataset.label}: ${item.parsed.y}${unit ? " " + unit : ""}`,
              afterBody: (items) => {
                const i = items[0].dataIndex;
                const lines = [];
                if (flagged[i]) lines.push("ML: flagged by Isolation Forest");
                if (ctxEvents[i]) lines.push(`Check-in: ${ctxEvents[i]}`);
                const cp = (markers || []).find((m) => m.type === "change" && m.index === i);
                if (cp) lines.push(`${cp.label}`);
                return lines;
              },
            },
          },
        },
        scales: {
          x: { grid: { display: false }, border: { color: cssVar("--border") }, ticks: { color: cssVar("--chart-axis"), font: { family: "Inter", size: 11 }, maxTicksLimit: 8, maxRotation: 0 } },
          y: { grid: { color: cssVar("--chart-grid") }, border: { display: false }, ticks: { color: cssVar("--chart-axis"), font: { family: "Inter", size: 11 }, maxTicksLimit: 5 } },
        },
      },
    });
  }

  function destroy(name) {
    if (state.charts[name]) { state.charts[name].destroy(); state.charts[name] = null; }
  }

  /* ------------------------------------------------------------------------
     DASHBOARD
     ------------------------------------------------------------------------ */
  const DASH_METRICS = {
    speed: { label: "Speed", get: (s) => s.wpm, base: (b) => b && b.wpm, unit: "wpm", title: "Typing speed" },
    corrections: { label: "Corrections", get: (s) => Math.round(s.correction_rate * 1000) / 10, base: (b) => b && num(b.correction_rate) ? Math.round(b.correction_rate * 1000) / 10 : null, unit: "%", title: "Correction rate" },
    pauses: { label: "Pauses", get: (s) => s.pause_count, base: (b) => b && b.pause_count, unit: "", title: "Pauses per session" },
    rhythm: { label: "Rhythm", get: (s) => s.rhythm_variability, base: (b) => b && num(b.rhythm_variability) ? Math.round(b.rhythm_variability * 100) / 100 : null, unit: "s", title: "Rhythm variability" },
  };

  function renderDashboard() {
    const root = el("mkDashboard");
    if (!root) return;
    const p = state.payload;
    const key = dataState();
    const S = STATES[key];
    const hero = `
      <section class="mk-hero" aria-labelledby="heroTitle">
        <div>
          <p class="mk-eyebrow">Longitudinal behavioral change monitoring</p>
          <h2 id="heroTitle">Understand changes in your everyday behavior.</h2>
          <p>MindKey learns your personal behavioral baseline and uses ML + agentic investigation to identify persistent changes — without collecting what you type.</p>
          <div class="mk-hero-actions">
            <button class="btn btn-primary" data-nav="lab">Explore demo</button>
            <button class="btn btn-ghost" data-nav="investigation">View AI investigation</button>
          </div>
        </div>
        <div class="mk-hero-side">
          <div class="mk-pipeline" aria-label="How MindKey works">
            <span class="step on">Baseline</span><span class="arrow">→</span>
            <span class="step on">Change</span><span class="arrow">→</span>
            <span class="step on">ML evidence</span><span class="arrow">→</span>
            <span class="step on">Agent</span><span class="arrow">→</span>
            <span class="step on">Insight</span>
          </div>
          ${SourceChip()}
        </div>
      </section>`;

    if (state.loading) { root.innerHTML = hero + `<div class="mk-section">${LoadingState()}</div>`; return; }
    if (!p) { root.innerHTML = hero + `<div class="mk-section">${ErrorState(state.error || "NO_DATA")}</div>`; return; }

    const levels = ["Minimal", "Moderate", "Persistent"];
    const status = `
      <section class="mk-state tone-${S.tone}" aria-live="polite">
        <div>
          ${StatusBadge(key)}
          <h2>${esc(S.head)}</h2>
          <p>${esc(S.text(stateCtx()))}</p>
        </div>
        <div class="mk-level">
          <p class="mk-eyebrow">Behavioral change</p>
          <div class="mk-level-scale">${levels.map((l) => `<span class="${S.level === l ? "on" : ""}">${l}</span>`).join("")}</div>
          ${S.level ? "" : `<p class="mk-muted" style="font-size:12px;margin-top:6px">Not enough evidence to grade</p>`}
        </div>
      </section>`;

    const rows = signalRows(p);
    const pick = ["speed", "corrections", "pauses", "rhythm"];
    const cards = pick.map((k) => rows.find((r) => r.signal === k)).filter(Boolean);
    const overview = cards.length ? `
      <section class="mk-section" aria-labelledby="ovTitle">
        <div class="mk-section-head"><h3 id="ovTitle">Behavioral overview</h3><p class="section-note">Latest session vs. your 30-day personal baseline</p></div>
        <div class="mk-metrics">${cards.map(MetricCard).join("")}</div>
      </section>` : "";

    const m = DASH_METRICS[state.dashMetric];
    const trend = `
      <div class="mk-card">
        <div class="mk-chart-head">
          <div><h3 class="mk-card-title" style="margin:0">Behavioral trend</h3>
          <p class="chart-sub">${esc(m.title)} · personal baseline · change points · check-ins</p></div>
          <div class="mk-seg-mini" role="group" aria-label="Trend metric">${Object.entries(DASH_METRICS)
            .map(([k, v]) => `<button data-dash-metric="${k}" aria-pressed="${k === state.dashMetric}">${v.label}</button>`).join("")}</div>
        </div>
        <div class="mk-chart-wrap"><canvas id="mkDashChart" role="img" aria-label="${esc(m.title)} over time compared with the personal baseline"></canvas></div>
        <div class="mk-legend" aria-hidden="true">
          <span><i class="l-line"></i>Your sessions</span><span><i class="l-base"></i>Personal baseline</span>
          <span><i class="l-change"></i>Change point</span><span><i class="l-context"></i>Check-in</span><span><i class="l-flag"></i>ML-flagged session</span>
        </div>
        <p class="sr-only" id="mkDashChartSummary"></p>
      </div>`;

    const invCard = `
      <div class="mk-card mk-inv-card tone-${S.tone}">
        <div class="card-head" style="margin-bottom:8px"><h3 class="mk-card-title" style="margin:0">AI investigation</h3>${StatusBadge(key)}</div>
        <p class="mk-quote">${esc(p.conclusion.statement)}</p>
        <div class="mk-inv-stats">
          <div class="mk-stat"><b>${p.toolsCalled.length + (p.trace.length ? 1 : 0)}</b><span>tool calls</span></div>
          <div class="mk-stat"><b>${p.evidence.length || (p.report ? p.report.observations.length : 0)}</b><span>evidence items</span></div>
          <div class="mk-stat"><b>${p.report ? p.report.rejected_claims.length : "—"}</b><span>unsupported claims</span></div>
        </div>
        <button class="btn btn-primary btn-sm" data-nav="investigation">View investigation</button>
      </div>`;

    const why = whyFlagged(p);
    const changed = `
      <div class="mk-card">
        <h3 class="mk-card-title">What changed?</h3>
        ${why.reasons.length && key !== "STABLE" && key !== "MONITORING"
          ? `<ol class="mk-list">${why.reasons.map((r, i) => `<li><span class="ix">${i + 1}</span><span class="body">${esc(r.text)} ${r.ids.map((x) => `<span class="mk-evid">${esc(x)}</span>`).join(" ")}</span></li>`).join("")}</ol>`
          : `<p class="mk-muted" style="font-size:14px">No meaningful change from your baseline. ${p.invalidSessions ? `${p.invalidSessions} session failed validation and was excluded.` : ""}</p>`}
      </div>`;

    const events = contextEvents(p);
    const agentView = contextFindings(p).slice(0, -1);
    const context = `
      <div class="mk-card">
        <div class="card-head" style="margin-bottom:8px"><h3 class="mk-card-title" style="margin:0">Context</h3>${p.source === "demo" ? `<span class="mk-chip mk-chip-demo">Scenario check-ins</span>` : ""}</div>
        ${ContextList(events.slice(-5))}
        ${agentView.length ? `<p class="card-note" style="margin-top:10px"><strong>Agent:</strong> ${esc(agentView.join(" "))}</p>` : ""}
        <p class="card-note" style="margin-top:6px">Check-ins from the last 7 days are read by the AI investigation — the day and the factor only. Notes stay private.</p>
        <button class="btn btn-ghost btn-sm" data-nav="checkins" style="margin-top:10px">Add a check-in</button>
      </div>`;

    root.innerHTML = `${hero}${status}${overview}
      <div class="mk-section mk-grid-main">
        <div class="mk-col">${trend}${changed}</div>
        <div class="mk-col">${invCard}${context}</div>
      </div>`;

    drawDashChart();
  }

  function drawDashChart() {
    destroy("dash");
    const canvas = el("mkDashChart");
    const sessions = (typeof App !== "undefined" ? App.sessions : []).filter((s) => s.is_valid !== false && s.status !== "invalid");
    const m = DASH_METRICS[state.dashMetric];
    if (!canvas) return;
    if (typeof Chart === "undefined" || sessions.length < 2) {
      canvas.parentElement.innerHTML = `<p class="chart-fallback">${sessions.length < 2 ? "Not enough sessions yet to draw a trend." : "Charts need internet access to load Chart.js."}</p>`;
      return;
    }
    const base = m.base(typeof App !== "undefined" ? App.baseline : null);
    const markers = markersFor(sessions);
    state.charts.dash = lineChart(canvas, { sessions, get: m.get, base, unit: m.unit, label: m.title, markers });
    const sum = el("mkDashChartSummary");
    if (sum) {
      const last = sessions[sessions.length - 1];
      sum.textContent = `${m.title}: ${sessions.length} sessions. Latest ${m.get(last)} ${m.unit}. Baseline ${num(base) ? base : "not available"} ${m.unit}. ${markers.filter((x) => x.type === "change").map((x) => x.label + " marked.").join(" ")}`;
    }
  }

  /* ------------------------------------------------------------------------
     AI INVESTIGATION
     ------------------------------------------------------------------------ */
  let replayTimer = null;

  function renderInvestigation({ replay = false } = {}) {
    const root = el("mkInvestigation");
    if (!root) return;
    const p = state.payload;
    if (state.loading) { root.innerHTML = LoadingState(); return; }
    if (!p) { root.innerHTML = ErrorState(state.error || "NO_DATA"); return; }
    const r = p.report;
    const finalKey = (() => { const was = state.replaying; state.replaying = false; const k = dataState(); state.replaying = was; return k; })();
    const key = replay ? "INVESTIGATING" : finalKey;
    const S = STATES[key];
    const steps = traceSteps(p);
    const why = whyFlagged(p);
    const titles = {
      PERSISTENT_CHANGE: "Persistent behavioral change",
      CHANGE_DETECTED: "Behavioral change detected",
      STABLE: "No behavioral change",
      MONITORING: "No behavioral change",
      BASELINE_FORMING: "Baseline still forming",
      INSUFFICIENT_EVIDENCE: "Evidence insufficient",
      NO_DATA: "No sessions to investigate",
      INVESTIGATING: "Behavioral change investigation",
    };

    const header = `
      <section class="mk-card mk-inv-header tone-${S.tone}" id="mkInvHeader">
        <div class="mk-inv-top">
          <div>
            <p class="mk-eyebrow">AI investigation</p>
            <h2 id="mkInvTitle">${esc(titles[finalKey])}</h2>
          </div>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <span id="mkInvBadge">${StatusBadge(key)}</span>
            ${SourceChip()}
          </div>
        </div>
        <div class="mk-meta">
          <span>Trigger session <b>${esc(p.finalAssessment.trigger_session_id || (r && r.session_id) || "—")}</b></span>
          <span>Iterations <b>${esc(p.iterations ?? "—")}</b></span>
          <span>Tools called <b>${esc((p.toolsCalled.length || 0) + (p.trace.length ? 1 : 0))}</b></span>
          <span>Trace events <b>${esc(p.traceCount || steps.length)}</b></span>
          <span>Stop reason <b>${esc(STOP_NAMES[p.stopReason] || human(p.stopReason) || "—")}</b></span>
          ${r && r.link ? `<span>Evidence digest <b class="mk-mono">${esc(r.link.evidence_digest.slice(0, 12))}…</b></span>` : ""}
        </div>
        <div class="btn-row" style="margin-top:14px">
          <button class="btn btn-ghost btn-sm" id="mkReplayBtn">${replay ? "Replaying…" : "Replay investigation"}</button>
          <button class="btn btn-ghost btn-sm" data-nav="lab">Try another scenario</button>
        </div>
      </section>`;

    const timeline = `
      <section class="mk-card" aria-labelledby="tlTitle">
        <div class="card-head" style="margin-bottom:6px"><h3 id="tlTitle" class="mk-card-title" style="margin:0">Investigation timeline</h3><span class="mk-muted" style="font-size:12px">${steps.length} steps</span></div>
        <ol class="mk-timeline" id="mkTimeline" aria-live="polite">
          ${steps.map((s, i) => `
            <li class="mk-tl-item kind-${s.kind} ${replay ? "pending" : "done"}" data-i="${i}">
              <span class="mk-tl-node" aria-hidden="true">${replay ? "" : CHECK}</span>
              <div>
                <div class="mk-tl-title"><span class="mk-tl-kind">${{ tool: "Tool", decision: "Decision", hyp: "Hypotheses", alt: "Alternatives", stop: "Stop", critic: "Critic", node: "Signal" }[s.kind] || "Step"}</span>${esc(s.title)}</div>
                ${s.detail ? `<p class="mk-tl-detail">${esc(s.detail)}</p>` : ""}
                ${s.items ? `<div class="mk-tl-sub">${s.items.map((it) => `<span class="mk-chip"><span class="st-text-${esc(it.st)}">●</span>${esc(it.id.startsWith("H") ? it.id : human(it.id))}: ${esc(STATUS_WORD[it.st] || it.st)}</span>`).join("")}</div>` : ""}
                ${s.evidence && s.evidence.length ? `<div class="mk-tl-sub">${[...new Set(s.evidence)].slice(0, 8).map((x) => `<span class="mk-evid">${esc(x)}</span>`).join("")}</div>` : ""}
                ${s.ts || s.raw ? `<p class="mk-tl-raw">${s.ts ? esc(String(s.ts).slice(11, 19)) + " · #" + esc(s.seq) : ""}${s.raw ? " · " + esc(s.raw.split("\n")[0]) : ""}</p>` : ""}
              </div>
            </li>`).join("")}
        </ol>
        <p class="mk-tl-foot">${p.trace.length ? "Every step is an event from the engine's recorded trace (deterministic clock). Replay paces the events for readability; order and content are unchanged." : "Rendered from the backend's investigation timeline."}</p>
      </section>`;

    const hyps = r ? `
      <section class="mk-card" aria-labelledby="hypTitle">
        <h3 id="hypTitle" class="mk-card-title">Hypotheses</h3>
        <div class="mk-hyp">${r.hypothesis_assessment.map(HypothesisCard).join("")}</div>
        <p class="mk-hyp-note" style="margin-top:10px">Qualitative states produced by the engine's evidence rules — not probabilities.</p>
      </section>` : "";

    const altFor = r ? r.alternatives.filter((a) => a.status === "weakened") : [];
    const altOpen = r ? r.alternatives.filter((a) => a.status === "supported" || a.status === "uncertain" || a.status === "unavailable") : [];
    const forAgainst = `
      <section class="mk-card" aria-labelledby="faTitle">
        <h3 id="faTitle" class="mk-card-title">Evidence for and against a real change</h3>
        <div class="mk-forag">
          <div class="for"><h4>Supports a real change</h4><ul>
            ${why.reasons.length && finalKey !== "STABLE" && finalKey !== "MONITORING" ? why.reasons.slice(0, 4).map((x) => `<li>${esc(x.text)}</li>`).join("") : "<li>No signal moved meaningfully from the baseline</li>"}
            ${altFor.slice(0, 2).map((a) => `<li>${esc(a.statement)} explanation weakened</li>`).join("")}
          </ul></div>
          <div class="against"><h4>Other explanations still open</h4><ul>
            ${altOpen.filter((a) => a.status === "supported").map((a) => `<li>${esc(a.statement)} — supported</li>`).join("")}
            ${why.context.slice(0, 2).map((c) => `<li>${esc(c)}</li>`).join("")}
            ${(p.dataQuality.valid_sessions ?? 99) < (p.minimum || 10) ? `<li>Only ${esc(p.dataQuality.valid_sessions)} valid sessions — below the model minimum of ${esc(p.minimum)}</li>` : ""}
          </ul></div>
        </div>
      </section>`;

    const whyCard = `
      <section class="mk-card" aria-labelledby="whyFTitle">
        <h3 id="whyFTitle" class="mk-card-title">Why was this flagged?</h3>
        ${finalKey === "STABLE" || finalKey === "MONITORING"
          ? `<p class="mk-muted">Nothing was flagged. Recent sessions match the personal baseline.</p>`
          : `<p style="font-size:14px;color:var(--text-2);margin-bottom:10px">MindKey ${finalKey === "PERSISTENT_CHANGE" ? "detected a persistent deviation" : "looked closer"} because:</p>
             <ol class="mk-list">${why.reasons.map((x, i) => `<li><span class="ix">${i + 1}</span><span class="body">${esc(x.text)} ${x.ids.map((id) => `<span class="mk-evid">${esc(id)}</span>`).join(" ")}</span></li>`).join("")}</ol>`}
        <h4 style="font-size:13px;font-weight:650;margin:16px 0 8px">Important context</h4>
        <ul class="mk-list">${why.context.map((c) => `<li><span class="ix">i</span><span class="body">${esc(c)}</span></li>`).join("")}</ul>
      </section>`;

    const sigs = signalRows(p);
    const depth = p.evidence.find((e) => e.kind === "persistence_depth");
    const robust = p.evidence.find((e) => e.kind === "persistence_robustness");
    const anomaly = p.evidence.find((e) => e.kind === "anomaly");
    const detectRow = (ok, label, note) => `<li><span class="ck ${ok === true ? "yes" : ok === false ? "no" : "na"}" aria-hidden="true">${ok === true ? "✓" : ok === false ? "–" : "?"}</span><span>${esc(label)}${note ? ` <span class="mk-muted">· ${esc(note)}</span>` : ""}</span></li>`;
    const explorer = sigs.length ? `
      <section class="mk-card" aria-labelledby="evxTitle">
        <div class="card-head" style="margin-bottom:8px"><h3 id="evxTitle" class="mk-card-title" style="margin:0">Evidence explorer</h3><span class="mk-muted" style="font-size:12px">Click a signal to see the underlying evidence</span></div>
        <div class="mk-evx">
          <div class="evx-head"><span>Signal</span><span>Current</span><span>Baseline (30 d)</span><span>Change</span><span></span></div>
          ${sigs.map((s) => {
            const moved = num(s.relative_change) && Math.abs(s.relative_change) >= p.movedThreshold;
            const persisted = depth && depth.status === "sustained" && moved;
            return `<details>
              <summary>
                <span class="lbl">${esc(s.label)}</span>
                <span class="num">${num(s.value) ? esc(s.sig.fmt(s.value)) : "—"} ${esc(s.sig.unit)}</span>
                <span class="num">${num(s.baseline_value) ? esc(s.sig.fmt(s.baseline_value)) : "—"} ${esc(s.sig.unit)}</span>
                <span class="chg">${pct(s.relative_change)}</span>
                <span class="chev" aria-hidden="true">›</span>
              </summary>
              <div class="mk-evx-body">
                <dl class="mk-kv">
                  <dt>Latest session</dt><dd>${num(s.value) ? esc(s.sig.fmt(s.value)) : "—"} ${esc(s.sig.unit)}</dd>
                  <dt>7-day mean</dt><dd>${num(s.recent7) ? esc(s.sig.fmt(s.recent7)) : "—"} ${esc(s.sig.unit)} ${s.recent7Id ? `<span class="mk-evid">${esc(s.recent7Id)}</span>` : ""}</dd>
                  <dt>30-day baseline</dt><dd>${num(s.baseline_value) ? esc(s.sig.fmt(s.baseline_value)) : "—"} ${esc(s.sig.unit)}</dd>
                  <dt>Change</dt><dd>${pct(s.relative_change)} (${esc(s.direction || "—")})</dd>
                  <dt>Persistence</dt><dd>${depth && num(depth.value) ? `${esc(depth.value)} of ${esc(depth.session_count ?? "—")} recent sessions` : "not assessed"}</dd>
                </dl>
                <div>
                  <p class="mk-eyebrow" style="margin-bottom:6px">Detected by</p>
                  <ul class="mk-detect">
                    ${detectRow(moved ? true : false, `Deviation beyond ${Math.round(p.movedThreshold * 100)}% of personal baseline`, s.id)}
                    ${detectRow(anomaly ? (anomaly.status === "true") : null, "Isolation Forest (per-user model)", anomaly ? `${anomaly.id}, trigger session` : "no stored result")}
                    ${detectRow(depth ? !!persisted : null, "Temporal persistence", depth ? `${depth.id} · ${human(depth.status)}` : "not assessed")}
                    ${detectRow(robust ? robust.status === "agrees" : null, "Window robustness", robust ? `${robust.id} · ${human(robust.status)}` : "not assessed")}
                  </ul>
                </div>
                <p class="mk-evx-stmt">${esc(s.statement)}</p>
              </div>
            </details>`;
          }).join("")}
        </div>
      </section>` : (r ? `
      <section class="mk-card"><h3 class="mk-card-title">Observations</h3><ul class="mk-list">${r.observations.slice(0, 8).map((o) => `<li><span class="ix">·</span><span class="body">${esc(o.statement)}</span></li>`).join("")}</ul></section>` : "");

    const alts = r ? `
      <section class="mk-card" aria-labelledby="altTitle">
        <h3 id="altTitle" class="mk-card-title">Alternative explanations checked</h3>
        <div class="mk-alts">${r.alternatives.map((a) => `<div class="mk-alt"><span>${esc(a.statement)}</span><span class="st st-text-${esc(a.status)}">${esc(STATUS_WORD[a.status] || human(a.status))}</span></div>`).join("")}</div>
        <p class="mk-hyp-note" style="margin-top:10px">Sleep, fatigue, stress, illness and distraction are decided by your check-ins from the last 7 days. "Unavailable" means no check-in or other data source can answer it; "Partially evaluated" means you checked in without mentioning it.</p>
      </section>` : "";

    const decision = r ? `
      <section class="mk-card" aria-labelledby="decTitle" id="mkDecision">
        <h3 id="decTitle" class="mk-card-title">Agent decision</h3>
        <div class="mk-decision">
          <blockquote>“${esc(r.conclusion.statement)}${finalKey === "PERSISTENT_CHANGE" || finalKey === "CHANGE_DETECTED" ? " MindKey will continue monitoring." : ""}”</blockquote>
          <p class="mk-muted" style="margin-top:8px;font-size:13px">Next: ${esc(r.next_action || "Continue monitoring.")}</p>
          <p class="mk-eyebrow" style="margin-top:14px">Uncertainty: ${esc(r.uncertainty.level)}</p>
          <div class="mk-reasons">${r.uncertainty.reasons.map((x) => `<span class="mk-chip">${esc(REASONS[x] || human(x))}</span>`).join("")}</div>
        </div>
        <details style="margin-top:12px"><summary style="cursor:pointer;font-size:13px;color:var(--text-2)">Limitations (${r.limitations.length})</summary>
          <ul class="mk-list" style="margin-top:8px">${r.limitations.map((l) => `<li><span class="ix">·</span><span class="body" style="font-size:13px;color:var(--text-2)">${esc(l)}</span></li>`).join("")}</ul></details>
        <p class="mk-disclaimer">${esc(r.disclaimer)}</p>
      </section>` : "";

    root.innerHTML = `${header}
      <div class="mk-section mk-grid-main">
        <div class="mk-col">${timeline}</div>
        <div class="mk-col" id="mkInvRight">${whyCard}${hyps}${forAgainst}</div>
      </div>
      <div class="mk-section mk-stack">${explorer}${alts}${decision}</div>`;

    el("mkReplayBtn").addEventListener("click", () => replayInvestigation());
    if (replay) runReplay(steps.length, finalKey);
  }

  function replayInvestigation() {
    state.replaying = true;
    renderInvestigation({ replay: true });
  }

  function runReplay(n, finalKey) {
    clearInterval(replayTimer);
    const items = [...document.querySelectorAll("#mkTimeline .mk-tl-item")];
    const right = el("mkInvRight");
    if (right) right.style.opacity = "0.4";
    let i = 0;
    const finish = () => {
      clearInterval(replayTimer);
      state.replaying = false;
      items.forEach((it) => { it.classList.remove("pending", "active"); it.classList.add("done"); it.querySelector(".mk-tl-node").innerHTML = CHECK; });
      if (right) right.style.opacity = "";
      const badge = el("mkInvBadge");
      if (badge) badge.innerHTML = StatusBadge(finalKey);
      const btn = el("mkReplayBtn");
      if (btn) btn.textContent = "Replay investigation";
      renderDashboard();
    };
    if (reduced()) return finish();
    const tick = () => {
      if (i > 0) { const prev = items[i - 1]; prev.classList.remove("active"); prev.classList.add("done"); prev.querySelector(".mk-tl-node").innerHTML = CHECK; }
      if (i >= items.length) return finish();
      const cur = items[i];
      cur.classList.remove("pending");
      cur.classList.add("active", "enter");
      requestAnimationFrame(() => cur.classList.remove("enter"));
      if (i > 3) cur.scrollIntoView({ block: "nearest", behavior: "smooth" });
      i++;
    };
    tick();
    replayTimer = setInterval(tick, 380);
  }

  /* ------------------------------------------------------------------------
     DEMO LAB
     ------------------------------------------------------------------------ */
  let labTimer = null;
  let labSelected = null;

  function renderLab() {
    const root = el("mkLab");
    if (!root) return;
    labSelected = labSelected || state.scenarioId || "persistent";
    const live = typeof API_BASE === "string" && API_BASE;
    root.innerHTML = `
      <div class="mk-banner" role="note">
        <span aria-hidden="true">ⓘ</span>
        <span><b>Demo data, real engine.</b> Each scenario is a set of synthetic typing sessions and check-ins. Every session is scored by the real per-user Isolation Forest, the same way the backend scores new sessions, and the investigation is the real MindKey agent (${live ? "fetched live from <code>/api/demo</code>" : "precomputed by <code>backend/scripts/export_demo.py</code>; a test keeps it identical to the engine"}). Nothing here is typed text, and no result is hand-written.</span>
      </div>
      <section class="mk-section" aria-labelledby="scenTitle">
        <div class="mk-section-head"><h3 id="scenTitle">1 · Choose a scenario</h3></div>
        <div class="mk-scen-grid" role="group" aria-label="Demo scenarios">
          ${SCENARIOS.map((s) => `<button class="mk-scen" data-scen="${s.id}" aria-pressed="${s.id === labSelected}">
            <b>${esc(s.title)}</b><p>${esc(s.blurb)}</p><span class="engine">engine: ${esc(s.key)}</span></button>`).join("")}
        </div>
        <div class="mk-lab-run">
          <button class="btn btn-primary" id="mkRunBtn">Run scenario</button>
          <span class="mk-muted" style="font-size:13px" id="mkRunNote">The whole dashboard — status, trends, sessions and investigation — updates to this scenario.</span>
        </div>
      </section>
      <section class="mk-section" aria-labelledby="flowTitle">
        <div class="mk-section-head"><h3 id="flowTitle">2 · Watch the pipeline</h3><p class="section-note">Data → ML → Agent → Evidence → Insight</p></div>
        <div class="mk-stages" id="mkStages" aria-live="polite">${stagesHTML(null)}</div>
      </section>`;

    root.querySelectorAll(".mk-scen").forEach((b) =>
      b.addEventListener("click", () => {
        labSelected = b.dataset.scen;
        root.querySelectorAll(".mk-scen").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      })
    );
    el("mkRunBtn").addEventListener("click", () => runScenario(labSelected));
    if (state.payload && state.scenarioId === labSelected) el("mkStages").innerHTML = stagesHTML(state.payload, 5);
  }

  function stagesHTML(p, upTo = 0, tickerLines = []) {
    const cls = (i) => (upTo > i ? "done" : upTo === i ? "active" : "idle");
    const dq = p ? p.dataQuality : {};
    const ml = p ? p.mlEvidence : null;
    const per = p ? p.persistence : {};
    const hy = p && p.report ? p.report.hypothesis_assessment : [];
    const S = p ? STATES[(() => { const w = state.replaying; state.replaying = false; const k = dataState(p); state.replaying = w; return k; })()] : null;
    const first = p && p.sessions.length ? fmtShort(p.sessions[0].session_start) : "";
    const last = p && p.sessions.length ? fmtShort(p.sessions[p.sessions.length - 1].session_start) : "";
    return `
      <div class="mk-stage ${cls(0)}"><h4>Data <span>1</span></h4>
        ${p && upTo > 0 ? `<p class="big">${p.sessions.length} sessions</p><p>${esc(dq.valid_sessions)} valid · ${esc(first)} – ${esc(last)}</p><p>Timing features only — no typed content.</p>` : `<p>Behavioral sessions from the desktop agent.</p>`}</div>
      <div class="mk-stage ${cls(1)}"><h4>ML <span>2</span></h4>
        ${p && upTo > 1 ? `<p class="big">${ml && ml.status === "ok" ? (ml.is_anomaly ? "Flagged" : "Not flagged") : "No model result"}</p>
          <p>${ml && ml.status === "ok" ? `Isolation Forest on the trigger session` : `No stored anomaly result${(dq.valid_sessions ?? 0) < (p.minimum || 10) ? ` — needs ${p.minimum} valid sessions` : ""}`}</p>
          <p>${num(per.signals_moved) ? `${per.signals_moved} of 6 signals moved vs. baseline` : ""}</p>` : `<p>Personal baseline, statistical drift and Isolation Forest.</p>`}</div>
      <div class="mk-stage ${cls(2)}"><h4>Agent <span>3</span></h4>
        ${p && upTo > 2 ? `<p class="big">${(p.toolsCalled.length || 0) + 1} tool calls</p><p>${esc(p.iterations)} iteration${p.iterations === 1 ? "" : "s"} · ${esc(p.traceCount)} trace events</p>` : `<p>Plans evidence, calls tools, stops when sufficient.</p>`}
        ${tickerLines.length ? `<div class="ticker">${tickerLines.slice(-4).map(esc).join("<br>")}</div>` : ""}</div>
      <div class="mk-stage ${cls(3)}"><h4>Evidence <span>4</span></h4>
        ${p && upTo > 3 ? `<p class="big">${p.evidence.length} items</p><p>${hy.map((h) => `${h.id} ${STATUS_WORD[h.status] || h.status}`).join(" · ")}</p><p>Critic: ${p.critic && /is_sound=True/.test(p.critic.detail) ? "0 unsupported claims" : "reviewed"}</p>` : `<p>Hypotheses and alternatives, each linked to evidence IDs.</p>`}</div>
      <div class="mk-stage ${cls(4)}"><h4>Insight <span>5</span></h4>
        ${p && upTo > 4 ? `<p style="margin-top:8px">${S ? `<span class="mk-status tone-${S.tone}"><span class="dot"></span>${esc(S.label)}</span>` : ""}</p><p style="font-size:13px;color:var(--text)">${esc(p.conclusion.statement)}</p>
          <button class="btn btn-primary btn-sm" data-nav="investigation" style="margin-top:10px">Open investigation</button>` : `<p>A grounded, non-diagnostic explanation.</p>`}</div>`;
  }

  async function runScenario(id) {
    clearInterval(labTimer);
    const btn = el("mkRunBtn");
    if (btn) { btn.disabled = true; btn.textContent = "Running…"; }
    const stages = el("mkStages");
    if (stages) stages.innerHTML = stagesHTML(null, 0);
    const p = await loadScenario(id);
    if (!p) {
      if (stages) stages.innerHTML = ErrorState("agent_unavailable");
      if (btn) { btn.disabled = false; btn.textContent = "Run scenario"; }
      return;
    }
    applyToDashboard();
    if (typeof onScenarioApplied === "function") onScenarioApplied();
    const steps = traceSteps(p);
    let stage = 1;
    let ticker = [];
    let si = 0;
    const done = () => {
      clearInterval(labTimer);
      if (stages) stages.innerHTML = stagesHTML(p, 5, ticker);
      if (btn) { btn.disabled = false; btn.textContent = "Run scenario"; }
      if (typeof showToast === "function") showToast(`Scenario loaded: ${byId(id).title}. The dashboard now shows this data.`);
      renderDashboard();
    };
    if (reduced()) return done();
    labTimer = setInterval(() => {
      if (stage === 3 && si < steps.length) {
        ticker.push(`#${steps[si].seq} ${steps[si].title}`);
        si += 3;
        if (stages) stages.innerHTML = stagesHTML(p, 3, ticker);
        return;
      }
      stage++;
      if (stage > 5) return done();
      if (stages) stages.innerHTML = stagesHTML(p, stage, ticker);
    }, 420);
    if (stages) stages.innerHTML = stagesHTML(p, 1);
  }

  /* ------------------------------------------------------------------------
     Wellbeing context timeline (demo scenario check-ins)
     ------------------------------------------------------------------------ */
  function renderWellbeingContext() {
    const root = el("mkWellbeingContext");
    if (!root) return;
    const p = state.payload;
    const demo = !p || p.source === "demo";
    const events = contextEvents(p);
    const findings = p ? contextFindings(p).slice(0, -1) : [];
    root.innerHTML = `
      <div class="mk-card" style="margin-bottom:16px">
        <div class="card-head" style="margin-bottom:8px"><h3 class="mk-card-title" style="margin:0">Context timeline</h3>${demo && p ? `<span class="mk-chip mk-chip-demo">Scenario · ${esc(byId(state.scenarioId).title)}</span>` : ""}</div>
        ${ContextList(events)}
        ${findings.length ? `<p class="card-note" style="margin-top:10px"><strong>What the agent made of it:</strong> ${esc(findings.join(" "))}</p>` : ""}
        <p class="card-note" style="margin-top:6px">${demo
          ? "These are the scenario's check-ins, and the agent read them. Check-ins you add below in demo mode stay in this browser and don't change the scenario."
          : "The AI investigation reads your check-ins from the last 7 days — the day and the factor only. Notes stay private. Each new check-in re-runs the investigation."}</p>
      </div>`;
  }

  /* ------------------------------------------------------------------------
     Theme
     ------------------------------------------------------------------------ */
  function initTheme() {
    let t = null;
    try { t = localStorage.getItem("mindkey.theme"); } catch (e) { /* optional */ }
    if (t === "dark" || t === "light") document.documentElement.dataset.theme = t;
  }
  function toggleTheme() {
    const cur = document.documentElement.dataset.theme ||
      (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("mindkey.theme", next); } catch (e) { /* optional */ }
    if (typeof App !== "undefined" && typeof rerenderCurrentView === "function") rerenderCurrentView();
  }

  function scenarioChip() {
    const c = el("mkScenarioChip");
    if (!c) return;
    const live = state.source !== "demo";
    c.innerHTML = live ? `<span class="lbl">Live data</span>` : `<span class="lbl">Scenario:</span> <strong>${esc(byId(state.scenarioId).title)}</strong>`;
  }

  /* What the Insights page needs, all from the engine's output. */
  function insightModel() {
    const p = state.payload;
    const was = state.replaying;
    state.replaying = false;
    const key = dataState(p);
    state.replaying = was;
    const r = p && p.report;
    const hyp = (id) => (r ? (r.hypothesis_assessment.find((h) => h.id === id) || {}).status : null);
    const ctxIds = new Set(["poor_sleep", "fatigue", "stress", "illness_or_mood", "distraction"]);
    const supportedContext = r
      ? r.alternatives.filter((a) => ctxIds.has(a.id) && a.status === "supported").map((a) => a.statement.toLowerCase())
      : [];
    const depth = p ? p.evidence.find((e) => e.kind === "persistence_depth") : null;
    return {
      key,
      state: STATES[key],
      conclusion: p ? p.conclusion : null,
      contextExplains: hyp("H2") === "supported",
      contextRuledOut: hyp("H2") === "weakened",
      supportedContext,
      depth,
      contextLines: p ? contextFindings(p).slice(0, -1) : [],
      events: contextEvents(p),
    };
  }

  /* Legacy "typing state" (Insights page) derived from the engine conclusion */
  function legacyTypingState() {
    const k = dataState();
    if (k === "PERSISTENT_CHANGE") return "alert";
    if (k === "CHANGE_DETECTED") return "warn";
    return "ok";
  }

  return {
    SCENARIOS, STATES, state, byId,
    loadScenario, loadLive, savedScenario, applyToDashboard, dataState,
    renderDashboard, drawDashChart, renderInvestigation, renderLab, renderWellbeingContext,
    replayInvestigation, runScenario, markersFor, markerPlugin, initTheme, toggleTheme, scenarioChip,
    legacyTypingState, insightModel,
    setDashMetric(k) { if (DASH_METRICS[k]) { state.dashMetric = k; renderDashboard(); } },
  };
})();

MK.initTheme();
