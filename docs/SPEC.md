You are the **Senior Product Designer + Frontend Engineer** responsible for upgrading the existing MindKey web application.

The live application is currently available at:

https://mindkey-hznm24bdp-akhilkadambala3-stars-projects.vercel.app/

The existing repository is already functional and contains a behavioral monitoring dashboard.

DO NOT rebuild the product from scratch.

Your job is to transform the current prototype into a polished, technically credible, hackathon-quality AI product interface.

The project is called:

# MindKey

### Longitudinal AI Behavioral Change Monitoring

The project combines:

* behavioral data collection
* personalized ML baselines
* anomaly/change detection
* temporal analysis
* contextual wellbeing signals
* an Agentic AI investigation system
* explainable evidence
* privacy-by-design

Another developer is building the ML system.

Another developer is building the Agentic AI system.

Your frontend must become the interface that makes those systems understandable and impressive.

---

# 1. FIRST — AUDIT THE EXISTING APPLICATION

Before changing anything:

Inspect the entire frontend repository.

Identify:

* framework
* components
* pages
* routing
* state management
* API layer
* chart implementation
* mock/demo data
* backend integration
* styling system
* responsive behavior
* reusable components
* loading/error states
* accessibility
* current design tokens

Also inspect the current live application and compare it against the code.

Do not assume the live deployment exactly matches the repository.

Create a concise:

KEEP
MODIFY
REPLACE
ADD

plan before implementing major changes.

---

# 2. PRODUCT POSITIONING

The interface should NOT feel like:

"medical diagnosis dashboard"

and should NOT imply:

"MindKey can diagnose dementia."

Instead communicate:

> MindKey learns your personal behavioral baseline and helps identify persistent changes in everyday interaction patterns.

The product should feel:

* intelligent
* trustworthy
* calm
* modern
* privacy-conscious
* technically sophisticated
* human-centered
* premium

Avoid fear-based healthcare UI.

Avoid giant red warning screens.

Avoid disease-risk percentages.

Avoid fake clinical certainty.

---

# 3. DESIGN DIRECTION

Create a premium AI/health-tech aesthetic.

Think:

* modern AI observability dashboard
* Apple Health-level simplicity
* Linear/Vercel-level interface quality
* subtle data visualization
* calm clinical design
* sophisticated dark/light surfaces
* strong typography
* restrained gradients
* subtle motion

Do NOT make it look like:

* generic SaaS
* generic ChatGPT wrapper
* crypto dashboard
* cyberpunk dashboard
* overly neon AI website
* university project

The design should look like something that could realistically become a startup product.

---

# 4. INFORMATION ARCHITECTURE

Improve the application into clear sections.

Recommended navigation:

Dashboard
Insights
Trends
AI Investigation
Sessions
Wellbeing
Privacy
Profile

Do not create unnecessary pages if the existing application can handle the functionality more cleanly.

---

# 5. REDESIGN THE MAIN DASHBOARD

The current dashboard primarily communicates:

"Your typing pattern is consistent."

Keep the useful information but make the hierarchy much stronger.

The dashboard should immediately answer:

1. How am I doing?
2. Has anything changed?
3. Why does MindKey think that?
4. What has the ML system observed?
5. Is the change persistent?
6. What is the AI investigating?

Suggested structure:

---

MindKey

Your behavioral baseline is stable

[Status]

Your recent interaction patterns remain within your personal baseline.

---

## Behavioral Overview

Typing Speed
238 WPM
↓ 6.4%

Correction Rate
7.2%
↑ 2.1%

Pause Frequency
4.3
↑ 0.4

Rhythm Stability
92%
↓ 1.8%

---

## Behavioral Trend

Interactive chart

Personal baseline
Recent behavior
Change points
Confidence/uncertainty

---

## AI Investigation

[No active investigation]

OR

[Investigation in progress]

"MindKey detected a persistent change across 3 signals."

[View investigation]

---

## What changed?

Show the most important evidence.

---

## Context

Sleep
Stress
Fatigue
etc.

---

Do not overwhelm the user with every metric simultaneously.

---

# 6. CREATE A REAL "AI INVESTIGATION" EXPERIENCE

This is the most important UI addition.

The agent should NOT look like a chatbot.

It should look like an investigation system.

Create a dedicated:

# AI Investigation

page.

Example:

---

AI INVESTIGATION

Behavioral change detected

Status:
Investigating

---

### Investigation Timeline

✓ New behavioral deviation detected

✓ Compared with 30-day personal baseline

✓ Checked persistence across sessions

✓ Examined recent behavioral trends

● Checking contextual factors...

○ Evidence synthesis

---

### Evidence

Typing speed
-16.5%

Correction rate
+39%

Pause frequency
+54%

Persistence
6 sessions

---

### Hypotheses

Temporary variation
████████

Contextual factor
██████

Persistent behavioral change
████

Data artifact
██

These must NOT be fake probabilities unless produced by an actual model.

If these are not real probabilities, represent them as qualitative states:

Supported
Possible
Uncertain
Weakened

---

### Evidence For

• change persisted across 6 sessions
• multiple signals changed
• baseline is stable

### Evidence Against

• poor sleep reported recently
• insufficient long-term data

---

### Agent Decision

"Evidence is currently insufficient to determine the cause of the behavioral change. MindKey will continue monitoring."

---

This should be one of the centerpiece features of the application.

---

# 7. AI ACTIVITY TIMELINE

Create a visually excellent investigation timeline.

Example:

09:42:12
Signal received

09:42:13
Baseline retrieved

09:42:13
Temporal trend analyzed

09:42:14
Persistence confirmed

09:42:15
Context retrieved

09:42:16
Alternative explanations evaluated

09:42:17
Evidence synthesis completed

09:42:18
Investigation complete

Make this feel like an AI agent actually working.

Use subtle animation.

Do not use excessive animations.

---

# 8. EVIDENCE EXPLORER

Create an expandable evidence panel.

Example:

Typing speed

Current:
238 WPM

Baseline:
285 WPM

Change:
-16.5%

Historical deviation:
2.1 robust standard deviations

Persistence:
6 sessions

Detected by:

✓ Statistical deviation
✓ Isolation Forest
✓ Temporal persistence

Clicking the signal should reveal the underlying evidence.

This is important because it demonstrates that the AI output is grounded in actual ML.

---

# 9. "WHY DID MINDKEY FLAG THIS?"

Create a dedicated explanation card.

Example:

### Why was this flagged?

MindKey detected a persistent deviation because:

1. typing speed decreased 16.5%
2. correction rate increased 39%
3. pause frequency increased 54%
4. the pattern persisted across 6 sessions

Then:

### Important context

Poor sleep was reported during 3 of those sessions.

This may explain some of the observed variation.

The system cannot determine the cause from behavioral data alone.

This section should make the product feel trustworthy.

---

# 10. TRENDS PAGE

Create a sophisticated longitudinal visualization.

Allow:

7 days
30 days
90 days
All time

Allow switching between:

Speed
Dwell
Flight
Corrections
Pauses
Rhythm
Composite behavioral change

The chart should show:

* personal baseline
* current behavior
* confidence/uncertainty where available
* change points
* persistent periods
* contextual events

For example:

```
   baseline ───────────────────
```

behavior
╲
╲
╲────── change point
╲
╲

Allow users to hover and understand exactly what happened.

---

# 11. DO NOT CREATE A SINGLE "RISK SCORE"

Do NOT make:

"72% cognitive risk"

or:

"Probability of dementia: 83%"

or similar.

Instead show:

Behavioral Change

Minimal
Moderate
Persistent

and explain the evidence.

The product is about detecting behavioral change, not diagnosing disease.

---

# 12. WELLBEING EXPERIENCE

The current wellbeing check-in is useful.

Improve it into a clean contextual timeline.

For example:

### Context timeline

Sep 18
Poor sleep

Sep 19
High stress

Sep 20
Normal

Sep 21
Traveling

Overlay these events on the behavioral chart.

This allows users to visually understand:

"Behavior changed around the same period as poor sleep."

That is far more useful than a generic questionnaire.

---

# 13. SESSION EXPLORER

Create a session table/card system.

Each session should show:

Date
Duration
Typing speed
Dwell
Flight
Corrections
Pauses
Behavioral status

Click a session to open:

### Session details

Feature values
vs baseline
Deviation
Data quality
Detection signals

This makes the system transparent.

---

# 14. PRIVACY CENTER

This should become a major differentiator.

Create:

# Privacy Center

Show:

### What MindKey collects

✓ Timing-derived typing features
✓ Session metadata
✓ Optional wellbeing responses
✓ Optional symptom responses

### What MindKey does NOT collect

✕ typed words
✕ passwords
✕ message contents
✕ documents
✕ private conversations

Use a visual architecture diagram:

Your device
↓
Feature extraction
↓
Aggregated behavioral metrics
↓
MindKey server

Make the privacy architecture easy to understand.

---

# 15. DATA CONTROL

Create obvious controls:

Pause monitoring

View collected data

Delete data

Application exclusions

Export data

Make these real if backend support exists.

If functionality is not implemented:

Clearly mark:

"Planned"

or

"Demo"

Do not pretend a feature works when it doesn't.

---

# 16. DEMO MODE

The current application already contains simulated data.

Keep this.

But turn it into a proper:

# Demo Lab

Allow judges to choose:

Stable baseline
Temporary fatigue
Persistent change
Sudden change
Recovery
Noisy data
Insufficient data

Then click:

[Run Scenario]

The entire dashboard should update.

This is VERY important for hackathons.

It lets judges understand the product in 30 seconds without installing the desktop agent.

---

# 17. CREATE A "LIVE AI" DEMO

For a selected scenario:

1. generate/activate behavioral event
2. ML detects change
3. investigation begins
4. timeline animates
5. agent calls tools
6. evidence appears
7. hypotheses update
8. final explanation appears

The demo should visually communicate:

DATA
→ ML
→ AGENT
→ EVIDENCE
→ INSIGHT

This should become the centerpiece of the presentation.

---

# 18. LOADING STATES

Do NOT show blank cards.

Create meaningful loading states.

Example:

Analyzing recent behavior...

Retrieving personal baseline...

Checking temporal persistence...

Evaluating contextual factors...

Building evidence summary...

These can correspond to actual backend/agent events when available.

Do not fake progress in production mode.

---

# 19. ERROR STATES

Create excellent error handling.

Examples:

ML unavailable

"Behavioral analysis is temporarily unavailable. Your collected data remains safe."

Agent unavailable

"AI investigation could not be completed. View the underlying behavioral evidence."

Insufficient data

"MindKey needs more sessions before it can establish a reliable personal baseline."

No sessions

"Start monitoring to begin building your personal baseline."

---

# 20. RESPONSIVE DESIGN

The application must work properly on:

Desktop
Laptop
Tablet
Mobile

Do not simply shrink desktop cards.

Reorganize the information hierarchy for smaller screens.

---

# 21. ACCESSIBILITY

Implement:

* keyboard navigation
* accessible labels
* sufficient contrast
* reduced-motion support
* readable font sizes
* semantic HTML
* accessible charts where possible
* clear focus states

Do not sacrifice accessibility for aesthetics.

---

# 22. MICROINTERACTIONS

Use subtle interactions:

* chart transitions
* status changes
* investigation events
* expanding evidence
* hover details
* baseline reveal
* session selection

Avoid:

* excessive bouncing
* huge animated gradients
* distracting particles
* unnecessary 3D effects

The product should feel premium, not gimmicky.

---

# 23. DESIGN SYSTEM

Create reusable design tokens:

colors
typography
spacing
border radius
shadows
transitions
status colors
chart styles

Create reusable components:

MetricCard
TrendChart
EvidenceCard
InvestigationTimeline
HypothesisCard
ContextEvent
StatusBadge
PrivacyCard
SessionTable
InsightCard

Avoid duplicating styling across pages.

---

# 24. DATA STATES

The UI must distinguish:

NO_DATA

BASELINE_FORMING

STABLE

MONITORING

CHANGE_DETECTED

INVESTIGATING

PERSISTENT_CHANGE

INSUFFICIENT_EVIDENCE

These should have clear visual states.

---

# 25. FRONTEND ↔️ ML CONTRACT

The frontend should consume structured ML outputs rather than hardcoding interpretations.

Example:

{
"feature": "typing_speed",
"baseline": 285,
"current": 238,
"change_percent": -16.5,
"persistence": 6,
"trend": "declining",
"detectors": [
"robust_zscore",
"isolation_forest"
]
}

Display the evidence dynamically.

Do not invent values.

---

# 26. FRONTEND ↔️ AGENT CONTRACT

The frontend should consume agent states such as:

INVESTIGATION_STARTED

PLANNING

RETRIEVING_HISTORY

ANALYZING_TRENDS

CHECKING_CONTEXT

EVALUATING_HYPOTHESES

SYNTHESIZING

COMPLETED

FAILED

Then render the investigation timeline accordingly.

---

# 27. HACKATHON MODE

Add a hidden or visible "Demo Mode" that lets judges experience the complete system.

Suggested flow:

[Launch Demo]

↓

Scenario:

"Persistent behavioral change"

↓

[Start Investigation]

↓

ML evidence appears

↓

Agent begins investigation

↓

Timeline updates

↓

Hypotheses appear

↓

Evidence gets collected

↓

Final explanation

↓

"View Evidence"

This should be extremely polished.

---

# 28. LANDING / FIRST IMPRESSION

The first screen should immediately communicate:

MindKey

"Understand changes in your everyday behavior."

Subtext:

"MindKey learns your personal behavioral baseline and uses ML + agentic investigation to identify persistent changes — without collecting what you type."

Then:

[Explore Demo]

[View Insights]

Avoid excessive marketing copy.

The product should demonstrate intelligence rather than claim it.

---

# 29. DO NOT ADD FAKE AI

This is extremely important.

Do not create UI that says:

"AI analyzing..."

if nothing is actually happening.

Do not fake agent traces.

Do not hardcode fake model results in production mode.

Demo mode may use deterministic simulated scenarios, but clearly distinguish:

DEMO DATA

from

LIVE DATA

The interface must be truthful.

---

# 30. PERFORMANCE

Optimize for:

* fast initial load
* lazy-loaded charts
* minimal unnecessary requests
* efficient state updates
* mobile performance
* API caching where appropriate

Do not sacrifice performance for animations.

---

# 31. FINAL VISUAL GOAL

When a hackathon judge opens MindKey, within 10 seconds they should understand:

1. This learns my personal baseline.
2. It detects behavioral changes.
3. ML provides the evidence.
4. An AI agent investigates the evidence.
5. I can see why the system reached its conclusion.
6. My raw typing content is not being collected.

The product should feel like:

AI observability
+
personal analytics
+
behavioral intelligence
+
agentic investigation

—not a generic health dashboard.

---

# 32. IMPLEMENTATION RULE

Do not modify everything at once.

Work in this order:

PHASE 1
Audit existing UI and architecture.

PHASE 2
Create design system/components.

PHASE 3
Redesign dashboard.

PHASE 4
Build Trends + Evidence Explorer.

PHASE 5
Build AI Investigation interface.

PHASE 6
Connect real ML outputs.

PHASE 7
Connect real agent states.

PHASE 8
Build Demo Lab.

PHASE 9
Privacy Center.

PHASE 10
Responsive/accessibility/performance polish.

After each phase:

* run the application
* test desktop
* test mobile
* inspect console
* verify API behavior
* verify no existing functionality broke

Do not proceed to the next phase until the current phase works.

---

# 33. FINAL STANDARD

Do not optimize for:

"How many features can we add?"

Optimize for:

"Can a judge understand the intelligence of the system in 60 seconds?"

The final product should tell a coherent story:

PERSONAL BASELINE
↓
BEHAVIORAL CHANGE
↓
ML EVIDENCE
↓
AGENT INVESTIGATION
↓
ALTERNATIVE EXPLANATIONS
↓
EVIDENCE-GROUNDED INSIGHT
↓
USER UNDERSTANDING

Build the interface around this story.

