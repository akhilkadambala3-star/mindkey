"""MindKey ML layer.

Detects deviation from a user's normal typing behavior using keystroke
dynamics (dwell time, flight time, typing speed, corrections, rhythm, pauses)
and scikit-learn's Isolation Forest.

Scope and boundaries:
- This layer only flags sessions that deviate from an individual's own
  historical typing pattern. It does NOT diagnose Parkinson's disease,
  Alzheimer's disease, dementia, or any other medical condition. It is a
  research-prototype signal, not a diagnostic tool.
- Persistence of trained models and multi-signal risk assessment are planned
  for later; this package currently only trains and evaluates in memory.
"""