"""MindKey deterministic evidence layer (Phase 2).

This package builds the ML <-> Agent contract and a read-only, deterministic
evidence layer on top of the existing data. It contains:

- ``units``: canonical units for behavioral features.
- ``contracts``: the ``BehavioralEvidence`` Pydantic contract.
- ``state``: the ``AgentState`` Pydantic state model (no orchestration).
- ``bridge``: the ML <-> Agent bridge (reuses ``ml/features.py`` validation).
- ``repository``: a read-only session repository abstraction.
- ``tools``: deterministic read-only evidence tools.
- ``adapter``: ``build_behavioral_evidence(user_id, session_id)``.

Explicit non-goals for this package (by design):

- No LLM calls, no LangGraph, and no investigation graph.
- No writes to any existing table and no changes to the typing pipeline.
- No typed content is read or processed: only structured numeric behavioral
  features and timestamps.
- No medical diagnosis and no causal claims. Anomaly scores are reported as
  numbers, never interpreted as a condition.

Importing this package does NOT require the Supabase environment to be
configured: the real repository imports the Supabase client lazily.
"""
