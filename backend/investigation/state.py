"""The ``AgentState`` Pydantic model (Phase 2).

This file defines **only** the agent's explicit state model. It contains no
graph, no nodes, no edges, no LLM calls, and no orchestration. A later phase
will add the LangGraph workflow that reads and writes this state.

The fields mirror the state skeleton in the master specification:

    trigger, ml_evidence, investigation_plan, tool_results, hypotheses,
    missing_evidence, context, final_assessment, limitations, next_action

plus bookkeeping fields the stopping/reporting requirements need
(``iteration``, ``tools_called``, ``stop_reason``).
"""

from typing import Literal

from pydantic import BaseModel, Field

HypothesisStatus = Literal["supported", "uncertain", "weakened"]


class Hypothesis(BaseModel):
    """One competing explanation under investigation.

    Evidence lists hold *references* to observations (e.g. tool-result ids or
    short factual statements), not free-form medical speculation.
    """

    hypothesis: str
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    status: HypothesisStatus = "uncertain"


class AgentState(BaseModel):
    """Explicit state container for a future investigation graph.

    All collections default to empty and all optional objects to ``None`` so a
    fresh investigation starts from a well-defined, inspectable state.
    """

    trigger: dict = Field(default_factory=dict)
    ml_evidence: dict | None = None
    investigation_plan: list[str] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    context: dict = Field(default_factory=dict)
    final_assessment: dict | None = None
    limitations: list[str] = Field(default_factory=list)
    next_action: str = ""

    # Evidence and trace artifacts (Phase 3.1). Kept as plain dicts so this
    # state model never imports the agent layer (investigation/ must not depend
    # on investigation/agent/); the agent validates them with its own Pydantic
    # models (EvidenceItem, TraceEvent).
    evidence: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)

    # Bookkeeping for stopping criteria and observability.
    iteration: int = 0
    tools_called: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
