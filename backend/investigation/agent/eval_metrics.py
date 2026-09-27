"""Deterministic quantitative evaluation over the investigation scenario catalog.

Why this exists
---------------
The evaluation catalog (``tests/agent/scenarios.py``) pins *behavioral*
expectations for 11 scenarios. This module adds the quantitative layer:
reproducible rates and counts derived **only** from data the committed
pipeline already produces. Nothing here re-runs business logic, invents a
score, or weakens a scenario expectation.

Design rules
------------
- Pure functions over already-computed outcomes. A "run outcome" is any object
  exposing ``result`` (an ``InvestigationResult``), ``report`` (a
  ``GroundedReport``) and ``scenario`` (with a ``key``) -- in production the
  catalog's ``ScenarioOutcome``. This module performs no I/O, constructs no
  repository, and never imports from ``tests`` (layering rule).
- Every formula states its numerator and denominator. A metric that cannot be
  measured from the outputs is reported as ``None``, never guessed.
- Deterministic: no clocks, no randomness; the same outcomes always produce
  the same numbers. JSON-serializable via :func:`evaluate_scenarios`.

Formulas (exact)
----------------
unsupported_claim_rate
    numerator   = len(report.rejected_claims)
    denominator = len(report.rejected_claims) + len(report.observations)
                  + len(report.hypothesis_assessment) + len(report.alternatives)
    i.e. rejected claims divided by every claim the critic examined (asserted
    plus rejected). A sound run is 0.0 by construction.

unnecessary_tool_call_rate (duplicate_tool_call_rate)
    numerator   = tool_call trace events whose (tool, args_fingerprint) pair
                  already appeared earlier in the same trace
    denominator = all tool_call trace events
    The architecture suppresses duplicates by fingerprint, so the rate is a
    regression guard (0.0); any future logic that wastes calls moves it.

contradiction_detection_rate
    denominator = the scenarios listed in ``CONTRADICTION_SCENARIOS`` (the
    catalog situations that contain conflicting evidence).
    numerator   = those where the conflict is *detected and contained*:
                  the conflict marker appears in the report's uncertainty
                  reasons / evidence AND the conclusion is not "grounded".
    Detection predicates are documented per scenario below.

guardrail_trigger_rate
    denominator = the scenarios listed in ``GUARDRAIL_SCENARIOS`` (situations
    that exist to exercise a guardrail).
    numerator   = those where the guardrail actually fired: the stop reason is
                  a bounded/gated stop (``evidence_insufficient``,
                  ``data_unavailable``, ``iteration_limit``,
                  ``tool_budget_exhausted``,
                  ``only_unanswerable_questions_remain``) or the robustness
                  gate blocked the persistence claim.

failure_recovery_rate
    denominator = the scenarios listed in ``FAILURE_SCENARIOS`` (situations
    where a data source fails).
    numerator   = those that recover safely: the run completes (never raises),
                  stops with an honest reason, and fabricates nothing.

traceability_rate
    numerator   = decided claims (every observation, plus hypotheses and
                  alternatives with a decided status) whose cited evidence ids
                  all resolve in the run's evidence registry
    denominator = the number of decided claims
    Undecided (``uncertain``/``unavailable``) claims legitimately cite no
    evidence and are excluded from the denominator, not counted as failures.

tool-call counts
    mean_tool_calls, max_tool_calls, min_tool_calls and the per-scenario
    counts, over the given outcomes.
"""

from typing import Any, Iterable

#: Version of the metrics payload shape.
EVAL_METRICS_SCHEMA_VERSION = "1.0"

#: Catalog scenarios whose situation contains conflicting evidence.
CONTRADICTION_SCENARIOS: tuple[str, ...] = (
    "robustness_disagreement",
    "ml_anomaly_only",
)

#: Catalog scenarios that exist to exercise a guardrail.
GUARDRAIL_SCENARIOS: tuple[str, ...] = (
    "insufficient_history",
    "zero_sessions",
    "ml_anomaly_only",
    "robustness_disagreement",
    "repository_unavailable",
)

#: Catalog scenarios where a data source fails and recovery must be safe.
FAILURE_SCENARIOS: tuple[str, ...] = (
    "repository_unavailable",
)

#: Stop reasons that mean a guardrail (bound or gate) fired. ``no_further_evidence``
#: and ``evidence_sufficient`` are ordinary completions, not guardrail triggers.
GUARDRAIL_STOP_REASONS: frozenset[str] = frozenset(
    {
        "evidence_insufficient",
        "data_unavailable",
        "iteration_limit",
        "tool_budget_exhausted",
        "only_unanswerable_questions_remain",
    }
)

#: Alternative/hypothesis statuses that make a claim "decided" for the
#: traceability denominator.
DECIDED_STATUSES: frozenset[str] = frozenset(
    {"supported", "weakened", "partially_evaluated"}
)


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------


def unsupported_claim_rate(report) -> float:
    """Rejected claims / all claims the critic examined (asserted + rejected)."""
    rejected = len(report.rejected_claims)
    examined = (
        rejected
        + len(report.observations)
        + len(report.hypothesis_assessment)
        + len(report.alternatives)
    )
    if examined == 0:
        return 0.0
    return rejected / examined


def duplicate_tool_call_rate(result) -> float:
    """Repeated (tool, fingerprint) tool_call events / all tool_call events."""
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    total = 0
    for event in result.state.trace:
        if event.get("event_type") != "tool_call":
            continue
        total += 1
        key = (event.get("tool"), event.get("args_fingerprint"))
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    if total == 0:
        return 0.0
    return duplicates / total


def traceability_rate(result, report) -> float:
    """Decided claims whose evidence ids resolve / decided claims."""
    registry_ids = {item.get("id") for item in result.state.evidence}

    decided = 0
    resolvable = 0

    for claim in report.observations:
        decided += 1
        if claim.evidence_ids and set(claim.evidence_ids) <= registry_ids:
            resolvable += 1

    for claim in report.hypothesis_assessment:
        if claim.status not in DECIDED_STATUSES:
            continue
        decided += 1
        if claim.evidence_ids and set(claim.evidence_ids) <= registry_ids:
            resolvable += 1

    for claim in report.alternatives:
        if claim.status not in DECIDED_STATUSES:
            continue
        decided += 1
        if claim.evidence_ids and set(claim.evidence_ids) <= registry_ids:
            resolvable += 1

    if decided == 0:
        return 1.0
    return resolvable / decided


# ---------------------------------------------------------------------------
# Catalog-level predicates (documented, deterministic)
# ---------------------------------------------------------------------------


def contradiction_detected(outcome) -> bool:
    """True when a conflicting-evidence situation is detected and contained.

    Predicates, by scenario key:

    - ``robustness_disagreement``: the robustness finding says ``disagrees``,
      the report's uncertainty reasons name ``window_robustness_disagrees``,
      and the conclusion is not "grounded".
    - ``ml_anomaly_only``: the stored anomaly flag is present as evidence
      while the history is below the model minimum, the persistence claim is
      blocked, and the conclusion is inconclusive on ``insufficient_history``.
    """
    key = outcome.scenario.key
    if key not in ("robustness_disagreement", "ml_anomaly_only"):
        return False
    report = outcome.report
    reasons = set(report.uncertainty.reasons)
    not_grounded = report.conclusion.status != "grounded"

    if key == "robustness_disagreement":
        return (
            outcome.robustness == "disagrees"
            and "window_robustness_disagrees" in reasons
            and not_grounded
        )
    if key == "ml_anomaly_only":
        kinds = {item.get("kind") for item in outcome.result.state.evidence}
        return (
            "anomaly" in kinds
            and not outcome.eligible
            and report.conclusion.status == "inconclusive"
            and report.conclusion.basis == "insufficient_history"
        )
    return False


def guardrail_triggered(outcome) -> bool:
    """True when a bound or gate actually fired for a guardrail scenario."""
    key = outcome.scenario.key
    if key not in GUARDRAIL_SCENARIOS:
        return False
    if outcome.stop_reason in GUARDRAIL_STOP_REASONS:
        return True
    # The robustness gate fired when disagreement blocked the persistence claim.
    if key == "robustness_disagreement" and outcome.robustness == "disagrees":
        return True
    return False


def failure_recovered(outcome) -> bool:
    """True when a failing-source scenario degrades safely, fabricating nothing."""
    key = outcome.scenario.key
    if key not in FAILURE_SCENARIOS:
        return False
    report = outcome.report
    fabricated = [
        item
        for item in outcome.result.state.evidence
        if item.get("available", True)
    ]
    return (
        outcome.stop_reason == "data_unavailable"
        and report.conclusion.status == "inconclusive"
        and report.conclusion.basis == "data_unavailable"
        and not fabricated
        and bool(outcome.result.state.next_action)
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def tool_call_counts(outcomes: Iterable[Any]) -> dict:
    """Mean / max / min / per-scenario executed tool-call counts."""
    per_scenario = {
        outcome.scenario.key: len(outcome.result.tool_calls) for outcome in outcomes
    }
    counts = list(per_scenario.values())
    return {
        "per_scenario": per_scenario,
        "mean_tool_calls": sum(counts) / len(counts) if counts else None,
        "max_tool_calls": max(counts) if counts else None,
        "min_tool_calls": min(counts) if counts else None,
    }


def evaluate_scenarios(outcomes: Iterable[Any]) -> dict:
    """Compute every metric over a set of run outcomes.

    Args:
        outcomes: run outcomes (e.g. ``tests.agent.scenarios.run_all()``).

    Returns:
        A JSON-serializable dict: ``schema_version``, the per-scenario table
        (stop reason, conclusion, counts, rates) and the aggregate metrics
        with their numerators/denominators. Deterministic: identical outcomes
        produce an identical payload.
    """
    outcomes = list(outcomes)

    per_scenario = []
    for outcome in outcomes:
        result = outcome.result
        report = outcome.report
        per_scenario.append(
            {
                "scenario": outcome.scenario.key,
                "stop_reason": result.stop_reason,
                "tool_calls": len(result.tool_calls),
                "evidence_items": len(result.state.evidence),
                "unsupported_claim_rate": unsupported_claim_rate(report),
                "duplicate_tool_call_rate": duplicate_tool_call_rate(result),
                "traceability_rate": traceability_rate(result, report),
                "contradiction_detected": contradiction_detected(outcome),
                "guardrail_triggered": guardrail_triggered(outcome),
                "failure_recovered": failure_recovered(outcome),
            }
        )

    by_key = {outcome.scenario.key: outcome for outcome in outcomes}
    present = set(by_key)

    # Rates are measured over the group scenarios actually present in the
    # outcomes, so a subset catalog reports what it ran -- nothing more.
    contradiction_present = [key for key in CONTRADICTION_SCENARIOS if key in present]
    contradiction_hits = sum(
        1 for key in contradiction_present if contradiction_detected(by_key[key])
    )
    guardrail_present = [key for key in GUARDRAIL_SCENARIOS if key in present]
    guardrail_hits = sum(
        1 for key in guardrail_present if guardrail_triggered(by_key[key])
    )
    recovery_present = [key for key in FAILURE_SCENARIOS if key in present]
    recovery_hits = sum(
        1 for key in recovery_present if failure_recovered(by_key[key])
    )

    all_unsupported = [entry["unsupported_claim_rate"] for entry in per_scenario]
    all_duplicates = [entry["duplicate_tool_call_rate"] for entry in per_scenario]
    all_traceability = [entry["traceability_rate"] for entry in per_scenario]

    return {
        "schema_version": EVAL_METRICS_SCHEMA_VERSION,
        "scenarios": per_scenario,
        "unsupported_claim_rate": (
            sum(all_unsupported) / len(all_unsupported) if all_unsupported else None
        ),
        "unnecessary_tool_call_rate": (
            sum(all_duplicates) / len(all_duplicates) if all_duplicates else None
        ),
        "mean_tool_calls": tool_call_counts(outcomes)["mean_tool_calls"],
        "max_tool_calls": tool_call_counts(outcomes)["max_tool_calls"],
        "min_tool_calls": tool_call_counts(outcomes)["min_tool_calls"],
        "per_scenario_tool_calls": tool_call_counts(outcomes)["per_scenario"],
        "contradiction_detection_rate": _rate(
            contradiction_hits, len(contradiction_present)
        ),
        "contradiction_detection_detail": {
            "numerator": contradiction_hits,
            "denominator": len(contradiction_present),
            "scenarios": list(CONTRADICTION_SCENARIOS),
            "scenarios_present": contradiction_present,
        },
        "guardrail_trigger_rate": _rate(guardrail_hits, len(guardrail_present)),
        "guardrail_trigger_detail": {
            "numerator": guardrail_hits,
            "denominator": len(guardrail_present),
            "scenarios": list(GUARDRAIL_SCENARIOS),
            "scenarios_present": guardrail_present,
        },
        "failure_recovery_rate": _rate(recovery_hits, len(recovery_present)),
        "failure_recovery_detail": {
            "numerator": recovery_hits,
            "denominator": len(recovery_present),
            "scenarios": list(FAILURE_SCENARIOS),
            "scenarios_present": recovery_present,
        },
        "traceability_rate": (
            sum(all_traceability) / len(all_traceability) if all_traceability else None
        ),
    }
