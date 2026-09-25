"""Evaluation scenario catalog (Phase 3.5).

A fixed catalog of *scenarios*: a synthetic, in-memory situation plus the
outcome the committed pipeline is expected to reach for it. The scenarios
exercise the whole non-LLM pipeline -- evidence, persistence, hypotheses,
alternatives, engine, critic and report -- and are the reproducible evaluation
surface for MindKey's behavioral monitoring.

Ground rules
------------
- The Phase 2 / 3.1 / 3.2 / 3.3 / 3.4 implementation is authoritative. The
  expectations here are *characterizations* of that behavior. If one ever
  disagrees with the committed code, the code is right: this file or a
  deliberate, separately-reviewed decision must change -- never the pipeline.
- All data is synthetic and lives in memory (``tests.fixtures``). No database
  is contacted, no repository writes happen and no network is used.
- Every scenario is deterministic: the catalog order is fixed, there is no
  randomness, and the trace clock is injected so report/timeline output is
  reproducible.
- Privacy and safety: only structured numeric features, ids and timestamps are
  used. No typed content, no keystrokes, no raw session content and no medical
  language.
"""

import json
import unittest.mock
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from investigation import repository as repository_module
from investigation.agent import (
    ALT_WINDOWS,
    CLINICAL_TERMS,
    SAFETY_EXEMPTIONS,
    GroundedReport,
    InvestigationResult,
    build_grounded_report,
    run_investigation,
)
from investigation.agent import engine as engine_module
from tests import fixtures

#: The synthetic user every scenario investigates.
USER = fixtures.DEFAULT_USER

#: Deterministic trace clock used by every scenario run.
_FROZEN_NOW = datetime(2026, 9, 24, 9, 42, tzinfo=timezone.utc)


def frozen_clock():
    """A zero-argument clock pinned to a fixed instant."""
    return _FROZEN_NOW


# ---------------------------------------------------------------------------
# Synthetic repositories (mirrors of the committed unit-test fixtures)
# ---------------------------------------------------------------------------


def _stable_repo():
    """A dense, unchanging history."""
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
            for i in range(1, 34)
        ]
    )


def _shifted_repo():
    """A single-signal recent shift on top of a dense baseline."""
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 36)
    ]
    sessions += [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=233.0)
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(sessions=sessions)


def _persistent_repo():
    """A multi-signal recent shift plus an anomaly flag on the trigger."""
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 36)
    ]
    sessions += [
        fixtures.make_session(
            session_id=f"R{i:04d}",
            days_ago=i,
            typing_speed=200.0,
            dwell_mean=0.20,
            flight_mean=0.14,
            rhythm_variability=0.30,
            correction_rate=0.12,
            pause_count=9,
        )
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(
        sessions=sessions,
        anomalies={"R0001": {"anomaly_score": 0.62, "is_anomaly": True}},
    )


def _thin_repo():
    """Only three stored sessions."""
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i) for i in range(3)
        ]
    )


def _empty_repo():
    """No stored sessions at all."""
    return fixtures.FakeRepository(sessions=[])


def _anomaly_only_repo():
    """Five sessions where only the trigger carries an anomaly flag."""
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i) for i in range(1, 6)
        ],
        anomalies={"S0001": {"anomaly_score": 0.99, "is_anomaly": True}},
    )


def _invalid_repo():
    """A dense valid history plus one row that fails validation."""
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(1, 20)
    ]
    sessions.append(fixtures.make_invalid_session(session_id="BAD1", days_ago=3))
    return fixtures.FakeRepository(sessions=sessions)


def _sparse_recent_repo():
    """A dense baseline window and only two recent sessions."""
    sessions = [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(1, 3)
    ]
    sessions += [
        fixtures.make_session(session_id=f"B{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 31)
    ]
    return fixtures.FakeRepository(sessions=sessions)


def _multiplier(mult):
    """Feature values scaled by ``mult`` from a natural reference profile."""
    return {
        "typing_speed": 300.0 * mult,
        "dwell_mean": 0.12 * mult,
        "flight_mean": 0.08 * mult,
        "rhythm_variability": 0.19 * mult,
        "correction_rate": min(0.99, 0.06 * mult),
        "pause_count": max(1, round(4 * mult)),
    }


def _crossover_repo():
    """A natural dataset whose recent window and 14-day window disagree.

    Normal days 15-40, a low run on days 1-7 and a high run on days 8-14, so
    whether a change looks sustained depends on which recent window is used.
    """
    sessions = []
    for day in range(15, 41):
        sessions.append(
            fixtures.make_session(
                session_id=f"N{day:04d}", days_ago=day, **_multiplier(1.0)
            )
        )
    for day in range(1, 8):
        sessions.append(
            fixtures.make_session(
                session_id=f"L{day:04d}", days_ago=day, **_multiplier(0.2)
            )
        )
    for day in range(8, 15):
        sessions.append(
            fixtures.make_session(
                session_id=f"H{day:04d}", days_ago=day, **_multiplier(2.5)
            )
        )
    return fixtures.FakeRepository(sessions=sessions)


# ---------------------------------------------------------------------------
# Hypothesis-state shorthands (H1 temporary, H2 contextual, H3 persistent,
# H4 data-quality artifact). These repeat across scenarios by design.
# ---------------------------------------------------------------------------

_H_NO_DEVIATION = (
    ("H1", "supported"),
    ("H2", "uncertain"),
    ("H3", "weakened"),
    ("H4", "supported"),
)
_H_PERSISTENT = (
    ("H1", "weakened"),
    ("H2", "uncertain"),
    ("H3", "supported"),
    ("H4", "weakened"),
)
_H_PERSISTENT_WITH_ARTIFACT = (
    ("H1", "weakened"),
    ("H2", "uncertain"),
    ("H3", "supported"),
    ("H4", "supported"),
)
_H_UNKNOWN = (
    ("H1", "uncertain"),
    ("H2", "uncertain"),
    ("H3", "uncertain"),
    ("H4", "uncertain"),
)


# ---------------------------------------------------------------------------
# Scenario model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expectation:
    """The outcome a scenario expects from the committed pipeline.

    ``required_open_questions`` and ``required_evidence_kinds`` are *presence*
    assertions (the expectation must be contained in the observed set), because
    the engine may also surface answerable questions or extra evidence.
    """

    stop_reason: str
    persistence_status: str
    persistence_classification: str
    eligible: bool
    robustness: str
    conclusion_status: str
    conclusion_basis: str
    uncertainty_level: str
    hypothesis_statuses: tuple = ()
    required_open_questions: tuple = ()
    alternative_statuses: tuple = ()
    required_evidence_kinds: tuple = ()
    next_action_contains: str | None = None


@dataclass(frozen=True)
class Scenario:
    """One synthetic situation plus its expected outcome."""

    key: str
    title: str
    description: str
    session_id: str
    expectation: Expectation
    repository_factory: Callable = field(default=None, repr=False)
    alt_windows: tuple = ALT_WINDOWS
    default_repository_unavailable: bool = False

    @property
    def supported_hypotheses(self):
        """The hypothesis ids the expectation marks as supported."""
        return tuple(
            hypothesis_id
            for hypothesis_id, status in self.expectation.hypothesis_statuses
            if status == "supported"
        )


# ---------------------------------------------------------------------------
# The catalog (fixed order)
# ---------------------------------------------------------------------------

EVALUATION_SCENARIOS = (
    Scenario(
        key="consistent",
        title="Consistent with the personal baseline",
        description=(
            "A dense, unchanging history: recent patterns match the baseline "
            "and no deviation is reported."
        ),
        session_id="S0001",
        repository_factory=_stable_repo,
        expectation=Expectation(
            stop_reason="no_further_evidence",
            persistence_status="stable",
            persistence_classification="no_current_deviation",
            eligible=False,
            robustness="not_assessed",
            conclusion_status="no_deviation",
            conclusion_basis="stable_baseline",
            uncertainty_level="moderate",
            hypothesis_statuses=_H_NO_DEVIATION,
        ),
    ),
    Scenario(
        key="recent_variation",
        title="Recent variation that does not persist",
        description=(
            "A single-signal recent shift over a dense baseline: a deviation "
            "is visible but not enough to support a sustained change."
        ),
        session_id="R0001",
        repository_factory=_shifted_repo,
        expectation=Expectation(
            stop_reason="no_further_evidence",
            persistence_status="recent_variation",
            persistence_classification="sustained",
            eligible=False,
            robustness="not_assessed",
            conclusion_status="preliminary",
            conclusion_basis="deviation_not_persistent",
            uncertainty_level="moderate",
            hypothesis_statuses=_H_NO_DEVIATION,
        ),
    ),
    Scenario(
        key="persistent_multi_signal",
        title="Persistent multi-signal change",
        description=(
            "A multi-signal recent shift plus a stored anomaly flag: the "
            "persistence claim is eligible and the windows agree."
        ),
        session_id="R0001",
        repository_factory=_persistent_repo,
        expectation=Expectation(
            stop_reason="evidence_sufficient",
            persistence_status="persistent_change",
            persistence_classification="sustained",
            eligible=True,
            robustness="agrees",
            conclusion_status="grounded",
            conclusion_basis="grounded_persistent_change",
            uncertainty_level="moderate",
            hypothesis_statuses=_H_PERSISTENT,
            required_evidence_kinds=("anomaly",),
        ),
    ),
    Scenario(
        key="insufficient_history",
        title="Insufficient history",
        description=(
            "Only three stored sessions: below the model minimum, so no "
            "behavioral conclusion can be reached."
        ),
        session_id="S0001",
        repository_factory=_thin_repo,
        expectation=Expectation(
            stop_reason="evidence_insufficient",
            persistence_status="insufficient_data",
            persistence_classification="insufficient_data",
            eligible=False,
            robustness="not_assessed",
            conclusion_status="inconclusive",
            conclusion_basis="insufficient_history",
            uncertainty_level="high",
            hypothesis_statuses=_H_NO_DEVIATION,
            required_open_questions=("context_factors", "capture_change"),
            required_evidence_kinds=("anomaly_unavailable",),
            next_action_contains="model minimum",
        ),
    ),
    Scenario(
        key="zero_sessions",
        title="No stored sessions",
        description=(
            "An empty in-memory store: no evidence can be built, so the "
            "investigation is inconclusive and nothing is fabricated."
        ),
        session_id="S0001",
        repository_factory=_empty_repo,
        expectation=Expectation(
            stop_reason="evidence_insufficient",
            persistence_status="insufficient_data",
            persistence_classification="insufficient_data",
            eligible=False,
            robustness="not_assessed",
            conclusion_status="inconclusive",
            conclusion_basis="insufficient_history",
            uncertainty_level="high",
            hypothesis_statuses=_H_NO_DEVIATION,
            required_evidence_kinds=("signal_unavailable",),
            next_action_contains="model minimum",
        ),
    ),
    Scenario(
        key="ml_anomaly_only",
        title="Anomaly flag without supporting history",
        description=(
            "Five sessions where only the trigger is flagged anomalous: the "
            "flag alone never supports a persistent-change claim."
        ),
        session_id="S0001",
        repository_factory=_anomaly_only_repo,
        expectation=Expectation(
            stop_reason="evidence_insufficient",
            persistence_status="insufficient_data",
            persistence_classification="no_current_deviation",
            eligible=False,
            robustness="not_assessed",
            conclusion_status="inconclusive",
            conclusion_basis="insufficient_history",
            uncertainty_level="high",
            hypothesis_statuses=_H_NO_DEVIATION,
            required_evidence_kinds=("anomaly",),
            next_action_contains="model minimum",
        ),
    ),
    Scenario(
        key="invalid_rows",
        title="One invalid stored row",
        description=(
            "A dense valid history plus a row that fails validation: the "
            "invalid row is reported as a data-quality artifact, not evidence."
        ),
        session_id="S0001",
        repository_factory=_invalid_repo,
        expectation=Expectation(
            stop_reason="no_further_evidence",
            persistence_status="stable",
            persistence_classification="no_current_deviation",
            eligible=False,
            robustness="not_assessed",
            conclusion_status="no_deviation",
            conclusion_basis="stable_baseline",
            uncertainty_level="moderate",
            hypothesis_statuses=_H_NO_DEVIATION,
            required_evidence_kinds=("quality_invalid",),
        ),
    ),
    Scenario(
        key="sparse_recent",
        title="Sparse recent sampling",
        description=(
            "A dense baseline window and only two recent sessions: the "
            "recording cadence itself is a candidate explanation."
        ),
        session_id="R0001",
        repository_factory=_sparse_recent_repo,
        expectation=Expectation(
            stop_reason="no_further_evidence",
            persistence_status="stable",
            persistence_classification="insufficient_data",
            eligible=False,
            robustness="not_assessed",
            conclusion_status="no_deviation",
            conclusion_basis="stable_baseline",
            uncertainty_level="moderate",
            hypothesis_statuses=_H_NO_DEVIATION,
            alternative_statuses=(
                ("unusual_workload_or_schedule", "partially_evaluated"),
            ),
        ),
    ),
    Scenario(
        key="robustness_disagreement",
        title="Windows disagree about persistence",
        description=(
            "Recent and 14-day comparison windows disagree, so the "
            "persistence claim is blocked and the report stays preliminary."
        ),
        session_id="L0001",
        repository_factory=_crossover_repo,
        expectation=Expectation(
            stop_reason="only_unanswerable_questions_remain",
            persistence_status="persistent_change",
            persistence_classification="sustained",
            eligible=False,
            robustness="disagrees",
            conclusion_status="preliminary",
            conclusion_basis="deviation_not_persistent",
            uncertainty_level="moderate",
            hypothesis_statuses=_H_NO_DEVIATION,
            required_open_questions=("context_factors", "capture_change"),
            next_action_contains="user-reported context",
        ),
    ),
    Scenario(
        key="robustness_agreement",
        title="Windows agree about persistence",
        description=(
            "The same crossover dataset compared with different windows: the "
            "windows now agree and the persistence claim becomes eligible."
        ),
        session_id="L0001",
        repository_factory=_crossover_repo,
        alt_windows=((7, 30),),
        expectation=Expectation(
            stop_reason="evidence_sufficient",
            persistence_status="persistent_change",
            persistence_classification="sustained",
            eligible=True,
            robustness="agrees",
            conclusion_status="grounded",
            conclusion_basis="grounded_persistent_change",
            uncertainty_level="moderate",
            hypothesis_statuses=_H_PERSISTENT_WITH_ARTIFACT,
        ),
    ),
    Scenario(
        key="repository_unavailable",
        title="Stored sessions cannot be read",
        description=(
            "The production repository cannot be constructed: the "
            "investigation stops as data-unavailable and fabricates nothing."
        ),
        session_id="S0001",
        repository_factory=None,
        default_repository_unavailable=True,
        expectation=Expectation(
            stop_reason="data_unavailable",
            persistence_status="unknown",
            persistence_classification="insufficient_data",
            eligible=False,
            robustness="not_assessed",
            conclusion_status="inconclusive",
            conclusion_basis="data_unavailable",
            uncertainty_level="high",
            hypothesis_statuses=_H_UNKNOWN,
            next_action_contains="Restore read access",
        ),
    ),
)


def scenario_keys():
    """The scenario keys, in fixed catalog order."""
    return tuple(scenario.key for scenario in EVALUATION_SCENARIOS)


def get_scenario(key):
    """Look up one scenario by key."""
    for scenario in EVALUATION_SCENARIOS:
        if scenario.key == key:
            return scenario
    raise KeyError(key)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioOutcome:
    """The observed outcome of one scenario run."""

    scenario: Scenario
    as_of: datetime
    result: InvestigationResult
    report: GroundedReport

    @property
    def stop_reason(self):
        return self.result.stop_reason

    @property
    def persistence(self):
        return (self.result.final_assessment or {}).get("persistence") or {}

    @property
    def persistence_status(self):
        return self.persistence.get("status")

    @property
    def persistence_classification(self):
        return self.persistence.get("classification")

    @property
    def eligible(self):
        return bool(self.persistence.get("eligible"))

    @property
    def robustness(self):
        return self.persistence.get("robustness")

    @property
    def conclusion_status(self):
        return self.report.conclusion.status

    @property
    def conclusion_basis(self):
        return self.report.conclusion.basis

    @property
    def uncertainty_level(self):
        return self.report.uncertainty.level

    @property
    def hypothesis_statuses(self):
        return {claim.id: claim.status for claim in self.report.hypothesis_assessment}

    @property
    def supported_hypotheses(self):
        return tuple(
            claim.id
            for claim in self.report.hypothesis_assessment
            if claim.status == "supported"
        )

    @property
    def open_question_keys(self):
        return tuple(
            entry.get("key")
            for entry in (self.result.final_assessment or {}).get("open_questions") or []
        )

    @property
    def alternative_statuses(self):
        return {item.candidate: item.status for item in self.result.alternatives or []}

    @property
    def evidence_kinds(self):
        return frozenset(item.get("kind") for item in self.result.state.evidence)

    @property
    def next_action(self):
        return self.result.state.next_action or ""

    @property
    def timeline(self):
        from investigation.agent import render_timeline_for

        return tuple(render_timeline_for(self.report, self.result, clock=frozen_clock))

    @property
    def report_lines(self):
        from investigation.agent import render_report_lines

        return tuple(render_report_lines(self.report))

    def render_text(self):
        """The scenario's report plus timeline as one plain-text string."""
        from investigation.agent import render

        return render(self.report, self.result)


def run_scenario(scenario, *, clock=frozen_clock, as_of=None):
    """Run one scenario against the committed pipeline and return its outcome.

    Args:
        scenario: a :class:`Scenario`.
        clock: the trace clock; defaults to a frozen one for determinism.
        as_of: reference time for windowed analysis. Defaults to the wall clock
            so the synthetic sessions (built relative to now) stay aligned.

    The repository is always read-only and in memory; no network, database or
    environment variable is involved. The one exception is the outage
    scenario, which replaces the production repository factory with one that
    raises, so nothing is ever contacted.
    """
    reference = datetime.now(timezone.utc) if as_of is None else as_of

    if scenario.default_repository_unavailable:
        with unittest.mock.patch.object(
            engine_module,
            "default_repository",
            side_effect=repository_module.RepositoryError("scenario: store unavailable"),
        ):
            result = run_investigation(
                USER,
                scenario.session_id,
                clock=clock,
                as_of=reference,
                alt_windows=scenario.alt_windows,
            )
    else:
        result = run_investigation(
            USER,
            scenario.session_id,
            repository=scenario.repository_factory(),
            clock=clock,
            as_of=reference,
            alt_windows=scenario.alt_windows,
        )

    report = build_grounded_report(result, clock=clock)
    return ScenarioOutcome(
        scenario=scenario, as_of=reference, result=result, report=report
    )


def run_all(*, clock=frozen_clock):
    """Run every scenario, in catalog order."""
    return tuple(run_scenario(scenario, clock=clock) for scenario in EVALUATION_SCENARIOS)


# ---------------------------------------------------------------------------
# Privacy / safety scanning helpers (shared by the Phase 3.5 tests)
# ---------------------------------------------------------------------------

#: Stored-row column names. These are internal storage details; user-visible
#: text must describe behavior, never echo raw columns or row fields.
RAW_FIELD_NAMES = (
    "dwell_mean",
    "flight_mean",
    "typing_speed",
    "correction_rate",
    "rhythm_variability",
    "pause_count",
    "session_start",
    "session_end",
    "keystroke",
    "typed_text",
    "password",
    "raw_session",
)


def clinical_terms_in(text):
    """Clinical denylist terms in ``text``, after removing fixed negations.

    Mirrors the critic's own convention: the Phase 2 safety negations are
    stripped first, then the closed denylist is scanned.
    """
    cleaned = str(text or "").lower()
    for phrase in SAFETY_EXEMPTIONS:
        cleaned = cleaned.replace(phrase, "")
    return tuple(term for term in CLINICAL_TERMS if term in cleaned)


def raw_field_names_in(text):
    """Raw stored-column names present in ``text``."""
    return tuple(name for name in RAW_FIELD_NAMES if name in str(text or ""))


def outcome_texts(outcome):
    """Every user-visible or structured surface produced by one outcome."""
    return {
        "rendered": outcome.render_text(),
        "report": outcome.report.model_dump_json(),
        "trace": json.dumps(outcome.result.state.trace, default=str),
        "assessment": json.dumps(outcome.result.final_assessment, default=str),
        "observations": json.dumps(
            [item.model_dump(mode="json") for item in outcome.report.observations],
            default=str,
        ),
    }
