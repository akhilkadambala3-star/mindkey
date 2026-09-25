"""Tests for the Phase 3.2 hypothesis catalog and evaluation.

Phase 2's persistence verdict stays authoritative: H3 may only be *supported*
when the conservative eligibility gate has already passed, and the catalog
cannot be replaced by a caller.
"""

import unittest
from datetime import datetime, timezone
from typing import get_args

from pydantic import ValidationError

from investigation import tools
from investigation.adapter import build_behavioral_evidence
from investigation.agent import (
    EvidenceRegistry,
    HYPOTHESIS_CATALOG,
    InvestigationTrace,
    QUESTION_KEYS,
    QUESTION_TEXT,
    QUESTION_TOOLS,
    UNANSWERABLE_QUESTIONS,
    answerable_questions,
    assess_persistence,
    create_hypotheses,
    evaluate_hypotheses,
    is_answerable,
    question_text,
    register_from_behavioral_evidence,
    unanswerable_questions,
)
from investigation.agent.persistence import PersistenceFinding
from investigation.state import Hypothesis, HypothesisStatus
from tests import fixtures

USER = "user-1"

PHASE2_TOOLS = {
    "get_ml_evidence",
    "get_recent_sessions",
    "get_historical_baseline",
    "calculate_behavioral_drift",
    "compare_time_windows",
}

#: Phase 2's fixed non-diagnostic statement, the only allowed exemption below.
DISCLAIMER = "an anomaly score is not a diagnosis."

#: Vocabulary that must never appear in behavioral hypotheses.
CLINICAL_TERMS = (
    "parkinson",
    "dementia",
    "alzheimer",
    "depress",
    "bipolar",
    "diagnos",
    "disease",
    "disorder",
    "decline",
    "impair",
    "patient",
    "clinical",
)


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


def _stable_repo():
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
            for i in range(1, 35)
        ]
    )


def _persistent_repo():
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


def _shifted_repo():
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 36)
    ]
    sessions += [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=233.0)
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(sessions=sessions)


def _thin_repo():
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i) for i in range(3)
        ]
    )


def _ml_anomaly_only_repo():
    """Five valid sessions with a stored anomaly flag: not enough for a model."""
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(
        sessions=sessions,
        anomalies={"S0001": {"anomaly_score": 0.99, "is_anomaly": True}},
    )


def _alt_drift():
    from investigation.tools import BehavioralDriftOutput, FeatureDrift
    from investigation.units import CANONICAL_FEATURE_KEYS

    features = []
    for key in CANONICAL_FEATURE_KEYS:
        relative = -0.3 if key == "typing_speed_cpm" else 0.5
        features.append(
            FeatureDrift(
                key=key,
                label=key,
                canonical_unit="unit",
                available=True,
                baseline_value=1.0,
                recent_value=1.0 + relative,
                absolute_change=relative,
                relative_change=relative,
                direction="increase" if relative > 0 else "decrease",
            )
        )
    return BehavioralDriftOutput(
        available=True,
        user_id=USER,
        recent_window_days=14,
        baseline_window_days=14,
        recent_sample_count=10,
        baseline_sample_count=10,
        sufficient=True,
        features=features,
        note="synthetic",
    )


def _scenario(repository, session_id="R0001", **persistence_kwargs):
    evidence = build_behavioral_evidence(USER, session_id, repository=repository)
    anomaly = tools.get_ml_evidence(USER, session_id, repository=repository)
    recent = tools.get_recent_sessions(
        USER, limit=500, window_days=7, repository=repository
    )
    registry = EvidenceRegistry()
    register_from_behavioral_evidence(registry, evidence, anomaly)
    finding = assess_persistence(
        registry, evidence, recent, as_of=datetime.now(timezone.utc), **persistence_kwargs
    )
    evaluation = evaluate_hypotheses(
        registry, evidence, finding, create_hypotheses()
    )
    return registry, evidence, finding, evaluation


def _by_id(evaluation):
    return {reasoning.id: reasoning for reasoning in evaluation.reasoning}


# ---------------------------------------------------------------------------
# Catalog and question registry
# ---------------------------------------------------------------------------


class CatalogTests(unittest.TestCase):
    def test_catalog_is_exactly_four_fixed_hypotheses(self):
        self.assertEqual([spec.id for spec in HYPOTHESIS_CATALOG], ["H1", "H2", "H3", "H4"])
        self.assertEqual(
            [spec.statement for spec in HYPOTHESIS_CATALOG],
            [
                "Temporary behavioral variation",
                "Contextual disruption (sleep, fatigue, stress or workload)",
                "Persistent behavioral change",
                "Data-quality or capture artifact",
            ],
        )

    def test_catalog_uses_no_clinical_language(self):
        blob = " ".join(
            f"{spec.statement} {spec.scope}".lower() for spec in HYPOTHESIS_CATALOG
        )
        for term in CLINICAL_TERMS:
            self.assertNotIn(term, blob)

    def test_question_registry_is_consistent(self):
        self.assertEqual(set(QUESTION_TEXT), set(QUESTION_TOOLS))
        self.assertEqual(QUESTION_KEYS, tuple(QUESTION_TEXT))
        self.assertTrue(all(text for text in QUESTION_TEXT.values()))
        for key, tool_names in QUESTION_TOOLS.items():
            with self.subTest(question=key):
                self.assertTrue(set(tool_names) <= PHASE2_TOOLS)
        self.assertEqual(UNANSWERABLE_QUESTIONS, ("context_factors", "capture_change"))

    def test_question_helpers(self):
        self.assertFalse(is_answerable("context_factors"))
        self.assertFalse(is_answerable("capture_change"))
        self.assertTrue(is_answerable("window_robustness"))
        self.assertEqual(
            answerable_questions(
                ["context_factors", "window_robustness", "capture_change"]
            ),
            ["window_robustness"],
        )
        self.assertEqual(
            unanswerable_questions(
                ["context_factors", "window_robustness", "capture_change"]
            ),
            ["context_factors", "capture_change"],
        )
        self.assertEqual(
            question_text("context_factors"), QUESTION_TEXT["context_factors"]
        )
        self.assertEqual(question_text("not_a_question"), "an unregistered question")

    def test_create_hypotheses_starts_uncertain(self):
        hypotheses = create_hypotheses()
        self.assertEqual(len(hypotheses), 4)
        for hypothesis, spec in zip(hypotheses, HYPOTHESIS_CATALOG):
            self.assertEqual(hypothesis.hypothesis, spec.statement)
            self.assertEqual(hypothesis.status, "uncertain")
            self.assertEqual(hypothesis.supporting_evidence, [])
            self.assertEqual(hypothesis.contradicting_evidence, [])


# ---------------------------------------------------------------------------
# Evaluation rules
# ---------------------------------------------------------------------------


class EvaluationTests(unittest.TestCase):
    def test_persistent_scenario_supports_only_h3(self):
        _, _, _, evaluation = _scenario(_persistent_repo())
        reasoning = _by_id(evaluation)
        self.assertEqual(reasoning["H3"].status, "supported")
        self.assertEqual(reasoning["H1"].status, "weakened")
        self.assertEqual(reasoning["H2"].status, "uncertain")
        self.assertEqual(reasoning["H4"].status, "uncertain")

    def test_clean_data_refutes_the_data_quality_hypothesis(self):
        _, _, _, evaluation = _scenario(_persistent_repo(), alt_drifts=[_alt_drift()])
        reasoning = _by_id(evaluation)
        self.assertEqual(reasoning["H4"].status, "weakened")
        self.assertTrue(reasoning["H4"].contradicting_evidence)
        self.assertEqual(reasoning["H3"].status, "supported")

    def test_recent_variation_supports_temporary_variation(self):
        _, _, finding, evaluation = _scenario(_shifted_repo())
        reasoning = _by_id(evaluation)
        self.assertEqual(finding.phase2_status, "recent_variation")
        self.assertEqual(reasoning["H1"].status, "supported")
        self.assertEqual(reasoning["H3"].status, "weakened")

    def test_insufficient_data_weakens_h3_and_supports_h4(self):
        _, _, _, evaluation = _scenario(_thin_repo(), session_id="S0001")
        reasoning = _by_id(evaluation)
        self.assertEqual(reasoning["H3"].status, "weakened")
        self.assertEqual(reasoning["H4"].status, "supported")
        self.assertEqual(reasoning["H1"].status, "supported")

    def test_ml_anomaly_alone_cannot_support_h3(self):
        registry, _, _, evaluation = _scenario(_ml_anomaly_only_repo(), session_id="S0001")
        anomaly = registry.filter(kind="anomaly")
        self.assertEqual(len(anomaly), 1)
        self.assertTrue(anomaly[0].available)
        self.assertEqual(anomaly[0].status, "true")
        reasoning = _by_id(evaluation)
        self.assertNotEqual(reasoning["H3"].status, "supported")
        self.assertEqual(reasoning["H3"].status, "weakened")

    def test_thin_depth_weakens_h3_even_when_phase2_says_persistent(self):
        registry, evidence, _, _ = _scenario(_shifted_repo())
        depth_ids = [item.id for item in registry.filter(kind="persistence_depth")]
        finding = PersistenceFinding(
            phase2_status="persistent_change",
            phase2_signals_moved=4,
            phase2_supporting_sessions=5,
            eligible_for_persistence_claim=False,
            downgrade_reason="persistence_depth_insufficient",
            classification="recent_run",
            depth=2,
            minimum_recent_sessions=5,
            robustness="not_assessed",
        )
        evaluation = evaluate_hypotheses(registry, evidence, finding, create_hypotheses())
        reasoning = _by_id(evaluation)
        self.assertEqual(reasoning["H3"].status, "weakened")
        self.assertEqual(reasoning["H3"].contradicting_evidence, depth_ids)

    def test_robustness_disagreement_weakens_h3_and_supports_h4(self):
        _, _, _, evaluation = _scenario(
            _persistent_repo(), alt_drifts=[_alt_drift_reversed()]
        )
        reasoning = _by_id(evaluation)
        self.assertEqual(reasoning["H3"].status, "weakened")
        self.assertEqual(reasoning["H4"].status, "supported")

    def test_h2_is_uncertain_and_names_the_missing_context(self):
        for repository, session_id in (
            (_persistent_repo(), "R0001"),
            (_shifted_repo(), "R0001"),
            (_thin_repo(), "S0001"),
        ):
            with self.subTest(session=session_id):
                registry, _, _, evaluation = _scenario(repository, session_id)
                reasoning = _by_id(evaluation)
                self.assertEqual(reasoning["H2"].status, "uncertain")
                self.assertEqual(reasoning["H2"].supporting_evidence, [])
                self.assertIn("context_factors", reasoning["H2"].missing_evidence)
                self.assertFalse(registry.filter(kind="context_present"))

    def test_support_and_contradiction_never_overlap(self):
        _, _, _, evaluation = _scenario(_persistent_repo())
        for reasoning in evaluation.reasoning:
            with self.subTest(hypothesis=reasoning.id):
                self.assertEqual(
                    set(reasoning.supporting_evidence)
                    & set(reasoning.contradicting_evidence),
                    set(),
                )

    def test_supported_hypotheses_always_cite_evidence(self):
        _, _, _, evaluation = _scenario(_thin_repo(), session_id="S0001")
        for reasoning in evaluation.reasoning:
            if reasoning.status == "supported":
                self.assertTrue(reasoning.supporting_evidence, reasoning.id)

    def test_every_cited_id_resolves_in_the_registry(self):
        registry, _, _, evaluation = _scenario(_persistent_repo(), alt_drifts=[_alt_drift()])
        for reasoning in evaluation.reasoning:
            for item_id in reasoning.supporting_evidence + reasoning.contradicting_evidence:
                self.assertIsNotNone(registry.get(item_id), item_id)

    def test_statuses_come_from_the_phase2_literal(self):
        allowed = set(get_args(HypothesisStatus))
        _, _, _, evaluation = _scenario(_persistent_repo())
        for reasoning in evaluation.reasoning:
            self.assertIn(reasoning.status, allowed)
        with self.assertRaises(ValidationError):
            Hypothesis(hypothesis="x", status="definitely_supported")

    def test_missing_evidence_union_is_canonical_and_deduplicated(self):
        _, _, _, evaluation = _scenario(_persistent_repo())
        self.assertEqual(
            evaluation.missing_evidence, ["window_robustness", "context_factors"]
        )
        self.assertEqual(len(set(evaluation.missing_evidence)), len(evaluation.missing_evidence))

    def test_foreign_hypotheses_are_rejected(self):
        registry, evidence, finding, _ = _scenario(_shifted_repo())
        foreign = create_hypotheses()
        foreign[2].hypothesis = "Cognitive decline"
        with self.assertRaises(ValueError):
            evaluate_hypotheses(registry, evidence, finding, foreign)

        with self.assertRaises(ValueError):
            evaluate_hypotheses(registry, evidence, finding, create_hypotheses()[:3])

    def test_evaluation_is_deterministic(self):
        repository = _persistent_repo()
        evidence = build_behavioral_evidence(USER, "R0001", repository=repository)
        anomaly = tools.get_ml_evidence(USER, "R0001", repository=repository)
        recent = tools.get_recent_sessions(USER, limit=500, window_days=7, repository=repository)
        as_of = datetime.now(timezone.utc)

        results = []
        for _ in range(2):
            registry = EvidenceRegistry()
            register_from_behavioral_evidence(registry, evidence, anomaly)
            finding = assess_persistence(
                registry, evidence, recent, as_of=as_of, alt_drifts=[_alt_drift()]
            )
            results.append(
                evaluate_hypotheses(registry, evidence, finding, create_hypotheses())
            )
        self.assertEqual(results[0].model_dump(), results[1].model_dump())

    def test_trace_records_one_update_per_hypothesis(self):
        repository = _persistent_repo()
        evidence = build_behavioral_evidence(USER, "R0001", repository=repository)
        anomaly = tools.get_ml_evidence(USER, "R0001", repository=repository)
        recent = tools.get_recent_sessions(USER, limit=500, window_days=7, repository=repository)
        registry = EvidenceRegistry()
        register_from_behavioral_evidence(registry, evidence, anomaly)
        finding = assess_persistence(registry, evidence, recent, as_of=datetime.now(timezone.utc))

        trace = InvestigationTrace(clock=lambda: datetime(2026, 9, 24, 9, 42, tzinfo=timezone.utc))
        evaluate_hypotheses(registry, evidence, finding, create_hypotheses(), trace=trace)

        updates = [e for e in trace if e.event_type == "hypothesis_updated"]
        self.assertEqual(len(updates), 4)
        self.assertEqual([event.detail for event in updates], [
            "H1:weakened", "H2:uncertain", "H3:supported", "H4:uncertain",
        ])


# ---------------------------------------------------------------------------
# End-to-end (3.1 -> 3.2) scenarios
# ---------------------------------------------------------------------------


class PipelineTests(unittest.TestCase):
    def test_stable_history_never_supports_persistent_change(self):
        _, evidence, finding, evaluation = _scenario(_stable_repo(), session_id="S0001")
        reasoning = _by_id(evaluation)
        self.assertEqual(finding.classification, "no_current_deviation")
        self.assertFalse(finding.eligible_for_persistence_claim)
        self.assertNotEqual(reasoning["H3"].status, "supported")
        self.assertEqual(reasoning["H1"].status, "supported")
        self.assertTrue(evidence.data_quality.meets_model_minimum)

    def test_sustained_scenario_is_eligible_and_persistent(self):
        _, evidence, finding, evaluation = _scenario(_persistent_repo())
        reasoning = _by_id(evaluation)
        self.assertEqual(finding.classification, "sustained")
        self.assertTrue(finding.eligible_for_persistence_claim)
        self.assertEqual(reasoning["H3"].status, "supported")
        self.assertTrue(reasoning["H3"].supporting_evidence)
        self.assertEqual(
            evidence.temporal_analysis.persistence.status, "persistent_change"
        )

    def test_thin_history_never_supports_persistent_change(self):
        _, _, finding, evaluation = _scenario(_thin_repo(), session_id="S0001")
        reasoning = _by_id(evaluation)
        self.assertEqual(finding.classification, "insufficient_data")
        self.assertFalse(finding.eligible_for_persistence_claim)
        self.assertEqual(reasoning["H3"].status, "weakened")
        self.assertEqual(reasoning["H4"].status, "supported")

    def test_no_generated_text_contains_clinical_language(self):
        registry, _, finding, evaluation = _scenario(_persistent_repo())
        blob = " ".join(
            [item.statement for item in registry.all()]
            + [hypothesis.hypothesis for hypothesis in evaluation.hypotheses]
            + [finding.classification, finding.phase2_status, str(finding.downgrade_reason)]
        ).lower()

        # The one permitted clinical-looking token is Phase 2's fixed safety
        # *negation*; it must be present, and it is the only exemption.
        self.assertIn(DISCLAIMER, blob)
        blob = blob.replace(DISCLAIMER, "")
        for term in CLINICAL_TERMS:
            self.assertNotIn(term, blob)


def _alt_drift_reversed():
    """The same alternative windows with every sign flipped."""
    drift = _alt_drift()
    for feature in drift.features:
        feature.relative_change = -feature.relative_change
        feature.direction = "decrease" if feature.relative_change < 0 else "increase"
    return drift


if __name__ == "__main__":
    unittest.main()
