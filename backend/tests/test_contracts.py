"""Tests for the Pydantic contract and state models."""

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from investigation.contracts import (
    SCHEMA_VERSION,
    BehavioralEvidence,
    CanonicalSession,
    ContextEvidence,
    DataQuality,
    PersistenceAssessment,
    SignalEvidence,
    TemporalAnalysis,
    TriggerInfo,
    Uncertainty,
    WindowSummary,
)
from investigation.state import AgentState, Hypothesis


def _minimal_evidence():
    return BehavioralEvidence(
        user_id="user-1",
        generated_at=datetime.now(timezone.utc),
        trigger=TriggerInfo(session_id="S0001"),
        temporal_analysis=TemporalAnalysis(
            recent=WindowSummary(label="recent", window_days=7),
            baseline=WindowSummary(label="baseline", window_days=30),
            persistence=PersistenceAssessment(status="insufficient_data"),
        ),
        data_quality=DataQuality(),
        context=ContextEvidence(),
        uncertainty=Uncertainty(),
    )


class ContractTests(unittest.TestCase):
    def test_minimal_evidence_constructs(self):
        evidence = _minimal_evidence()
        self.assertEqual(evidence.schema_version, SCHEMA_VERSION)
        self.assertEqual(evidence.signals, [])
        self.assertEqual(evidence.limitations, [])

    def test_canonical_session_defaults_to_unavailable(self):
        session = CanonicalSession()
        self.assertFalse(session.is_valid)
        self.assertIsNone(session.typing_speed_cpm)

    def test_missing_values_are_allowed(self):
        signal = SignalEvidence(
            key="typing_speed_cpm",
            label="Typing speed",
            canonical_unit="characters per minute",
            available=False,
        )
        self.assertIsNone(signal.value)

    def test_invalid_direction_is_rejected(self):
        with self.assertRaises(ValidationError):
            SignalEvidence(
                key="typing_speed_cpm",
                label="Typing speed",
                canonical_unit="characters per minute",
                available=True,
                direction="upward",
            )

    def test_invalid_persistence_status_is_rejected(self):
        with self.assertRaises(ValidationError):
            PersistenceAssessment(status="definitely_dementia")

    def test_evidence_has_no_diagnosis_field(self):
        evidence = _minimal_evidence()
        self.assertNotIn("diagnosis", evidence.model_dump())
        self.assertNotIn("condition", evidence.model_dump())


class StateTests(unittest.TestCase):
    def test_fresh_state_has_expected_defaults(self):
        state = AgentState()
        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.hypotheses, [])
        self.assertIsNone(state.stop_reason)
        self.assertEqual(state.next_action, "")

    def test_hypothesis_defaults_to_uncertain(self):
        hypothesis = Hypothesis(hypothesis="Temporary variation")
        self.assertEqual(hypothesis.status, "uncertain")
        self.assertEqual(hypothesis.supporting_evidence, [])

    def test_state_collections_are_independent(self):
        first = AgentState()
        second = AgentState()
        first.hypotheses.append(Hypothesis(hypothesis="H1"))
        self.assertEqual(second.hypotheses, [])


if __name__ == "__main__":
    unittest.main()
